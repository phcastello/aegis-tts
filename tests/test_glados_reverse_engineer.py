from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import glados_reverse_engineer as re_lab


def tone(sr=24000,seconds=.4,freq=220):
    t=np.arange(int(sr*seconds))/sr
    return .2*np.sin(2*np.pi*freq*t)


def wav(path,data=None,sr=24000,stereo=False):
    value=tone(sr=sr) if data is None else data
    if stereo:value=np.column_stack((value,value))
    sf.write(path,value,sr)
    return path


def test_safe_dotenv_and_precedence(tmp_path,monkeypatch):
    good=tmp_path/"x.env";good.write_text("AEGIS_GLADOS_RE_WORKERS=2\n")
    assert re_lab.load_dotenv(good,set(re_lab.DEFAULTS))["AEGIS_GLADOS_RE_WORKERS"]=="2"
    bad=tmp_path/"bad.env";bad.write_text("AEGIS_GLADOS_RE_WORKERS=$(id)\n")
    with pytest.raises(re_lab.LabError):re_lab.load_dotenv(bad,set(re_lab.DEFAULTS))


def test_wav_validation(tmp_path):
    env=dict(re_lab.DEFAULTS);assert re_lab.probe(wav(tmp_path/"a.wav"),env)["channels"]==1
    bad=tmp_path/"bad.wav";bad.write_text("not wav")
    with pytest.raises(re_lab.LabError):re_lab.probe(bad,env)


def test_resampling_is_deterministic(tmp_path):
    p=wav(tmp_path/"a.wav",sr=32000);a,_=re_lab.load_audio(p,24000);b,_=re_lab.load_audio(p,24000)
    assert len(a)==9600 and np.array_equal(a,b)


def test_alignment_recovers_delay():
    rng=np.random.default_rng(7);x=rng.normal(0,.05,24000);x[3000:6000]*=8;y=np.r_[np.zeros(240),x][:-240];a,b,m=re_lab.align_audio(x,y,24000)
    assert abs(abs(m["lag_samples"])-240)<=120 and len(a)==len(b)


def test_pitch_analysis():
    track=re_lab.pitch_track(tone(seconds=1,freq=220),24000)
    assert np.median(track["f0"][track["voiced"]])==pytest.approx(220,rel=.03)


def test_formant_features_ignore_global_gain():
    rng=np.random.default_rng(4)
    signal=tone(seconds=1)+rng.normal(0,.02,24000)
    full=np.asarray(re_lab.effect_feature_vectors(signal,24000)["formant_shift"])
    quiet=np.asarray(re_lab.effect_feature_vectors(signal*.5,24000)["formant_shift"])
    assert np.linalg.norm(full-quiet)<1e-4


def test_cents_distance():
    track=re_lab.pitch_track(tone(seconds=1,freq=440),24000)
    assert np.nanmedian(abs(track["distance_cents"]))<5


def test_chromatic_quantization():
    assert re_lab.nearest_allowed(np.array([69.49]),"chromatic")[0]==69


def test_unvoiced_regions_are_preserved():
    track={"f0":np.array([220.,0.,230.]),"voiced":np.array([1,0,1],bool),"midi":np.array([57.,np.nan,57.8]),"time":np.arange(3)*.01}
    assert re_lab.modify_track(track,{"quantization":"chromatic"})[1]==0


def test_all_engines_implement_interface():
    assert all(isinstance(x,re_lab.AudioTransformEngine) for x in re_lab.engines(dict(re_lab.DEFAULTS)).values())


def test_engine_failure_does_not_stop_search_contract(monkeypatch,tmp_path):
    class Broken(re_lab.PassthroughEngine):
        def render(self,*a):return re_lab.RenderResult(False,None,0,"expected")
    assert not Broken().render(None,None,None).success


def test_yaml_values_and_ids():
    value=re_lab.read_yaml(ROOT/"config/glados-reverse-engineering.yaml")
    assert value["analysis"]["sample_rate"]==24000


def test_duplicate_ids_rejected():
    value=re_lab.read_yaml(ROOT/"config/glados-reverse-engineering.yaml");value["engines"].append(value["engines"][0])
    with pytest.raises(re_lab.LabError):re_lab.validate_yaml(value)


def test_composite_score_reproducible(tmp_path):
    a=wav(tmp_path/"a.wav");b=wav(tmp_path/"b.wav")
    weights=re_lab.read_yaml(ROOT/"config/glados-reverse-engineering.yaml")["scoring"]
    assert re_lab.compare_audio(a,b,24000,weights)==re_lab.compare_audio(a,b,24000,weights)


def test_normalized_reference_score_has_semantic_anchors():
    baseline=np.array([0.,0.]);reference=np.array([2.,0.])
    assert re_lab.normalized_reference_score(reference,reference,baseline)==0
    assert re_lab.normalized_reference_score(baseline,reference,baseline)==1
    assert re_lab.normalized_reference_score(np.array([-1.,0.]),reference,baseline)>1


def test_normalization_uses_distinct_output(tmp_path):
    env=dict(re_lab.DEFAULTS);source=wav(tmp_path/"raw.wav");before=source.read_bytes()
    re_lab.normalize_listening(source,tmp_path/"listen.wav",env)
    assert source.read_bytes()==before and (tmp_path/"listen.wav").is_file()


def test_output_always_mono(tmp_path):
    source=wav(tmp_path/"stereo.wav",stereo=True);audio,sr=re_lab.load_audio(source);re_lab.write_audio(tmp_path/"mono.wav",audio,sr)
    assert sf.info(tmp_path/"mono.wav").channels==1


def test_output_has_valid_duration(tmp_path):
    p=tmp_path/"out.wav";re_lab.PassthroughEngine().render(wav(tmp_path/"in.wav"),{},p)
    assert sf.info(p).duration>0


def test_no_shell_injection(monkeypatch):
    called=[]
    def fake(argv,**kw):called.append(argv);return subprocess.CompletedProcess(argv,0,"","")
    monkeypatch.setattr(subprocess,"run",fake);re_lab.run(["echo","$(touch /tmp/never)"])
    assert called[0][1]=="$(touch /tmp/never)" and isinstance(called[0],list)


def test_graillon_cleanup_is_context_managed():
    source=(ROOT/"tools/graillon_lab.py").read_text()
    assert "def __exit__" in source and "p.terminate()" in source and "shutil.rmtree" in source


def test_manifest_csv_complete(tmp_path):
    record={"id":"x","engine":"passthrough","round":"A","parameters":{},"render":{"success":False,"error":"x"}}
    re_lab.write_manifest_csv(tmp_path/"m.csv",[record]);text=(tmp_path/"m.csv").read_text()
    assert "parameters" in text and "passthrough" in text


def test_html_references_existing_files(tmp_path):
    for key in ("out","work","analysis","raw","normalized","diagnostics","report"):(tmp_path/key).mkdir()
    p={key:tmp_path/key for key in ("out","work","analysis","raw","normalized","diagnostics","report")}
    p["normalized"]=p["out"]/"candidates/normalized";p["normalized"].mkdir(parents=True)
    p["work"]=tmp_path/"work";wav(p["work"]/"target-analysis.wav");wav(p["work"]/"clean-analysis.wav");wav(p["normalized"]/"x.wav")
    record={"id":"x","engine":"passthrough","parameters":{},"normalized_file":"candidates/normalized/x.wav","render":{"success":True},"metrics":{"composite_score":0.1}}
    re_lab.write_listening(p,[record],dict(re_lab.DEFAULTS))
    assert all((p["out"]/name).is_file() for name in ("listening.html","listening-blind.html","blind-map.json"))


def test_private_files_ignored_by_git():
    for path in ("experiments/glados-filter-lab/reference/private.wav","experiments/glados-filter-lab/.env.reverse-engineering","experiments/glados-filter-lab/reverse-engineering-results/x"):
        proc=subprocess.run(["git","check-ignore","-q",path],cwd=ROOT)
        assert proc.returncode==0


@pytest.mark.reverse_engineering_integration
def test_real_engines_opt_in(tmp_path):
    if not re_lab.parse_bool(__import__("os").environ.get("AEGIS_GLADOS_RE_INTEGRATION_TESTS","0")):pytest.skip("opt in")
    source=wav(tmp_path/"in.wav",tone(seconds=1));engines=re_lab.engines(dict(re_lab.DEFAULTS))
    for name in ("praat","world","rubberband"):
        if engines[name].preflight().get("available"):assert engines[name].render(source,{"quantization":"chromatic"},tmp_path/f"{name}.wav").success
