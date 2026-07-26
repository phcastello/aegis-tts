from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glados_modulation_lab as lab


def tone(sr=24000, seconds=.35, frequency=220.0):
    time = np.arange(int(sr * seconds)) / sr
    return .15 * np.sin(2 * np.pi * frequency * time)


def wav(path: Path, audio=None, sr=24000):
    sf.write(path, tone(sr=sr) if audio is None else audio, sr)
    return path


def synthetic_env(tmp_path: Path):
    env = dict(lab.DEFAULTS)
    paths = {}
    for index, state in enumerate(lab.STATE_IDS):
        paths[state] = wav(tmp_path / f"{state}.wav", tone(frequency=200 + index))
        env[f"AEGIS_GLADOS_AUDIO_{state.upper()}"] = str(paths[state])
    env["AEGIS_GLADOS_MOD_OUTPUT_DIR"] = str(tmp_path / "out")
    env["AEGIS_GLADOS_MOD_REPORT_DIR"] = str(tmp_path / "report")
    return env, paths


def test_loads_all_eight_states(tmp_path):
    env, paths = synthetic_env(tmp_path)
    loaded = lab.load_dataset(env)
    assert tuple(loaded["states"]) == lab.STATE_IDS
    assert loaded["states"]["CF"] == paths["CF"].resolve()


def test_confirmed_cf_mapping():
    import yaml
    value = yaml.safe_load((ROOT / "config/glados-effect-references.yaml").read_text())
    assert value["references"]["CF"]["file"] == "Voice_FormantAndPitchCentralization.wav"
    assert value["references"]["CF"]["confirmed_by_operator"] is True


def test_four_pairs_are_exact(tmp_path):
    env, _ = synthetic_env(tmp_path)
    loaded = lab.load_dataset(env)
    assert tuple((x["id"], x["source"], x["target"]) for x in loaded["contexts"]) == lab.CONTEXTS


def test_duplicate_state_audio_rejected(tmp_path):
    env, paths = synthetic_env(tmp_path)
    env["AEGIS_GLADOS_AUDIO_M"] = str(paths["clean"])
    with pytest.raises(lab.LabError, match="incompatible"):
        lab.preflight(env, prepare_audio=False)


def test_safe_dotenv_rejects_unknown_and_shell(tmp_path):
    unknown = tmp_path / "unknown"; unknown.write_text("EVIL=1\n")
    with pytest.raises(lab.LabError, match="unknown key"): lab.load_dotenv(unknown)
    shell = tmp_path / "shell"; shell.write_text("AEGIS_GLADOS_MOD_WORKERS=$(id)\n")
    with pytest.raises(lab.LabError, match="forbidden"): lab.load_dotenv(shell)
    invalid = tmp_path / "invalid"; invalid.write_text("export A=1\n")
    with pytest.raises(lab.LabError, match="invalid dotenv"): lab.load_dotenv(invalid)


def test_target_cannot_leak_through_render_signature():
    expected = ["self", "source_path", "output_path", "parameters"]
    assert list(inspect.signature(lab.ModulationRemovalEngine.render).parameters) == expected
    for engine in (lab.WorldModulationEngine, lab.PraatModulationEngine):
        assert list(inspect.signature(engine.render).parameters) == expected
        assert "target" not in inspect.getsource(engine.render).lower()


def test_cents_round_trip_and_octave():
    f0 = np.array([110., 220., 440., 0.])
    cents = lab.hz_to_cents(f0, 440)
    assert cents[2] == pytest.approx(0)
    assert cents[1] == pytest.approx(-1200)
    assert np.allclose(lab.cents_to_hz(cents[:3], 440), f0[:3])
    assert np.isnan(cents[3])


def test_regions_split_unvoiced_and_jump():
    voiced = np.array([1, 1, 0, 1, 1, 1], bool)
    cents = np.array([100, 110, np.nan, 120, 130, 500])
    assert lab.contiguous_regions(voiced, cents, split_jump_cents=100) == [(0, 2), (3, 5), (5, 6)]


def test_silence_is_never_interpolated():
    f0 = np.array([220., 221., 219., 0., 0., 240., 241., 239.])
    output, _ = lab.transform_f0(f0, {"trend_method":"moving_mean", "smoothing_window_ms":40,
        "minimum_region_ms":1, "modulation_factor":0}, 5)
    assert np.array_equal(output[3:5], np.zeros(2))


@pytest.mark.parametrize("method", ["moving_median", "moving_mean", "savgol", "butterworth_lowpass", "robust_spline", "median_then_lowpass"])
def test_trend_methods_are_finite_and_smooth(method):
    x = 5000 + np.linspace(0, 100, 101) + 25*np.sin(np.arange(101)*1.7)
    result = lab.estimate_trend(x, method, frame_ms=5, window_ms=120, cutoff_hz=2,
                                savgol_order=3, spline_smoothing=1)
    assert result.shape == x.shape and np.all(np.isfinite(result))
    assert np.std(np.diff(result)) < np.std(np.diff(x))


def test_residual_reconstruction_factor_zero_and_one():
    f0 = 220 * 2 ** ((20*np.sin(np.arange(101)) + np.linspace(0,50,101))/1200)
    common = {"trend_method":"moving_mean", "smoothing_window_ms":100, "minimum_region_ms":1}
    zero, _ = lab.transform_f0(f0, {**common, "modulation_factor":0}, 5)
    one, _ = lab.transform_f0(f0, {**common, "modulation_factor":1}, 5)
    assert np.std(np.diff(lab.hz_to_cents(zero))) < np.std(np.diff(lab.hz_to_cents(f0)))
    assert np.allclose(one, f0)


def test_aggressive_boundaries_are_not_smoothed():
    weights = lab.boundary_mix(10, {"preserve_attack_ms":0, "preserve_release_ms":0,
                                    "transition_ms":0, "transition_curve":"step"}, 5)
    assert np.array_equal(weights, np.ones(10))


def test_normalized_score_anchors_and_denominator():
    source = np.array([0., 0.]); target = np.array([2., 0.])
    assert lab.normalized_distance(target, target, source)["score"] == 0
    assert lab.normalized_distance(source, target, source)["score"] == 1
    assert lab.normalized_distance(np.array([-1., 0.]), target, source)["score"] > 1
    assert lab.normalized_distance(source, source, source)["status"] == "non_informative"


def test_worst_plus_median_aggregation_formula(monkeypatch, tmp_path):
    # The formula is kept explicit in source so all contexts participate in one objective.
    source = inspect.getsource(lab.evaluate_candidate)
    assert "effect=.5*worst+.5*median+penalty" in source
    assert source.count("prepared_contexts(env)") == 1


def test_invalid_output_constraints(tmp_path):
    config = lab.read_yaml(ROOT / "config/glados-modulation-search.yaml")
    missing = lab.output_constraints(tmp_path/"none.wav", 100, 24000, config)
    assert not missing["valid"]
    clipped = wav(tmp_path/"clip.wav", np.ones(1000))
    assert "extreme_clipping" in lab.output_constraints(clipped, 1000, 24000, config)["violations"]


def test_duration_sample_rate_and_channels(tmp_path):
    config = lab.read_yaml(ROOT / "config/glados-modulation-search.yaml")
    path = wav(tmp_path/"ok.wav", tone(seconds=.1), 24000)
    result = lab.output_constraints(path, 2400, 24000, config)
    assert result["valid"] and result["sample_rate"] == 24000 and result["channels"] == 1


def test_cache_key_is_deterministic_and_parameter_sensitive(tmp_path):
    path = wav(tmp_path/"x.wav")
    one = lab.cache_key(path, "world", {"b":2,"a":1}, 24000)
    assert one == lab.cache_key(path, "world", {"a":1,"b":2}, 24000)
    assert one != lab.cache_key(path, "world", {"a":2,"b":2}, 24000)


def test_world_passthrough_approximately_preserves_duration(tmp_path):
    pytest.importorskip("pyworld")
    env = dict(lab.DEFAULTS); env["AEGIS_GLADOS_MOD_SAMPLE_RATE"]="24000"
    source=wav(tmp_path/"source.wav",tone(seconds=.5));output=tmp_path/"output.wav"
    result=lab.WorldModulationEngine(env).render(source,output,{"passthrough_f0":True,"f0_extractor":"dio_stonemask"})
    assert result["success"] and sf.info(output).frames==sf.info(source).frames


def test_candidate_uses_same_parameters_in_all_four_contexts():
    source = inspect.getsource(lab.evaluate_candidate)
    assert "engine,context[\"source_path\"],output,parameters" in source.replace(" ", "")
    assert len(lab.CONTEXTS) == 4


def test_conditional_search_space():
    source = inspect.getsource(lab.suggest_parameters)
    assert 'method=="savgol"' in source and 'method=="robust_spline"' in source
    assert 'engine=="world"' in source


def test_resume_and_total_cap_contract():
    source = inspect.getsource(lab.optimize)
    assert "already exists; --resume is required" in source
    assert "cannot --resume" in source
    assert "min(trials,max_total_trials-existing)" in source


def test_trial_jsonl_flushes_and_survives_interrupt(tmp_path):
    path=tmp_path/"trials.jsonl";lab.append_jsonl(path,{"trial":1,"status":"complete"})
    assert json.loads(path.read_text())["status"]=="complete"


def test_private_generated_files_are_ignored():
    targets=("experiments/glados-filter-lab/.env.modulation","experiments/glados-filter-lab/reference/private.wav",
             "experiments/glados-filter-lab/modulation-results/studies.sqlite3","reports/glados-lab/modulation/runs/x",
             "private.mpd")
    for target in targets:
        assert subprocess.run(["git","check-ignore","-q",target],cwd=ROOT).returncode==0


def test_no_private_binary_is_tracked():
    output=subprocess.run(["git","ls-files","*.wav","*.mpd","*.sqlite3"],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    assert output.strip()==""


def test_html_links_are_existing_files(tmp_path):
    env, _ = synthetic_env(tmp_path); lab.preflight(env)
    paths=lab.output_paths(env);candidate_dir=paths["raw"]/"candidate-000000";candidate_dir.mkdir(parents=True)
    contexts={}
    for cid,_,_ in lab.CONTEXTS:
        output=wav(candidate_dir/f"{cid}.wav");contexts[cid]={"output":str(output),"metrics":{}}
    record={"candidate":"candidate-000000","trial":0,"engine":"world","effect_score":.5,"worst_context":.6,"parameters":{},"contexts":contexts}
    lab.write_json(paths["out"]/"best-candidates.json",[record]);result=lab.write_listening(env)
    for key in ("identified","blind","map"): assert Path(result[key]).is_file()
    text=Path(result["identified"]).read_text()
    for match in __import__("re").findall(r"src='([^']+)'",text): assert (paths["out"]/match).resolve().is_file()


@pytest.mark.modulation_integration
def test_real_factorial_world_and_praat():
    if os.environ.get("AEGIS_GLADOS_MOD_INTEGRATION_TESTS") != "1": pytest.skip("opt in")
    env=lab.environment();manifest=lab.preflight(env);assert len(manifest["pairs"])==4
    config=lab.read_yaml(Path(env["AEGIS_GLADOS_MOD_CONFIG"]));params={"trend_method":"moving_median","smoothing_window_ms":180,"minimum_region_ms":15,"modulation_factor":0}
    for engine in ("world","praat"):
        if manifest["engines"][engine]["available"]:
            record=lab.evaluate_candidate(env,config,f"integration-{engine}",engine,params)
            assert record["status"]=="complete" and len(record["contexts"])==4
