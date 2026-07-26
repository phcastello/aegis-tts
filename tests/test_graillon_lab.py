import json, os, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import graillon_lab as lab
import graillon_pipe as pipe
import graillon_round2 as round2
import graillon_round3 as round3

def fake_inventory():
    data=json.loads((ROOT/'experiments/glados-filter-lab/graillon-ports.json').read_text()) if (ROOT/'experiments/glados-filter-lab/graillon-ports.json').exists() else None
    if data:return data
    return lab.discover(lab.config())

def test_dotenv_rejects_unknown_and_does_not_evaluate(tmp_path):
    p=tmp_path/'x';p.write_text('AEGIS_GRAILLON_SAMPLE_RATE=24000\nEVIL=$(touch no)\n')
    with pytest.raises(lab.LabError,match='chave desconhecida'):lab.load_dotenv(p)

def test_dotenv_and_environment_precedence(tmp_path,monkeypatch):
    p=tmp_path/'x';p.write_text('AEGIS_GRAILLON_SAMPLE_RATE=16000\n')
    monkeypatch.setenv('AEGIS_GRAILLON_SAMPLE_RATE','24000')
    class A: env_file=str(p);input=None;output_dir=None;preset_config=None
    assert lab.config(A())['AEGIS_GRAILLON_SAMPLE_RATE']=='24000'

def test_semantic_controls_and_ranges_are_checked():
    inv=fake_inventory(); data=lab.read_yaml(ROOT/'config/graillon-presets.yaml')
    g07=next(x for x in data['presets'] if x['id']=='G07'); controls=lab.resolve_preset(g07,inv,lab.config())
    sm=lab.semantic_map();assert controls[sm['allow_g']]==1 and controls[sm['allow_c']]==0
    bad=dict(g07);bad['controls']={'not_real':.5}
    with pytest.raises(lab.LabError,match='desconhecido'):lab.resolve_preset(bad,inv,lab.config())
    bad=dict(g07);bad['raw_controls']={'p17':2}
    with pytest.raises(lab.LabError,match='fora da faixa'):lab.resolve_preset(bad,inv,lab.config())

def test_missing_symbol_is_rejected():
    inv=fake_inventory();inv={'ports':[x for x in inv['ports'] if x['symbol']!='p17']}
    with pytest.raises(lab.LabError,match='ausentes'):lab.validate_inventory(inv)

def test_jalv_arguments_are_safe_list():
    args=lab.jalv_args(lab.config(),{'p17':1.0,'p18':0.0},'safe-client')
    assert isinstance(args,list) and '-c' in args and 'p17=1' in args and args[0]=='jalv'

def test_s16le_alignment():
    class R:
        def __init__(self):self.bits=[b'a',b'bc',b'def',b'']
        def read(self,n):return self.bits.pop(0)
    assert b''.join(pipe.aligned_chunks(R()))==b'abcdef'
    class Odd:
        def __init__(self):self.done=False
        def read(self,n):
            if self.done:return b''
            self.done=True;return b'x'
    with pytest.raises(pipe.PipeError):list(pipe.aligned_chunks(Odd()))

def test_no_process_killers_in_lab_sources():
    text=(ROOT/'tools/graillon_lab.py').read_text()+(ROOT/'tools/graillon_pipe.py').read_text()
    assert 'pkill ' not in text and 'killall ' not in text

def test_failed_record_keeps_manifest_and_html(tmp_path):
    c=lab.config();source=ROOT/'samples/glados-lab/source.wav'
    lab.write_manifest(c,tmp_path,[{'id':'G03','name':'x','description':'x','success':False,'error':'expected'}],{'source':str(source)})
    assert (tmp_path/'manifest.json').exists() and (tmp_path/'listening.html').exists()

def test_round2_ids_masks_and_forbidden_effects():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round2.yaml'); presets=data['presets']
    assert len(presets)==27 and len({p['id'] for p in presets})==27
    for p in presets:
        controls=p['controls']
        if p['id'] not in ('R200','R203'):
            assert controls['compressor']==controls['gate']==controls['chorus']==controls['doubler']==0.0
    for p in [x for x in presets if 'R250'<=x['id']<='R256']:
        assert all('allow_'+n in p['controls'] for n in lab.NOTES)
        assert {p['controls']['allow_'+n] for n in lab.NOTES} <= {0.0,1.0}

def test_round2_families_vary_only_their_stated_dimension():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round2.yaml')['presets']
    byid={x['id']:x['controls'] for x in data}
    def changed(ids):
        first=byid[ids[0]]; return {k for k in first if any(byid[i].get(k)!=first.get(k) for i in ids[1:])}
    assert changed(['R210','R211','R212'])=={'pitch_engine'}
    assert changed(['R220','R221','R222','R223'])=={'wet_mix','dry_mix'}
    assert changed(['R230','R231','R232','R233','R234','R235'])=={'smooth','inertia'}

def test_round2_engine_values_and_base_materialization():
    engines=round2.ENGINES; assert engines=={'G2':1.0,'G3':0.5,'I2':0.0}
    data=lab.read_yaml(ROOT/'config/graillon-presets-round2.yaml')
    pre=next(x for x in data['presets'] if x['id']=='R220')
    assert round2.materialize(pre,data,'G3')['controls']['pitch_engine']==0.5

def test_round2_controls_are_not_overridden_by_legacy_g_series_defaults():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round2.yaml')
    pre=round2.materialize(next(x for x in data['presets'] if x['id']=='R221'),data,'G3')
    controls=lab.resolve_preset(pre,fake_inventory(),lab.config())
    sm=lab.semantic_map()
    assert controls[sm['wet_mix']]==pytest.approx(.9)
    assert controls[sm['dry_mix']]==pytest.approx(.1)
    assert controls[sm['formant_enabled']]==pytest.approx(1)
    assert controls[sm['fx_enabled']]==pytest.approx(1)


def test_optimizer_explicit_formant_controls_are_not_overridden():
    pre={'id':'RE','name':'optimizer','description':'test','base_state':'official-hard-tune',
         'round2':True,'controls':{'formant_enabled':1.0,'formant_amount':1.0,
                                   'formant_shift':.625,'correction_amount':0.0}}
    controls=lab.resolve_preset(pre,fake_inventory(),lab.config());sm=lab.semantic_map()
    assert controls[sm['formant_enabled']]==pytest.approx(1.0)
    assert controls[sm['formant_amount']]==pytest.approx(1.0)
    assert controls[sm['formant_shift']]==pytest.approx(.625)

def test_every_round2_yaml_control_maps_to_an_in_range_lv2_symbol():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round2.yaml')
    inv=fake_inventory(); sm=lab.semantic_map()
    for source in data['presets']:
        if source['id']=='R200':
            continue
        pre=round2.materialize(source,data,'G3')
        resolved=lab.resolve_preset(pre,inv,lab.config())
        for semantic,value in pre['controls'].items():
            assert resolved[sm[semantic]]==pytest.approx(float(value))

def test_round2_bypass_classification_uses_pcm_not_wav_container_hash():
    record={'success':True,'sha256':'a'}
    assert round2.classify(record,b'\x01\x00',b'\x01\x00')=='identical_to_bypass'

def test_round3_is_small_explicit_and_uses_calibrated_plus_1p5_formant_shift():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round3.yaml')
    assert data['round']=='round3' and len(data['presets'])==10
    assert len({p['id'] for p in data['presets']})==10
    shifted=[p for p in data['presets'] if p['controls']['formant_shift']!=.5]
    assert [p['id'] for p in shifted]==['S304','S314']
    assert all(p['controls']['formant_shift']==pytest.approx(.5625) for p in shifted)
    for pre in data['presets']:
        c=pre['controls']
        assert c['compressor']==c['gate']==c['chorus']==c['doubler']==0.0
        assert all(c['allow_'+note] in (0.0,1.0) for note in lab.NOTES)

def test_round3_formant_calibration_is_linear_from_official_range():
    calibration=json.loads((ROOT/'experiments/glados-filter-lab/graillon-formant-shift.json').read_text())
    assert calibration['physical_range_semitones']==[-12.0,12.0]
    assert calibration['plus_one_point_five_semitones']==pytest.approx(.5+1.5/24)

def test_round4_separates_formant_amount_from_shift():
    data=lab.read_yaml(ROOT/'config/graillon-presets-round4.yaml')
    assert data['round']=='round4' and len(data['presets'])==14
    for p in data['presets']:
        c=p['controls']
        assert c['wet_mix']==1.0 and c['dry_mix']==0.0
        assert c['smooth']==0.0 and c['inertia']==0.0
        assert c['compressor']==c['gate']==c['chorus']==c['doubler']==0.0
    amount=[p for p in data['presets'] if 'amount' in p['family']]
    shift=[p for p in data['presets'] if 'shift' in p['family']]
    assert {p['controls']['formant_shift'] for p in amount}=={.5625}
    assert {p['controls']['formant_amount'] for p in shift}=={1.0}
    assert {p['controls']['formant_shift'] for p in shift}=={.625,.6875,.75}
