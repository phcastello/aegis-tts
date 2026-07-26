from __future__ import annotations
import ast, inspect, os, sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aegis_robotic_voice.manifests import append_jsonl, read_jsonl
from aegis_robotic_voice.timing import stats
from aegis_robotic_voice.transforms.envelope import transform_envelope
from aegis_robotic_voice.transforms.aperiodicity import transform_aperiodicity
from aegis_robotic_voice.synthesis.overlap_add import overlap_add
from aegis_robotic_voice.synthesis.lpc_engine import analyze_lpc

def test_manifests_are_incremental(tmp_path):
    p=tmp_path/"m.jsonl"; append_jsonl(p,{"n":1}); append_jsonl(p,{"n":2}); assert read_jsonl(p)==[{"n":1},{"n":2}]
def test_envelope_shape_and_aperiodicity_domain():
    x=np.ones((3,20)); assert transform_envelope(x,formant_scale=1.1).shape==x.shape
    ap=transform_aperiodicity(np.ones((3,20)),np.array([True,False,True]),aperiodicity_voiced_scale=.2); assert np.all((0<=ap)&(ap<=1))
def test_overlap_add_and_lpc_no_nan():
    assert np.array_equal(overlap_add([np.ones(3),np.ones(3)],2,5),np.ones(5))
    a=analyze_lpc(np.random.default_rng(1).normal(size=400),16000,np.ones(5)*100,np.ones(5,dtype=bool)); assert np.isfinite(a.coefficients).all()
def test_rtf_statistics():
    s=stats([.1,.2,.3]); assert s["min"]==.1 and s["p90"]>.2
def test_no_shell_true_or_source_waveform_argument():
    text="\n".join(p.read_text() for p in (ROOT/"src").rglob("*.py")); assert "shell=True" not in text
    assert "def synthesize_world(representation" in text and "def synthesize_lpc(representation" in text
def test_gitignore_is_module_scoped_for_new_lab_rules():
    text=(ROOT.parents[1]/".gitignore").read_text(); assert "modules/aegis-robotic-voice-lab/**/*.wav" in text
def test_presets_are_curated_not_cartesian():
    import yaml
    data=yaml.safe_load((ROOT/"config/presets.yaml").read_text()); assert set(data["world"])=={"W00","W01","W02","W03","W04"}; assert set(data["lpc"])=={"L00","L01","L02","L03"}
def test_cuda_measurement_contract_is_present():
    worker=(ROOT/"scripts/chatterbox_worker.py").read_text(); source=(ROOT/"src/aegis_robotic_voice/source/chatterbox_ptbr.py").read_text(); assert worker.count("torch.cuda.synchronize()") >= 2 and "torch.inference_mode" in source
def test_qwen_cold_matrix_is_explicit_and_uses_only_strict_selected_effects():
    import yaml
    data=yaml.safe_load((ROOT/"config/qwen-cold.yaml").read_text())
    assert list(data["profiles"])==["Q00","Q01","Q02","Q03","Q04"]
    assert set(data["effects"])=={"L00","W04"}
    assert data["profiles"]["Q00"]["temperature"] <= .05
    source=(ROOT/"src/aegis_robotic_voice/qwen_cold.py").read_text()
    assert "shell=True" not in source
    assert "source_waveform" not in source
    assert "Aninha_AegisVoice4.wav" in (ROOT/"config/qwen-cold.yaml").read_text()
def test_selected_q00_w04_is_pinned_and_strict():
    import yaml
    data=yaml.safe_load((ROOT/"config/qwen-selected-q00-w04.yaml").read_text())
    assert data["selection"]["source_profile"] == "Q00"
    assert data["selection"]["effect_preset"] == "W04"
    assert data["qwen_vllm"]["stage_0"]["top_k"] == 1
    assert data["world_w04"]["source_waveform_is_audible"] is False
def test_no_production_environment_target():
    text=(ROOT/"scripts/create_environments.sh").read_text(); assert "private/environments" in text and "AEGIS" not in text
@pytest.mark.robotic_voice_integration
def test_integration_is_opt_in():
    if os.getenv("AEGIS_ROBOTIC_VOICE_INTEGRATION") != "1": pytest.skip("set AEGIS_ROBOTIC_VOICE_INTEGRATION=1")
    assert (ROOT/"private/environments/chatterbox/bin/python").exists()
