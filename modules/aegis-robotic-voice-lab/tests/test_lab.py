from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]; import sys; sys.path.insert(0,str(ROOT/"src"))
from aegis_robotic_voice.config import load_dotenv, yaml_config
from aegis_robotic_voice.source.chatterbox_ptbr import validate_checkpoint_identity
from aegis_robotic_voice.transforms.prosody import transform_f0
from aegis_robotic_voice.synthesis.excitation import harmonic_excitation, deterministic_colored_noise
from aegis_robotic_voice.synthesis.lpc_engine import stabilize_poles, LPCRepresentation, synthesize_lpc
from aegis_robotic_voice.synthesis.world_engine import synthesize_world
from aegis_robotic_voice.audio import validate_audio

def test_env_is_literal_and_known(tmp_path):
    p=tmp_path/".env"; p.write_text("AEGIS_CHATTERBOX_MODEL=x\n")
    assert load_dotenv(p)["AEGIS_CHATTERBOX_MODEL"]=="x"
    p.write_text("NOPE=1\n")
    with pytest.raises(ValueError): load_dotenv(p)
def test_corpus_ids_are_unique():
    ids=[x["id"] for x in yaml_config("corpus.yaml")["sentences"]]; assert len(ids)==len(set(ids))==15
def test_regional_identity_rejects_wrong_checkpoint():
    good={"model":"ResembleAI/Chatterbox-Multilingual-pt-br","locale":"pt-BR","language_id":"pt","decoder":"V3","tokenizer":"grapheme_mtl_merged_expanded_v1.json","t3":"t3_pt_br.safetensors","s3gen":"s3gen_v3.safetensors"}; validate_checkpoint_identity(good)
    for key,bad in [("locale","pt-PT"),("model","ResembleAI/chatterbox"),("decoder","V2")]:
        x=good.copy(); x[key]=bad
        with pytest.raises(ValueError): validate_checkpoint_identity(x)
def test_f0_never_crosses_silence_and_quantizes():
    f=np.array([100.,110.,0.,0.,180.,190.]); v=f>0; got=transform_f0(f,v,modulation_factor=0,range_compression=0,quantization="fixed_cents",quantization_step_cents=100)
    assert np.all(got[2:4]==0) and np.all(got[v]>0)
def test_synthetic_noise_is_deterministic_and_harmonics_bandlimited():
    assert np.array_equal(deterministic_colored_noise(100,7),deterministic_colored_noise(100,7)); assert np.isfinite(harmonic_excitation(1000,8000,100)).all()
def test_lpc_stabilization_and_strict_synth_signature():
    assert np.max(np.abs(np.roots(stabilize_poles(np.array([1.,-1.2]))))) < 1
    assert "source_waveform" not in inspect.signature(synthesize_lpc).parameters
    assert "source_waveform" not in inspect.signature(synthesize_world).parameters
    r=LPCRepresentation(16000,400,80,160,np.array([[1.,-.5]]),np.array([.1]),np.array([100.]),np.array([True]),np.zeros((1,1)),np.array([1.]))
    assert np.isfinite(synthesize_lpc(r,seed=1)).all()
def test_audio_validation():
    validate_audio(np.ones(10)*.01,16000,10)
    with pytest.raises(ValueError): validate_audio(np.array([np.nan]),16000)
