"""The production candidate factory must preserve model levels during inference."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from training.piper_adapter import PiperUnitAdapter
from training.prepare_dataset import sha256


def test_factory_passes_no_normalization_to_runtime_and_preserves_pcm_level(tmp_path, monkeypatch):
    seen = []

    class SynthesisConfig:
        def __init__(self, normalize_audio=True):
            self.normalize_audio = normalize_audio

    class Voice:
        def __init__(self, **kwargs):
            pass

        def synthesize(self, text, syn_config=None):
            seen.append(syn_config)
            values = np.array([0., .125, -.125, 0.])
            if syn_config is None or syn_config.normalize_audio:
                values /= np.max(np.abs(values))
            yield SimpleNamespace(sample_channels=1, sample_rate=24000, audio_float_array=values)

    ort = ModuleType('onnxruntime')
    ort.SessionOptions = SimpleNamespace
    ort.InferenceSession = lambda *args, **kwargs: object()
    piper = ModuleType('piper')
    piper.PiperVoice = Voice
    config_module = ModuleType('piper.config')
    config_module.PiperConfig = SimpleNamespace(from_dict=lambda value: value)
    config_module.SynthesisConfig = SynthesisConfig
    for name, module in [('onnxruntime', ort), ('piper', piper), ('piper.config', config_module)]:
        monkeypatch.setitem(sys.modules, name, module)
    model, config = tmp_path/'candidate.onnx', tmp_path/'candidate.json'
    model.write_bytes(b'factory contract fixture; not a voice model')
    config.write_text(json.dumps({'espeak': {'voice': 'pt-br'}}))
    adapter = PiperUnitAdapter.load(model, config, sha256(model))
    assert adapter.synthesis_config.normalize_audio is False
    pcm = np.frombuffer(b''.join(adapter.stream_unit('Teste.')), dtype='<i2')
    assert pcm.tolist() == [0, 4096, -4096, 0]
    assert seen == [adapter.synthesis_config]


@pytest.mark.parametrize('sample_rate', [22050, 24000])
def test_disabling_normalization_does_not_hide_model_pcm_rails(sample_rate):
    class RailVoice:
        def synthesize(self, text, syn_config=None):
            assert syn_config.normalize_audio is False
            values = np.zeros(1024)
            values[100] = 1.
            yield SimpleNamespace(sample_channels=1, sample_rate=sample_rate, audio_float_array=values)

    adapter = PiperUnitAdapter(RailVoice(), syn_config=SimpleNamespace(normalize_audio=False))
    with pytest.raises(ValueError, match='clipped audio'):
        list(adapter.stream_unit('Clipping.'))
