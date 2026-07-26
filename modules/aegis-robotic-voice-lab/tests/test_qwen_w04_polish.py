from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aegis_robotic_voice.mastering.deesser import deess
from aegis_robotic_voice.qwen_w04_polish import _safe_path, _validate_document, _world_parameters
from aegis_robotic_voice.synthesis.world_engine import synthesize_world
from aegis_robotic_voice.transforms.envelope import attenuate_sibilance_envelope


def _catalog():
    return yaml.safe_load((ROOT / "config" / "qwen-w04-sibilance-polish-v0.2.yaml").read_text())


def _focus_catalog():
    return yaml.safe_load((ROOT / "config" / "qwen-w04-sibilance-focus-v0.3.yaml").read_text())


def _aperiodicity_catalog():
    return yaml.safe_load((ROOT / "config" / "qwen-w04-aperiodicity-focus-v0.4.yaml").read_text())


def _aggressive_catalog():
    return yaml.safe_load((ROOT / "config" / "qwen-w04-sibilance-aggressive-v0.5.yaml").read_text())


def _aperiodicity_sweep_catalog():
    return yaml.safe_load((ROOT / "config" / "qwen-w04-aperiodicity-sweep-v0.6.yaml").read_text())


def test_polish_catalog_pins_q00_and_all_three_conservative_variants():
    data = _catalog()
    assert data["source"]["profile"] == "Q00"
    assert len(data["source"]["expected_sha256"]) == 15
    assert list(data["variants"]) == ["S01", "S02", "S03"]
    assert [data["variants"][code]["spectral_sibilance"]["shelf_db"] for code in data["variants"]] == [-2.0, -3.5, -5.0]
    assert [data["variants"][code]["deesser"]["max_reduction_db"] for code in data["variants"]] == [2.0, 3.5, 5.0]
    _validate_document(data)


def test_focus_catalog_selects_four_stress_sentences_and_ten_ordered_variants():
    data = _focus_catalog()
    layout = _validate_document(data)
    assert [sentence["id"] for sentence in layout["sentences"]] == ["sibilants", "numbers", "long", "sarcastic"]
    assert list(data["variants"]) == [f"T{number:02d}" for number in range(1, 11)]
    assert data["variants"]["T01"]["deesser"]["enabled"] is False
    assert data["variants"]["T09"]["deesser"]["max_reduction_db"] == 12.0
    assert data["reporting"]["include_raw_in_listening"] is False


def test_aperiodicity_catalog_varies_only_the_bounded_voiced_scale():
    data = _aperiodicity_catalog()
    _validate_document(data)
    scales = [variant["voiced_aperiodicity_scale"] for variant in data["variants"].values()]
    assert scales[:5] == [0.14, 0.17, 0.20, 0.23, 0.26]
    effective = _world_parameters(data, data["variants"]["A05"])
    assert effective["aperiodicity_voiced_scale"] == 0.26
    assert effective["formant_scale"] == data["world_w04"]["formant_scale"] == 1.10
    assert effective["quantization_step_cents"] == data["world_w04"]["quantization_step_cents"] == 200


def test_aggressive_catalog_has_audible_high_band_controls():
    data = _aggressive_catalog()
    _validate_document(data)
    assert list(data["variants"]) == [f"G{number:02d}" for number in range(1, 11)]
    assert data["variants"]["G10"]["spectral_sibilance"]["shelf_db"] == -36.0
    assert data["variants"]["G10"]["deesser"]["max_reduction_db"] == 28.0
    spectrum = np.ones((2, 257))
    voiced = np.array([False, False])
    broad = attenuate_sibilance_envelope(spectrum, 24000, voiced, cutoff_hz=5200, shelf_db=-18)
    tight = attenuate_sibilance_envelope(spectrum, 24000, voiced, cutoff_hz=5200, shelf_db=-18, shelf_width_hz=800)
    bin_6500 = int(6500 / 12000 * 256)
    assert tight[0, bin_6500] < broad[0, bin_6500]


def test_raw_aperiodicity_sweep_is_one_sentence_and_spans_the_full_world_range():
    data = _aperiodicity_sweep_catalog()
    layout = _validate_document(data)
    assert [sentence["id"] for sentence in layout["sentences"]] == ["sibilants"]
    assert layout["baseline_stage"] == "raw"
    assert [variant["voiced_aperiodicity_scale"] for variant in data["variants"].values()] == [0.0, 0.03, 0.08, 0.14, 0.20, 0.30, 0.50, 0.75, 1.0]
    assert data["reporting"]["listening_stage"] == "raw"


def test_polish_rejects_protected_paths_before_any_audio_is_opened():
    with pytest.raises(ValueError, match="protected/AegisVoicev0.1"):
        _safe_path("../../protected/AegisVoicev0.1/immutable.wav", label="test")


def test_sibilance_shelf_skips_sustained_voiced_frames_and_reduces_unvoiced_high_band():
    spectral = np.ones((4, 65))
    result = attenuate_sibilance_envelope(spectral, 24000, np.array([True, True, False, False]), cutoff_hz=5200, shelf_db=-5, transition_frames=0)
    assert np.array_equal(result[:2], spectral[:2])
    assert result[2:, -1].mean() < spectral[2:, -1].mean()
    assert np.array_equal(result[:, 0], spectral[:, 0])


def test_dynamic_deesser_only_reduces_an_over_threshold_sibilance_band():
    rate = 24000
    quiet = np.full(rate, 1e-5)
    assert np.array_equal(deess(quiet, rate, threshold_dbfs=-32, max_reduction_db=5), quiet)
    samples = np.arange(rate) / rate
    sharp = 0.4 * np.sin(2 * np.pi * 6500 * samples)
    reduced = deess(sharp, rate, threshold_dbfs=-40, max_reduction_db=5)
    assert len(reduced) == len(sharp)
    assert np.sqrt(np.mean(reduced * reduced)) < np.sqrt(np.mean(sharp * sharp))
    assert np.array_equal(deess(sharp, rate, enabled=False, deesser_enabled=True), sharp)


def test_renderer_and_script_cannot_use_a_source_waveform_or_qwen_runtime():
    assert "source_waveform" not in inspect.signature(synthesize_world).parameters
    source = (ROOT / "src" / "aegis_robotic_voice" / "qwen_w04_polish.py").read_text()
    script = (ROOT / "scripts" / "run_qwen_w04_polish.sh").read_text().lower()
    assert "from .qwen_cold" not in source
    assert "subprocess" not in source
    assert "qwen-cold-run" not in script
    focus_script = (ROOT / "scripts" / "run_qwen_w04_sibilance_focus.sh").read_text().lower()
    assert "qwen-cold-run" not in focus_script
    assert "docker" not in focus_script.splitlines()[-1]
    aperiodicity_script = (ROOT / "scripts" / "run_qwen_w04_aperiodicity_focus.sh").read_text().lower()
    assert "qwen-cold-run" not in aperiodicity_script
    aggressive_script = (ROOT / "scripts" / "run_qwen_w04_sibilance_aggressive.sh").read_text().lower()
    assert "qwen-cold-run" not in aggressive_script
