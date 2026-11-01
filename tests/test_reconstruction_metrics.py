import numpy as np
import pytest

from training.reconstruct import compare_spectra, texture_metrics


def test_reconstruction_identity_metrics_and_spectral_texture_are_not_perceptual_approval():
    values = .2 * np.sin(np.arange(22050) * 2 * np.pi * 200 / 22050)
    comparison = compare_spectra(values, values)
    assert comparison['log_spectral_mae'] == pytest.approx(0)
    assert comparison['spectral_convergence'] == pytest.approx(0)
    features = texture_metrics(values, 22050)
    assert 180 < features['spectral_centroid_hz'] < 220
    assert features['rms_dbfs'] == pytest.approx(-16.9897, abs=.001)
    assert features['peak'] < .201
    assert features['power_above_8000hz_ratio'] < 1e-8


def test_level_change_remains_visible_without_loudness_normalization():
    values = .2 * np.sin(np.arange(22050) * 2 * np.pi * 200 / 22050)
    target, rendered = texture_metrics(values, 22050), texture_metrics(values / 2, 22050)
    assert target['rms_dbfs'] - rendered['rms_dbfs'] == pytest.approx(6.0206, abs=.001)
    assert compare_spectra(values, values / 2)['spectral_convergence'] == pytest.approx(.5)


def test_source_high_frequency_power_is_measured_before_resampling():
    source = .2 * np.sin(np.arange(48000) * 2 * np.pi * 12000 / 48000)
    assert texture_metrics(source, 48000)['power_above_11025hz_ratio'] > .99
