"""Native audio integrity and explicit browser quantization, without a model."""
import hashlib
import html.parser
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training import f5_listening_compat as compat


def package(tmp_path, values, subtype, rate=48000):
    directory = tmp_path / 'listening'
    directory.mkdir(mode=0o700)
    source = directory / 'native"&clip.wav'
    sf.write(source, np.asarray(values), rate, subtype=subtype)
    original = source.read_bytes()
    (directory / 'manifest-sha256.json').write_text(json.dumps({source.name: hashlib.sha256(original).hexdigest()}))
    (directory / 'index.html').write_text('<!doctype html><h1>Avaliação</h1><article><audio controls src="native&quot;&amp;clip.wav"></audio></article>')
    return directory, source, original


class AudioParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.audio_sources = []
        self.downloads = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'audio':
            self.audio_sources.append(attributes['src'])
        if tag == 'a' and attributes.get('data-native-download') == 'true':
            self.downloads.append(attributes['href'])


@pytest.mark.parametrize('subtype', ['FLOAT', 'PCM_24'])
def test_finite_native_preserved_and_browser_copy_quantized_only_once(tmp_path, subtype):
    directory, source, original = package(tmp_path, [0, 0.1234567, -0.3456789, 0.75, -0.88], subtype)
    receipt = compat.process(directory)
    row = receipt['files'][0]
    assert source.read_bytes() == original
    assert row['native_subtype'] == subtype
    assert row['status'] == 'converted'
    target = directory / row['playback_path']
    assert sf.info(target).subtype == 'PCM_16'
    assert sf.info(target).samplerate == 48000
    assert sf.info(target).frames == sf.info(source).frames
    values, rate = sf.read(source, dtype='float64')
    copied, copied_rate = sf.read(target, dtype='float64')
    assert copied_rate == rate
    assert np.max(np.abs(values-copied)) <= 0.500001 / 32768
    assert row['maximum_error_lsb'] <= 0.500001
    parser = AudioParser()
    parser.feed((directory / 'index.html').read_text())
    assert parser.audio_sources == [row['playback_path']]
    assert parser.downloads == [source.name]
    manifest = json.loads((directory / 'manifest-sha256.json').read_text())
    assert set(manifest) == {source.name, row['playback_path']}
    assert all(compat.digest(directory/p) == sha for p,sha in manifest.items())
    first = target.read_bytes()
    compat.process(directory)
    assert target.read_bytes() == first
    assert source.read_bytes() == original
    assert len(list((directory / 'playback').rglob('*.wav'))) == 1
    html = (directory / 'index.html').read_text()
    assert html.count('id="browser-playback-compat"') == 1
    assert html.count('data-native-download="true"') == 1


@pytest.mark.parametrize('values,reason', [([0, 1.2, -1.1], 'declined_overrange_or_rounding_overflow'), ([0, 0.99999999], 'declined_overrange_or_rounding_overflow'), ([0, float('nan')], 'declined_nonfinite_or_empty')])
def test_overrange_or_nonfinite_float_retains_original_without_clipping(tmp_path, values, reason):
    directory, source, original = package(tmp_path, values, 'FLOAT', rate=24000)
    receipt = compat.process(directory)
    row = receipt['files'][0]
    assert row['status'] == reason
    assert row['playback_created'] is False
    assert source.read_bytes() == original
    assert not (directory / 'playback').exists()
    parser = AudioParser()
    parser.feed((directory / 'index.html').read_text())
    assert parser.audio_sources == [source.name]
    assert parser.downloads == [source.name]
    assert len(json.loads((directory / 'manifest-sha256.json').read_text())) == 1


def test_manifest_tampering_refuses_before_any_conversion(tmp_path):
    directory, source, original = package(tmp_path, [0, .1, -.1], 'FLOAT')
    source.write_bytes(original + b'tampered')
    before = (directory / 'index.html').read_bytes()
    with pytest.raises(ValueError, match='does not match manifest'):
        compat.process(directory)
    assert not (directory / 'playback').exists()
    assert (directory / 'index.html').read_bytes() == before
