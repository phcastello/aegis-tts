"""Dependency-free checks for evaluation selection, duration and private packaging."""
import html.parser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from benchmarks.architecture_compare.f5_probe import duration_frames
from training import f5_evaluate as evaluate


def test_duration_does_not_silently_truncate_long_generation():
    # The heuristic includes conditioning frames; long content must fail clearly.
    with pytest.raises(ValueError, match='exceeds'):
        duration_frames(600, 'Uma frase.', 'Uma frase longa. ' * 100, 20)
    assert duration_frames(600, 'Olá.', 'Olá.', 20) == 1200
    with pytest.raises(ValueError, match='required'):
        duration_frames(600, '', 'Uma frase.', 20)


def test_corpus_contains_separate_raw_spoken_and_long_cases():
    rows = evaluate.corpora()
    identifiers = [r['id'] for r in rows]
    assert len(identifiers) == len(set(identifiers))
    rows = {r['id']: r for r in rows}
    assert rows['numbers_raw']['input_mode'] == 'verbatim'
    assert rows['numbers_spoken']['input_mode'] == 'approved_spoken'
    assert rows['numbers_raw']['text'] != rows['numbers_spoken']['text']
    assert len(rows['long_stability']['text']) > 250
    assert 'aegis_command' in rows


def test_roundtrip_references_choose_whole_training_rows_only(tmp_path):
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    train = [{'id': name, 'duration': seconds, 'text': 'Uma frase inteira.'} for name, seconds in [('long', 17), ('near', 6.1), ('short', 3), ('exact', 6), ('close', 5.7)]]
    (prepared / 'train.jsonl').write_text('\n'.join(map(json.dumps, train)))
    (prepared / 'test.jsonl').write_text(json.dumps({'id': 'heldout', 'duration': 6, 'text': 'Não usar.'}))
    selected = evaluate.refs(tmp_path)
    assert [r['id'] for r in selected] == ['exact', 'near', 'close']
    assert all(r in train for r in selected)
    assert all(r['text'] == 'Uma frase inteira.' for r in selected)


class AudioSources(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attributes):
        if tag == 'audio':
            self.sources.append(dict(attributes)['src'])


def test_package_uses_portable_relative_paths_and_exact_hashes(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    original = root / 'roundtrip' / 'clip'
    original.mkdir(parents=True)
    (original / 'A-original-treated.wav').write_bytes(b'original-treated')
    candidate = root / 'evaluations' / 'best'
    candidate.mkdir(parents=True)
    (candidate / 'quote"&name.wav').write_bytes(b'candidate-unmodified')
    old = tmp_path / 'old'
    qwen = old / 'qwen-fixed-cohort'
    qwen.mkdir(parents=True)
    (qwen / 'existing.wav').write_bytes(b'qwen-existing')
    monkeypatch.setattr(evaluate, 'OLD', old)
    evaluate.package(root)
    dest = root / 'listening'
    document = (dest / 'index.html').read_text()
    parser = AudioSources()
    parser.feed(document)
    assert len(parser.sources) == 3
    assert 'pending' in document.lower() or 'escuta' in document.lower()
    manifest = json.loads((dest / 'manifest-sha256.json').read_text())
    for relative in parser.sources:
        assert not urlsplit(relative).scheme
        assert not relative.startswith('/')
        path = dest / unquote(relative)
        assert path.resolve().is_relative_to(dest.resolve())
        assert path.is_file()
        assert evaluate.digest(path) == manifest[relative]
    # Repackaging cannot transform audio or manufacture fresh benchmark evidence.
    evaluate.package(root)
    assert json.loads((dest / 'manifest-sha256.json').read_text()) == manifest


def test_closed_gpu_authorization_refuses_before_model_load(tmp_path, monkeypatch):
    (tmp_path / 'gpu-window-authorized.json').write_text(json.dumps({'owner_authorized': True, 'production_stopped': True, 'status': 'gpu_phases_closed'}))
    monkeypatch.setattr(evaluate, 'bootstrap', lambda: None)
    monkeypatch.setattr(evaluate, 'load_model', lambda path: pytest.fail('must reject before model load'))
    with pytest.raises(ValueError, match='active'):
        evaluate.final(tmp_path)


def test_local_helpers_ignore_unrelated_benchmarks_package(monkeypatch):
    import sys
    from types import ModuleType
    # The real isolated environment contains a regular package with this name.
    # Evaluation must select the repository's standalone helpers explicitly.
    monkeypatch.setitem(sys.modules, 'benchmarks', ModuleType('benchmarks'))
    from f5_probe import duration_frames as local_duration
    from benchmark import pcm_metrics as local_metrics
    import inspect
    expected = Path(__file__).resolve().parents[1] / 'benchmarks' / 'architecture_compare'
    assert Path(inspect.getfile(local_duration)).parent == expected
    assert Path(inspect.getfile(local_metrics)).parent == expected


def test_word_rate_changes_only_neural_duration_budget():
    frames = evaluate.word_rate_duration(947, 'A gaveta azul contém três chaves e um bilhete de Joana.', 124.60845)
    audio_seconds = (frames - 947 - 1) * evaluate.HOP / evaluate.RATE
    assert abs(audio_seconds - 11 * 60 / 124.60845) <= evaluate.HOP / evaluate.RATE / 2
    with pytest.raises(ValueError):
        evaluate.word_rate_duration(947, '', 124.6)
    with pytest.raises(ValueError, match='maximum'):
        evaluate.word_rate_duration(947, 'uma palavra ' * 500, 124.6)
