"""Small synthetic-only dataset tests. No provider calls or personal voice assets."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import resample_poly

SPEC = importlib.util.spec_from_file_location('dataset_pipeline', Path(__file__).resolve().parents[1] / 'scripts/dataset_pipeline.py')
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)
RATE = 8000


def records(count=6):
    return [{'id': f'test-{i}', 'group': f'g-{i}', 'group_id': f'g-{i}', 'split': 'test',
             'text': f'Exemplo sintético de teste número {i}.', 'stage': 'pilot'} for i in range(count)]


def config(rows):
    return {'provider': 'openai-compatible-http', 'base_url': 'https://invalid.example/speech',
            'model': 'synthetic-test', 'voice': 'synthetic-no-real-voice', 'api_key_env': None,
            'training_rights_confirmed': True, 'voice_rights_confirmed': True,
            'terms_url': 'https://invalid.example/terms', 'terms_reviewed_on': '2026-11-01',
            'currency': 'USD', 'price_per_1000_characters': 1, 'max_cost': 10,
            'max_characters': 10000, 'max_requests': 30, 'max_examples': 20,
            'concurrency': 1, 'retries': 0, 'sample_rate': RATE,
            'separator_seconds': 2, 'bookend_seconds': 2,
            'generation_approved': True, 'approval_reference': 'synthetic-test-only',
            'corpus_review_sha256': pipeline.digest(pipeline.canonical(rows))}


def tone(index, internal=False):
    # Distinctive envelopes and lengths, with no abrupt/large edge or clipping.
    length = int(RATE * (1.2 + index * .23))
    t = np.arange(length) / RATE
    envelope = np.sin(np.linspace(0, np.pi, length)) ** 2
    values = .18 * envelope * (1 + .3 * np.sin(2 * np.pi * (2 + index) * t)) * np.sin(2 * np.pi * (170 + index * 37) * t)
    if internal:
        middle = len(values) // 2
        values = np.r_[values[:middle], np.zeros(round(3.2 * RATE)), values[middle:]]
    return np.r_[np.zeros(round(.12 * RATE)), values, np.zeros(round(.15 * RATE))]


@pytest.fixture
def master(tmp_path):
    rows = records()
    cfg = config(rows)
    def sender(payload):
        index = int(payload['input'].split()[-1].rstrip('.'))
        return pipeline.wav_bytes(tone(index, internal=index == 2), RATE)
    pipeline.generate(rows, cfg, tmp_path, transport=sender)
    pipeline.assemble(rows, cfg, tmp_path)
    return tmp_path, json.loads((tmp_path / 'manifest.json').read_text())


def extraction(master, values=None, rate=RATE):
    directory, manifest = master
    source, _ = pipeline.audio_read(directory / 'master_raw.wav')
    processed = directory / 'master_processed.wav'
    pipeline.atomic(processed, pipeline.wav_bytes(source if values is None else values, rate))
    target = directory / 'processed'
    target.mkdir(exist_ok=True)
    report = pipeline.extract(directory / 'manifest.json', processed, target)
    rows = [json.loads(line) for line in (target / 'dataset.jsonl').read_text().splitlines()]
    return report, rows


def test_cache_rerun_never_calls_provider_and_hash_verified(tmp_path):
    rows, calls = records(1), []
    cfg = config(rows)
    def sender(payload):
        calls.append(payload)
        return pipeline.wav_bytes(tone(0), RATE)
    assert pipeline.generate(rows, cfg, tmp_path, transport=sender)['completed'] == 1
    assert pipeline.generate(rows, cfg, tmp_path, transport=lambda _: pytest.fail('cache miss'))['cached'] == 1
    assert len(calls) == 1
    saved, path, *_ = pipeline.cached(rows[0], cfg, tmp_path)
    assert saved['parameters']['voice'] == 'synthetic-no-real-voice'
    path.write_bytes(b'corrupt')
    assert pipeline.cached(rows[0], cfg, tmp_path)[0] is None


def test_dry_run_missing_fields_and_enforced_ceiling(tmp_path):
    rows = records(2)
    estimate = pipeline.generate(rows, {}, tmp_path, dry_run=True)
    assert 'voice' in estimate['missing_configuration']
    assert 'generation_approved' in estimate['missing_configuration']
    assert estimate['estimated_cost_first_attempt'] is None
    with pytest.raises(ValueError, match='missing explicit'):
        pipeline.generate(rows, {}, tmp_path, transport=lambda _: pytest.fail('must not call'))
    cfg = config(rows)
    cfg['max_requests'] = 1
    result = pipeline.generate(rows, cfg, tmp_path, transport=lambda _: pipeline.wav_bytes(tone(0), RATE))
    assert result['completed'] == 1 and result['failed'] == 1
    assert len(json.loads((tmp_path / 'generation-ledger.json').read_text())['attempts']) == 1


def test_corpus_review_binding_blocks_changed_text(tmp_path):
    rows = records(1)
    cfg = config(rows)
    rows[0]['text'] += ' Modificado.'
    with pytest.raises(ValueError, match='corpus review SHA256'):
        pipeline.generate(rows, cfg, tmp_path, transport=lambda _: pytest.fail('unreviewed corpus'))


def test_master_single_file_manifest_samples_and_internal_pause(master):
    directory, manifest = master
    assert len(list(directory.glob('master_raw*'))) == 1
    assert len(manifest['clips']) == 6 and len(manifest['gaps']) == 7
    assert manifest['master_sha256'] == pipeline.digest((directory / 'master_raw.wav').read_bytes())
    assert manifest['duration'] > manifest['speech_clip_seconds'] + 10
    report, rows = extraction(master)
    assert report['accepted'] == 6, report
    assert [row['id'] for row in rows] == [row['id'] for row in records()]
    assert rows[2]['duration'] > 2.5  # Long internal pause is retained inside its own utterance.
    assert all(row['content_review'] == 'pending' for row in rows)


def test_padding_initial_and_end(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    report, rows = extraction(master, np.r_[np.zeros(round(.73 * RATE)), raw, np.zeros(round(.39 * RATE))])
    assert report['accepted'] == 6, report
    assert all(row['processed_start_sample'] >= round(.7 * RATE) for row in rows)


def test_resampling_export(master):
    directory, _ = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    report, rows = extraction(master, resample_poly(raw, 3, 1), RATE * 3)
    assert report['accepted'] == 6, report
    assert all(row['sample_rate'] == RATE and row['original_sample_rate'] == RATE * 3 for row in rows)


def test_small_effect_tails_preserve_margins(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    # One short decaying tail on every end, without changing the global timeline.
    values = raw.copy()
    for clip in manifest['clips']:
        end = clip['active_end_sample']
        n = round(.06 * RATE)
        tail = .003 * np.exp(-np.linspace(0, 7, n)) * np.sin(np.arange(n) * 2 * np.pi * 310 / RATE)
        values[end:end+n] += tail
    report, rows = extraction(master, values)
    assert report['accepted'] == 6, report
    assert all(row['duration'] > .3 for row in rows)


def test_altered_separator_quarantines_local_neighbors(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    values = raw.copy()
    gap = manifest['gaps'][2]
    n = gap['end'] - gap['start']
    values[gap['start']:gap['end']] = .02 * np.sin(np.arange(n) * 2 * np.pi * 180 / RATE)
    report, rows = extraction(master, values)
    assert report['quarantined'], report
    assert rows[0]['quality_status'] == 'accepted'
    assert rows[-1]['quality_status'] == 'accepted'
    assert any(rows[i]['quality_status'] == 'quarantine' for i in (1, 2))


def test_missing_segment_does_not_shift_following_transcripts(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    clip = manifest['clips'][2]
    values = np.r_[raw[:clip['start_sample']], raw[clip['end_sample']:]]
    report, rows = extraction(master, values)
    assert rows[2]['quality_status'] == 'quarantine', report
    assert rows[-1]['id'] == 'test-5' and rows[-1]['quality_status'] == 'accepted', report
    assert [row['id'] for row in rows] == [row['id'] for row in records()]


def test_global_duration_change_quarantined(master):
    directory, _ = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    # Deliberately change timing; keeping the WAV sample rate simulates a stretched export.
    report, rows = extraction(master, resample_poly(raw, 105, 100))
    assert report['accepted'] == 0, report
    assert all('global_time_stretch_detected' in row['reasons'] for row in rows)


def test_review_bound_to_processed_hash(master):
    directory, _ = master
    extraction(master)
    bad_review = directory / 'review.json'
    bad_review.write_text(json.dumps({'processed_master_sha256': 'wrong', 'examples': {}}))
    with pytest.raises(ValueError, match='processed master SHA256'):
        pipeline.extract(directory / 'manifest.json', directory / 'master_processed.wav', directory / 'processed', bad_review)


def test_corpus_ids_group_splits_and_diversity():
    rows = pipeline.corpus_read(pipeline.ROOT / 'dataset/corpus.pt-BR.jsonl')
    pilot = pipeline.selected(rows, 'pilot')
    assert len(pilot) == 112 and len(rows) == 792
    assert len({row['category'] for row in pilot}) == 11
    assert {row['split'] for row in pilot} == {'train', 'validation', 'test'}
    assert 10 < sum(len(row['text'].split()) for row in pilot) / 150 < 18


def test_content_approval_requires_identity_and_evidence(master):
    directory, _ = master
    extraction(master)
    review_path = directory / 'review.json'
    review_path.write_text(json.dumps({'processed_master_sha256': pipeline.digest((directory / 'master_processed.wav').read_bytes()),
                                      'examples': {'test-0': {'content_review': 'approved'}}}))
    with pytest.raises(ValueError, match='reviewer, reviewed_at'):
        pipeline.extract(directory / 'manifest.json', directory / 'master_processed.wav', directory / 'processed', review_path)


def test_content_audit_detects_missing_words_and_normalizes_digits():
    sys.path.insert(0, str(pipeline.ROOT / 'scripts'))
    from dataset_content_audit import comparison
    assert comparison('O valor é 12.', 'O valor é doze.')['content_check'] == 'pass'
    assert comparison('O sensor está pronto para medir.', 'O sensor está pronto.')['content_check'] == 'review_required'
    assert comparison('A luz ficou azul.', 'O barco chegou cedo.')['word_error_rate'] > .5
    assert comparison('A luz ficou azul.', '')['content_check'] == 'review_required'


def test_truncated_wav_rejected():
    data = pipeline.wav_bytes(tone(0), RATE)
    with pytest.raises(ValueError, match='truncated WAV'):
        pipeline.audio_read(data[:-20])


@pytest.mark.parametrize('which', [0, -1])
def test_missing_edge_with_repeated_layout_never_accepts_shifted_ids(tmp_path, which):
    rows = records(12)
    cfg = config(rows)
    # Equal durations/envelopes; frequency distinguishes identity for a human,
    # while the algorithm's envelope descriptor cannot distinguish these tones.
    def sender(payload):
        index = int(payload['input'].split()[-1].rstrip('.'))
        t = np.arange(RATE) / RATE
        values = .15 * np.sin(np.linspace(0, np.pi, RATE)) ** 2 * np.sin(2 * np.pi * (180 + index * 50) * t)
        return pipeline.wav_bytes(values, RATE)
    pipeline.generate(rows, cfg, tmp_path, transport=sender)
    pipeline.assemble(rows, cfg, tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    raw, _ = pipeline.audio_read(tmp_path / 'master_raw.wav')
    clip = manifest['clips'][which]
    values = np.r_[raw[:clip['start_sample']], raw[clip['end_sample']:]]
    report, actual = extraction((tmp_path, manifest), values)
    assert report['accepted'] == 0, report
    assert all(row['alignment_status'] == 'quarantine' for row in actual)
    assert [row['id'] for row in actual] == [row['id'] for row in rows]


@pytest.mark.parametrize('field', ['max_cost', 'price_per_1000_characters'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_budget_rejected(tmp_path, field, value):
    rows = records(1)
    cfg = config(rows)
    cfg[field] = value
    with pytest.raises(ValueError, match='must be nonnegative'):
        pipeline.generate(rows, cfg, tmp_path, transport=lambda _: pytest.fail('invalid budget must not call API'))


def test_memory_preflight_rejects_large_declared_wav_before_decode(monkeypatch):
    import struct
    data = bytearray(pipeline.wav_bytes(tone(0), RATE)[:44])
    declared = 192000 * 7200 * 2
    struct.pack_into('<I', data, 40, declared)
    struct.pack_into('<I', data, 4, declared + 36)
    with pytest.raises(ValueError, match='working-set limit'):
        pipeline.audio_read(bytes(data))


def test_missing_first_plus_tail_padding_does_not_truncate_later_attacks(tmp_path):
    rows = records(6)
    cfg = config(rows)
    durations = [.25, 1, 2, 4, 8, 10]
    def sender(payload):
        index = int(payload['input'].split()[-1].rstrip('.'))
        n = round(durations[index] * RATE)
        # Fast attack makes a shifted start immediately observable.
        envelope = np.ones(n)
        envelope[:80] = np.linspace(0, 1, 80)
        envelope[-80:] = np.linspace(1, 0, 80)
        values = .15 * envelope * np.sin(np.arange(n) * 2 * np.pi * (180 + index * 40) / RATE)
        return pipeline.wav_bytes(values, RATE)
    pipeline.generate(rows, cfg, tmp_path, transport=sender)
    pipeline.assemble(rows, cfg, tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    raw, _ = pipeline.audio_read(tmp_path / 'master_raw.wav')
    clip = manifest['clips'][0]
    values = np.r_[raw[:clip['start_sample']], raw[clip['end_sample']:], np.zeros(clip['end_sample'] - clip['start_sample'])]
    report, actual = extraction((tmp_path, manifest), values)
    assert report['alignment_method'] == 'monotonic_separator_match'
    assert actual[0]['quality_status'] == 'quarantine'
    assert all(row['quality_status'] == 'accepted' for row in actual[1:])
    for index in range(1, 6):
        assert actual[index]['duration'] >= durations[index]  # No 170ms attack truncation.


def test_clipping_quarantines_only_affected_clip(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    clip = manifest['clips'][3]
    raw[(clip['active_start_sample'] + clip['active_end_sample']) // 2] = 1
    report, rows = extraction(master, raw)
    assert 'clipping' in rows[3]['reasons']
    assert rows[0]['quality_status'] == rows[-1]['quality_status'] == 'accepted'


def test_duplicate_segments_quarantined(tmp_path):
    rows = records(3)
    cfg = config(rows)
    pipeline.generate(rows, cfg, tmp_path, transport=lambda _: pipeline.wav_bytes(tone(0), RATE))
    pipeline.assemble(rows, cfg, tmp_path)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    report, actual = extraction((tmp_path, manifest))
    assert report['accepted'] == 0
    assert all('duplicate_audio' in row['reasons'] for row in actual)


def test_out_of_order_segments_do_not_accept_reassigned_audio(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    pieces = [raw[:manifest['clips'][0]['start_sample']]]
    for position, index in enumerate([0, 4, 2, 3, 1, 5]):
        clip = manifest['clips'][index]
        pieces.append(raw[clip['start_sample']:clip['end_sample']])
        if position < 5:
            pieces.append(np.zeros(round(2 * RATE)))
    pieces.append(raw[manifest['clips'][-1]['end_sample']:])
    report, actual = extraction(master, np.concatenate(pieces))
    assert actual[1]['quality_status'] == actual[4]['quality_status'] == 'quarantine', report
    assert actual[0]['quality_status'] == actual[-1]['quality_status'] == 'accepted'


def test_positive_local_shift_same_total_length_preserves_full_clips(master):
    directory, manifest = master
    raw, _ = pipeline.audio_read(directory / 'master_raw.wav')
    position = manifest['clips'][2]['end_sample']
    padding = round(.2 * RATE)
    values = np.r_[raw[:position], np.zeros(padding), raw[position:-padding]]
    report, actual = extraction(master, values)
    assert report['alignment_method'] == 'monotonic_separator_match'
    assert report['accepted'] == 6, report
    assert actual[-1]['duration'] >= (manifest['clips'][-1]['active_end_sample'] - manifest['clips'][-1]['active_start_sample']) / RATE + .18


def test_full_plan_declared_working_set_fits_bound():
    # Header-only arithmetic: no multi-hour audio allocation/generation.
    planned_seconds = 118.1 * 60 + (792 - 1) * 2 + 4
    assert planned_seconds * 24000 * 8 * 3 < pipeline.MAX_AUDIO_WORK_BYTES
    assert 7200 * 192000 * 8 * 3 > pipeline.MAX_AUDIO_WORK_BYTES
