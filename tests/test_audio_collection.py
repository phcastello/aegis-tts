"""Small mechanical fixtures only; these are never imported as a speech corpus."""
import csv
import json
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from training.audio_collection import prepare_collection, assignment_sha256, SCHEMA
from training.prepare_dataset import read_pcm, sha256


def pcm(path, values, width=3, rate=48000):
    integers = np.rint(np.asarray(values) * 2 ** (width * 8 - 1)).astype(np.int64)
    if width == 2:
        raw = integers.astype('<i2').tobytes()
    else:
        raw = np.column_stack((integers & 255, (integers >> 8) & 255, (integers >> 16) & 255)).astype(np.uint8).tobytes()
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(raw)


def fixture_dataset(tmp_path):
    root = tmp_path / 'originals'
    root.mkdir()
    rows = []
    for index, split in enumerate(('train', 'validation', 'test')):
        source = root / f'{split}.wav'
        values = .2 * np.sin(np.arange(48000) * (index + 1) * .031)
        pcm(source, values)
        rows.append({'schema': SCHEMA, 'id': f'item{index}', 'group_id': f'group{index}',
                     'split': split, 'text': ['Ative o protocolo de navegação.', 'Você verificou a segurança?',
                                             'São oito horas e quinze minutos.'][index],
                     'category': f'category{index}', 'audio_path': source.name,
                     'duration_seconds': 1., 'sample_rate': 48000, 'channels': 1,
                     'audio_sha256': sha256(source)})
    return root, rows


def run(tmp_path, rows, root):
    manifest = root / 'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return prepare_collection(manifest, tmp_path / 'prepared', root)


def test_pcm24_signed_decode_and_truncation(tmp_path):
    source = tmp_path / 'packed.wav'
    values = np.array([-.75, -.5, -.0001, 0., .0001, .5, .75])
    pcm(source, values)
    decoded, rate = read_pcm(source)
    assert rate == 48000
    np.testing.assert_allclose(decoded, values, atol=1/8388608)
    source.write_bytes(source.read_bytes()[:-2])
    with pytest.raises(ValueError, match='truncated'):
        read_pcm(source)


def test_conversion_preserves_original_and_never_grants_approval(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    originals = {r['id']: sha256(root / r['audio_path']) for r in rows}
    report = run(tmp_path, rows, root)
    assert report['counts'] == {'train': 1, 'validation': 1, 'test': 1}
    assert report['status'] == 'prepared-pending-review'
    assert not report['training_authorized']
    output = tmp_path / 'prepared'
    prepared = [json.loads(line) for line in (output/'dataset.jsonl').read_text().splitlines()]
    assert all(r['content_review'] == 'pending' for r in prepared)
    assert all(r['provenance']['master_manifest_status'] == 'unverified' for r in prepared)
    assert all(r['source_hash_status'] == 'verified-against-declared' for r in prepared)
    assert 'item2' not in (output/'metadata.csv').read_text()
    assert len((output/'metadata.csv').read_text().splitlines()) == 2
    for row in rows:
        assert sha256(root / row['audio_path']) == originals[row['id']]
    for row in prepared:
        data, rate = read_pcm(Path(row['audio_filepath']))
        assert rate == 22050 and len(data) == 22050
        assert abs(row['rms'] - row['source_rms']) < 1e-4
        assert Path(row['audio_filepath']).stat().st_mode & 0o777 == 0o400
    assert report['dataset_sha256'] == sha256(output/'dataset.jsonl')
    with pytest.raises(ValueError, match='already exists'):
        prepare_collection(root/'dataset.jsonl', output, root)


@pytest.mark.parametrize('change,reason', [
    ({'audio_sha256': '0'*64}, 'SHA256 mismatch'),
    ({'duration_seconds': 2}, 'metadata differs'),
    ({'sample_rate': 44100}, 'metadata differs'),
    ({'channels': 2}, 'mono'),
    ({'audio_path': '../escape.wav'}, 'escapes dataset root'),
    ({'audio_path': 'C:\\Windows\\secret.wav'}, 'absolute paths'),
    ({'audio_path': 'absent.wav'}, 'missing'),
    ({'schema': 'aegis.something-else.v1'}, 'wrong schema'),
    ({'text': 'pipe|injected'}, 'transcription'),
    ({'duration_seconds': float('nan')}, 'Out of range'),
    ({'path': 'different.wav'}, 'contradictory aliases'),
])
def test_invalid_rows_are_inventory_not_silent_omissions(tmp_path, change, reason):
    root, rows = fixture_dataset(tmp_path)
    rows[0].update(change)
    report = run(tmp_path, rows, root)
    assert report['status'] == 'blocked' and report['accepted'] == 2
    assert reason in str(report['rejected'])
    assert (tmp_path/'prepared/import-report.json').is_file()


def test_duplicate_ids_and_group_leakage_reject_both_sides(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[1]['id'] = rows[0]['id']
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 1
    assert len(report['rejected']) == 2
    assert all('duplicate stable id' in str(r) for r in report['rejected'])


def test_group_leakage_and_hash_leakage(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[1]['group_id'] = rows[0]['group_id']
    rows[1]['audio_path'] = rows[0]['audio_path']
    rows[1]['audio_sha256'] = rows[0]['audio_sha256']
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 1
    assert 'group crosses splits' in str(report['rejected'])
    assert 'identical audio crosses splits' in str(report['rejected'])


def test_exact_duplicate_text_crosses_splits(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[1]['text'] = rows[0]['text'].upper()
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 1
    assert 'duplicate normalized transcription' in str(report['rejected'])


def test_near_variants_block_until_review_without_changing_test(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[1]['text'] = 'Ative o protocolo de navegação! Agora.'
    rows[0]['text'] = 'Ative o protocolo de navegação? Agora.'
    # A word difference rather than punctuation-only duplicate.
    rows[1]['text'] = 'Ative o protocolo de navegação? Agore.'
    report = run(tmp_path, rows, root)
    assert report['near_variant_pair_count'] == 1 and report['status'] == 'blocked'
    assert report['counts']['test'] == 1


def test_missing_source_hash_computed_but_not_claimed_verified(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    del rows[0]['audio_sha256']
    report = run(tmp_path, rows, root)
    first = json.loads((tmp_path/'prepared/dataset.jsonl').read_text().splitlines()[0])
    assert first['source_hash_status'] == 'computed-no-prior-hash'
    assert len(first['original_audio_sha256']) == 64
    assert report['content_review'] == 'pending'


def test_pcm_rail_clipping_and_silence_are_quarantined(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    pcm(root/'train.wav', np.ones(48000) * -1)
    pcm(root/'validation.wav', np.zeros(48000))
    for row in rows:
        row['audio_sha256'] = sha256(root/row['audio_path'])
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 1
    assert 'clipping' in str(report['rejected']) and 'silent' in str(report['rejected'])


def test_real_provenance_hashes_and_absence(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    master = root/'master.wav'
    master.write_bytes((root/'train.wav').read_bytes())
    provenance = tmp_path/'provenance.json'
    provenance.write_text(json.dumps({'artifacts': [
        {'role': 'processed_master', 'path': 'master.wav', 'sha256': sha256(master)},
        {'role': 'source_master', 'path': 'absent-master.wav'}]}))
    manifest = root/'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = prepare_collection(manifest, tmp_path/'prepared', root, provenance)
    items = report['provenance']['artifacts']
    assert items[0]['status'] == 'verified'
    assert items[1]['status'] == 'unverified' and 'sha256' not in items[1]


def test_companion_split_conflict_is_blocked(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    (root/'train.jsonl').write_text(json.dumps(dict(rows[0], text='wrong text'))+'\n')
    report = run(tmp_path, rows, root)
    assert report['status'] == 'blocked'
    assert 'train.jsonl conflicts' in str(report['structural_errors'])


@pytest.mark.parametrize('offset,value', [(28, 123), (32, 4)])
def test_wav_inconsistent_byte_rate_and_block_alignment_rejected(tmp_path, offset, value):
    source = tmp_path / 'bad-format.wav'
    pcm(source, np.zeros(4800), width=2)
    data = bytearray(source.read_bytes())
    struct.pack_into('<I' if offset == 28 else '<H', data, offset, value)
    source.write_bytes(data)
    with pytest.raises(ValueError, match='format metadata'):
        read_pcm(source)


def test_extensible_pcm24_reads_full_width_signed_samples(tmp_path):
    source = tmp_path / 'extensible.wav'
    values = np.array([-.75, -.125, 0., .125, .75, 0.])
    pcm(source, values)
    original = source.read_bytes()
    pcm_guid = bytes.fromhex('0100000000001000800000aa00389b71')
    fmt = struct.pack('<HHIIHHHHI', 0xfffe, 1, 48000, 144000, 3, 24, 22, 24, 4) + pcm_guid
    body = b'WAVEfmt ' + struct.pack('<I', len(fmt)) + fmt + original[36:]
    source.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)
    decoded, rate = read_pcm(source)
    assert rate == 48000
    np.testing.assert_allclose(decoded, values, atol=1/8388608)
    # Packed 24 bits with a different valid-bit count needs a different scale;
    # refusing it avoids silently attenuating the owner's audio.
    malformed = bytearray(source.read_bytes())
    struct.pack_into('<H', malformed, 38, 20)
    source.write_bytes(malformed)
    with pytest.raises(ValueError, match='full-width'):
        read_pcm(source)


def test_explicit_collection_adapter_accepts_rows_without_individual_schema(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    for row in rows:
        del row['schema']
    assert run(tmp_path, rows, root)['accepted'] == 3


@pytest.mark.parametrize('text', ['Aegis disse: "Sistemas prontos".',
                                 '"Sistemas prontos", disse Aegis.',
                                 'Ela respondeu "sim"', '"Sim"'])
def test_metadata_has_exact_two_columns_and_retains_reviewed_quotes(tmp_path, text):
    root, rows = fixture_dataset(tmp_path)
    rows[0]['text'] = text
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 3
    with (tmp_path / 'prepared/metadata.csv').open(newline='') as stream:
        parsed = list(csv.reader(stream, delimiter='|'))
    assert parsed[0] == ['item0.wav', rows[0]['text']]
    assert all(len(row) == 2 for row in parsed)


def test_conflicting_schema_names_are_inventory(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[0]['schema_version'] = 'different'
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 2
    assert 'contradictory schema' in str(report['rejected'])


def test_resampling_overshoot_is_quarantined_without_limiter_or_source_change(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    # A discontinuous, loud signal demonstrates that filtering can overshoot
    # even when no source sample touches a PCM rail.
    values = np.where(np.arange(48000) % 100 < 50, .95, -.95)
    pcm(root / 'train.wav', values)
    rows[0]['audio_sha256'] = sha256(root / 'train.wav')
    report = run(tmp_path, rows, root)
    assert report['accepted'] == 2
    assert 'clipped audio' in str(report['rejected'])
    assert sha256(root / 'train.wav') == rows[0]['audio_sha256']
    assert not (tmp_path / 'prepared/audio/item0.wav').exists()


def test_reduced_windows_companions_keep_identity_from_primary_manifest(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    for row in rows:
        companion = {'audio_filepath': row['audio_path'], 'text': row['text'],
                     'duration': row['duration_seconds']}
        (root / (row['split'] + '.jsonl')).write_text(json.dumps(companion) + '\n')
    report = run(tmp_path, rows, root)
    assert report['status'] == 'prepared-pending-review'
    assert report['accepted'] == 3


@pytest.mark.parametrize('change,reason', [
    ({'audio_filepath': 'test.wav'}, 'does not identify'),
    ({'text': 'Transcrição diferente'}, 'text/split'),
    ({'duration': 2.}, 'duration differs'),
    ({'group_id': 'different-group'}, 'group_id differs'),
])
def test_reduced_companion_conflicts_block_training(tmp_path, change, reason):
    root, rows = fixture_dataset(tmp_path)
    row = rows[0]
    companion = {'audio_filepath': row['audio_path'], 'text': row['text'], 'duration': 1.}
    companion.update(change)
    (root / 'train.jsonl').write_text(json.dumps(companion) + '\n')
    report = run(tmp_path, rows, root)
    assert report['status'] == 'blocked'
    assert reason in str(report['structural_errors'])


def split_plan_fixture(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    fourth = dict(rows[0], id='item3', group_id='group3', audio_path='extra.wav',
                  text='A frequência e o timbre permanecem consistentes.')
    pcm(root / fourth['audio_path'], .15 * np.sin(np.arange(48000) * .071))
    fourth['audio_sha256'] = sha256(root / fourth['audio_path'])
    rows.append(fourth)
    manifest = root / 'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assignments = {r['id']: r['split'] for r in rows}
    assignments['item3'] = 'validation'
    plan = {'source_manifest_sha256': sha256(manifest), 'assignments': assignments,
            'split_assignment_sha256': assignment_sha256(assignments), 'seed': 40733133,
            'rule': 'add group3 from train to validation'}
    return root, rows, manifest, plan


def test_hash_bound_split_plan_preserves_sources_and_final_test(tmp_path):
    root, rows, manifest, plan = split_plan_fixture(tmp_path)
    original_bytes = manifest.read_bytes()
    # Companion verification must use the original primary partitions.
    for split in ('train', 'validation', 'test'):
        reduced = [dict(audio_filepath=r['audio_path'], text=r['text'], duration=r['duration_seconds'])
                   for r in rows if r['split'] == split]
        (root / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in reduced))
    plan_path = tmp_path / 'split-plan.json'
    plan_path.write_text(json.dumps(plan))
    report = prepare_collection(manifest, tmp_path / 'prepared', root, split_plan=plan_path)
    assert report['counts'] == {'train': 1, 'validation': 2, 'test': 1}
    assert report['status'] == 'prepared-pending-review'
    assert report['split_plan']['final_test_unchanged']
    assert report['split_plan']['moved_ids'] == ['item3']
    assert report['split_plan']['split_assignment_sha256'] == plan['split_assignment_sha256']
    prepared = [json.loads(l) for l in (tmp_path / 'prepared/dataset.jsonl').read_text().splitlines()]
    moved = next(r for r in prepared if r['id'] == 'item3')
    assert moved['split'] == 'validation' and moved['source_split'] == 'train'
    assert moved['source_record']['split'] == 'train'
    assert manifest.read_bytes() == original_bytes


@pytest.mark.parametrize('change,reason', [
    ('source_hash', 'exact primary manifest'),
    ('assignment_hash', 'assignment SHA256 mismatch'),
    ('missing_id', 'every unique primary ID'),
    ('move_test', 'preserve original test'),
    ('new_test', 'preserve original test'),
    ('move_old_validation', 'preserve original test/validation'),
    ('break_group', 'breaks a primary group'),
    ('hide_source_leak', 'primary group already crosses'),
])
def test_split_plan_rejects_incomplete_unbound_or_leaking_changes(tmp_path, change, reason):
    root, rows, manifest, plan = split_plan_fixture(tmp_path)
    if change == 'source_hash':
        plan['source_manifest_sha256'] = '0' * 64
    elif change == 'assignment_hash':
        plan['split_assignment_sha256'] = '0' * 64
    elif change == 'missing_id':
        del plan['assignments']['item0']
    elif change == 'move_test':
        plan['assignments']['item2'] = 'train'
    elif change == 'new_test':
        plan['assignments']['item0'] = 'test'
    elif change == 'move_old_validation':
        plan['assignments']['item1'] = 'train'
    elif change in ('break_group', 'hide_source_leak'):
        rows[3]['group_id'] = rows[0 if change == 'break_group' else 1]['group_id']
        manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        plan['source_manifest_sha256'] = sha256(manifest)
    if change != 'assignment_hash':
        plan['split_assignment_sha256'] = assignment_sha256(plan['assignments'])
    plan_path = tmp_path / 'split-plan.json'
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=reason):
        prepare_collection(manifest, tmp_path / 'prepared', root, split_plan=plan_path)
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('architecture,rate', [('f5', 24000), ('magpie', 22050)])
def test_architecture_export_preserves_raw_transcripts_groups_and_originals(tmp_path, architecture, rate):
    root, rows = fixture_dataset(tmp_path)
    rows[0]['text'] = '  Amanhã, Júlia chegará às 07h35.  '
    manifest = root / 'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    original_hashes = [sha256(root / row['audio_path']) for row in rows]
    output = tmp_path / 'exported'
    report = prepare_collection(manifest, output, root, architecture=architecture)
    assert report['status'] == 'prepared-pending-review'
    assert report['architecture_manifest_status'] == 'intermediate-not-training-ready'
    assert report['sample_rate'] == rate
    assert report['training_authorized'] is False
    prepared = [json.loads(line) for line in (output/'dataset.jsonl').read_text().splitlines()]
    assert prepared[0]['source_record']['text'] == rows[0]['text']
    assert prepared[0]['text'] == rows[0]['text'].strip()
    assert [sha256(root / row['audio_path']) for row in rows] == original_hashes
    for split in ('train', 'validation', 'test'):
        native = [json.loads(line) for line in (output/f'{architecture}-{split}.jsonl').read_text().splitlines()]
        expected = [row for row in prepared if row['split'] == split]
        assert [row['id'] for row in native] == [row['id'] for row in expected]
        assert [row['group_id'] for row in native] == [row['group_id'] for row in expected]
        for row in native:
            values, actual = read_pcm(Path(row['audio_filepath']))
            assert actual == rate and len(values) == rate
            assert 'context_audio_filepath' not in row
            assert row['duration'] == 1.
    rights = json.loads((output/'training-rights.pending.json').read_text())
    assert rights['dataset_sha256'] == report['dataset_sha256']
    assert rights['source_manifest_sha256'] == sha256(manifest)
    assert not any(rights[key] for key in ('training_authorized', 'training_rights_confirmed', 'voice_rights_confirmed'))
    assert any('ElevenLabs' in gate for gate in report['training_gates'])
    with pytest.raises(ValueError, match='already exists'):
        prepare_collection(manifest, output, root, architecture=architecture)


def test_f5_export_keeps_cross_partition_groups_blocked(tmp_path):
    root, rows = fixture_dataset(tmp_path)
    rows[1]['group_id'] = rows[0]['group_id']
    manifest = root / 'dataset.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = prepare_collection(manifest, tmp_path/'f5', root, architecture='f5')
    assert report['status'] == 'blocked'
    assert report['accepted'] == 1
    assert len(report['rejected']) == 2
    assert (tmp_path/'f5/f5-train.jsonl').read_text() == ''
