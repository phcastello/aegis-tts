import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import wave

import numpy as np
import pytest

from training.prepare_dataset import prepare, select_records, write_pcm, sha256, resample_pcm
from training.piper_adapter import PiperUnitAdapter
from training.train_piper import partition_indices


def record(identifier='a', split='train', **extra):
    row = {'id': identifier, 'text': 'Não corte a primeira palavra.', 'split': split,
           'quality_status': 'accepted', 'alignment_status': 'aligned', 'content_review': 'approved',
           'source_master_sha256': '1'*64, 'processed_master_sha256': '2'*64}
    row.update(extra)
    return row


def test_import_requires_approved_processed_content_and_canonical_groups():
    rows, excluded = select_records([record(), record('b', content_review='pending'),
                                    record('c', quality_status='quarantine')])
    assert len(rows) == 1 and len(excluded) == 2
    with pytest.raises(ValueError, match='group crosses splits'):
        select_records([record(group_id='same'), record('b','test',group_id='same')])
    with pytest.raises(ValueError, match='provenance'):
        select_records([record(source_master_sha256=None)])


def test_import_does_not_silently_rewrite_numbers():
    rows, _ = select_records([record(text='São 08:45 em 01/11/2026.')])
    assert rows[0]['text'] == 'São 08:45 em 01/11/2026.'


def test_import_resamples_and_keeps_heldout_test_out_of_trainer(tmp_path):
    source = tmp_path/'source.wav'
    values = .2*np.sin(np.arange(24000)*2*np.pi*230/24000)
    write_pcm(source, values, 24000)
    rows = [record('a','train'),record('b','validation'),record('c','test')]
    for row in rows:
        row.update(audio_filepath=str(source),duration=1,sha256=sha256(source))
    manifest = tmp_path/'source.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = prepare(manifest,tmp_path/'prepared')
    assert summary['counts'] == {'train':1,'validation':1,'test':1}
    trainval = [json.loads(line) for line in (tmp_path/'prepared/trainval.jsonl').read_text().splitlines()]
    assert [r['split'] for r in trainval] == ['train','validation']
    with wave.open(str(tmp_path/'prepared/audio/a.wav')) as audio:
        assert audio.getframerate() == 22050 and audio.getnframes() == 22050
    assert partition_indices(trainval) == {'train':[0],'validation':[1]}
    with pytest.raises(ValueError, match='final test'):
        partition_indices(rows)


def test_import_rejects_changed_hash_and_clipping(tmp_path):
    source = tmp_path/'a.wav';write_pcm(source,np.zeros(24000),24000)
    manifest = tmp_path/'input.jsonl'
    manifest.write_text(json.dumps(record(audio_filepath=str(source),duration=1,sha256='0'*64)))
    with pytest.raises(ValueError,match='hash mismatch'):
        prepare(manifest,tmp_path/'out')
    with pytest.raises(ValueError,match='clipped'):
        write_pcm(tmp_path/'bad.wav',np.array([1.1]),24000)


class Voice:
    def synthesize(self,text):
        yield SimpleNamespace(sample_channels=1,sample_rate=22050,
                              audio_float_array=.1*np.sin(np.arange(22050)*.04))


def test_internal_adapter_public_rate_duration_and_cancellation():
    adapter = PiperUnitAdapter(Voice())
    chunks = list(adapter.stream_unit('Teste.'))
    assert all(len(c)%2 == 0 and len(c)<=2400 for c in chunks)
    assert len(b''.join(chunks)) == 48000
    assert list(adapter.stream_unit('Teste.',lambda:True)) == []
    assert len(b''.join(adapter.stream_unit('Outra unidade.'))) == 48000
    with pytest.raises(ValueError,match='too long'):
        list(adapter.stream_unit('x'*1201))


def test_adapter_releases_sequential_worker_after_cancel():
    adapter = PiperUnitAdapter(Voice())
    stream = adapter.stream_unit('A.'); next(stream);stream.close()
    assert adapter._lock.acquire(blocking=False)
    adapter._lock.release()


def test_resampling_does_not_change_playback_speed():
    values = np.sin(np.arange(22050)*2*np.pi*200/22050)*.1
    out = resample_pcm(values,22050,24000)
    frequency = np.argmax(abs(np.fft.rfft(out)))
    assert frequency == 200 and len(out) == 24000


def test_frontend_coverage_fails_unknown_symbols_without_dropping_them():
    from training.frontend_check import check_inventory
    config = {'espeak': {'voice': 'pt-br'}, 'phoneme_id_map': {'a':[14],'ɲ':[82],'̃':[141]}}
    passed = check_inventory([record()],config,lambda _: [['a','ɲ','̃']])
    assert passed['passed']
    failed = check_inventory([record()],config,lambda _: [['a','UNSUPPORTED']])
    assert not failed['passed'] and failed['unknown_by_id'] == {'a':['UNSUPPORTED']}


@pytest.mark.asyncio
async def test_adapter_matches_existing_service_signature_and_isolation():
    import asyncio
    adapter = PiperUnitAdapter(Voice())
    first = SimpleNamespace(normalized_text='Primeira.',request_id='one')
    second = SimpleNamespace(normalized_text='Segunda.',request_id='two')
    cancel = asyncio.Event()
    chunks = []
    async for block in adapter(first,cancel):
        chunks.append(block)
        cancel.set()
    assert len(chunks) == 1
    independent = [block async for block in adapter(second,asyncio.Event())]
    assert len(b''.join(independent)) == 48000


def test_training_guard_never_shares_the_production_gpu(monkeypatch):
    from training import train_piper
    monkeypatch.setattr(train_piper.subprocess,'run',
                        lambda *args,**kwargs:SimpleNamespace(stdout='840\n'))
    with pytest.raises(RuntimeError,match='exclusive GPU capacity unavailable'):
        train_piper.gpu_available()


def test_pcm16_rounding_never_wraps_near_positive_fullscale(tmp_path):
    from training.prepare_dataset import encode_pcm16
    for value in (.99999, .9999999):
        with pytest.raises(ValueError,match='PCM16 quantization'):
            write_pcm(tmp_path/'would-wrap.wav',np.array([value]),24000)
        assert not (tmp_path/'would-wrap.wav').exists()
    pcm = np.frombuffer(encode_pcm16(np.array([32767/32768, -.99999])),dtype='<i2')
    assert list(pcm) == [32767,-32768]


def test_adapter_rejects_quantization_overflow_without_a_wrapped_chunk():
    class OverflowVoice:
        def synthesize(self,text):
            yield SimpleNamespace(sample_channels=1,sample_rate=24000,
                                  audio_float_array=np.array([0.,.99999,0.]))
    with pytest.raises(ValueError,match='PCM16 quantization'):
        list(PiperUnitAdapter(OverflowVoice()).stream_unit('Overflow.'))


def test_training_import_requires_treated_master_provenance():
    with pytest.raises(ValueError,match='processed_master_sha256'):
        select_records([record(processed_master_sha256='')])
