#!/usr/bin/env python3
"""Plan by default: export/evaluate one private pilot candidate, never deploy it.

The local benchmark measures complete-sentence ONNX computation followed by PCM
delivery. It is not the production API's queue, streaming or cancellation test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import wave

import numpy as np

from .audio_collection import normalized_text
from .piper_adapter import PiperUnitAdapter
from .prepare_dataset import REPOSITORY, sha256
from .train_piper import BASE_SHA256

RATE = 24000
ACTIVITY_THRESHOLD_PCM16 = 100
# These are inference probes only. They are never added to training input.
PROMPTS = (
    ('statement', 'Os sensores da estação orbital estão prontos para uma nova manobra de aproximação.'),
    ('question_times', 'A reunião foi remarcada para as 07:35, ou permanece às 18:20?'),
    ('names', 'Joana e Guilherme enviaram o relatório de Florianópolis para João Pessoa.'),
    ('acronyms', 'Valide a conexão HTTPS, confira a CPU e registre o código XML no painel de diagnóstico.'),
    ('numbers_negation', 'Não foram encontrados 27 arquivos; o resultado correto foi de 143 documentos e 2,5 gigabytes.'),
    ('long', 'Antes de iniciar a próxima etapa, confirme a leitura dos sensores e mantenha o canal de comunicação aberto. '
     'A equipe de manutenção vai revisar os registros de temperatura, conferir os horários das últimas mensagens e comparar '
     'os resultados com as medições feitas na manhã anterior. Se alguma informação estiver incompleta, explique o que falta '
     'com clareza e aguarde uma nova instrução. Não repita uma ação já concluída, não omita os valores solicitados e preserve '
     'a sequência do relatório até o encerramento da transmissão.'),
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validate_sources(checkpoint: Path, index: Path, config: Path, prepared: Path) -> dict:
    checkpoint, index, config, prepared = (p.resolve() for p in (checkpoint, index, config, prepared))
    experiment = index.parent
    if index.name != 'checkpoint-index.json' or checkpoint.parent != experiment / 'checkpoints':
        raise ValueError('checkpoint must belong to the experiment/checkpoints named by checkpoint-index.json')
    entry = json.loads(index.read_text()).get(checkpoint.name, {})
    if not checkpoint.is_file() or entry.get('bytes') != checkpoint.stat().st_size or entry.get('sha256') != sha256(checkpoint):
        raise ValueError('checkpoint bytes/SHA256 differ from the experiment index')
    run_path = experiment / 'run.json'
    run = json.loads(run_path.read_text())
    if run.get('benchmark') is not False or run.get('base_sha256') != BASE_SHA256:
        raise ValueError('evaluation requires a real pilot experiment from the pinned base, not a timing benchmark')
    if run.get('prepared_sha256') != sha256(prepared / 'trainval.jsonl'):
        raise ValueError('prepared train/validation manifest differs from the pilot experiment')
    parameters = json.loads(config.read_text())
    experiment_config = experiment / 'AegisVoicev2.0.onnx.json'
    trained_config = json.loads(experiment_config.read_text())
    for key in ('phoneme_id_map', 'num_symbols', 'num_speakers', 'espeak', 'audio'):
        if parameters.get(key) != trained_config.get(key):
            raise ValueError('voice configuration differs from the trained frontend: ' + key)
    for key, default in [('phoneme_type', 'espeak'), ('vowel_clusters', None), ('hop_length', 256)]:
        if parameters.get(key, default) != trained_config.get(key, default):
            raise ValueError('voice configuration differs from the trained frontend: ' + key)
    if (parameters.get('espeak', {}).get('voice') != 'pt-br' or
            parameters.get('audio', {}).get('sample_rate') != 22050 or
            parameters.get('num_symbols') != 256 or parameters.get('num_speakers') != 1 or
            not parameters.get('phoneme_id_map') or parameters.get('phoneme_type', 'espeak') != 'espeak'):
        raise ValueError('expected the trained single-speaker pt-br/22050 frontend')
    rows = read_rows(prepared / 'dataset.jsonl')
    trainval = read_rows(prepared / 'trainval.jsonl')
    imported = json.loads((prepared / 'import-report.json').read_text())
    if (imported.get('dataset_sha256') != sha256(prepared / 'dataset.jsonl') or
            imported.get('trainval_sha256') != sha256(prepared / 'trainval.jsonl') or
            imported.get('status') == 'blocked'):
        raise ValueError('immutable import bindings are stale or blocked')
    if trainval != [row for row in rows if row['split'] != 'test']:
        raise ValueError('training manifest is not the exact train/validation projection')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate dataset ID')
    with (prepared / 'metadata.csv').open(newline='') as stream:
        if list(csv.reader(stream, delimiter='|')) != [[Path(r['audio_filepath']).name, r['text']] for r in trainval]:
            raise ValueError('prepared metadata differs from the trained manifest')
    return {'checkpoint': str(checkpoint), 'checkpoint_sha256': entry['sha256'],
            'checkpoint_bytes': entry['bytes'], 'epoch': entry.get('epoch'), 'global_step': entry.get('global_step'),
            'checkpoint_index': str(index), 'checkpoint_index_sha256': sha256(index),
            'run': str(run_path), 'run_sha256': sha256(run_path),
            'voice_config': str(config), 'voice_config_sha256': sha256(config),
            'experiment_config': str(experiment_config), 'experiment_config_sha256': sha256(experiment_config),
            'prepared': str(prepared), 'dataset_sha256': sha256(prepared / 'dataset.jsonl'),
            'trainval_sha256': sha256(prepared / 'trainval.jsonl')}


def evaluation_plan(checkpoint: Path, index: Path, config: Path, prepared: Path,
                    reference_ids=(), production_url=None) -> dict:
    bindings = validate_sources(checkpoint, index, config, prepared)
    rows = read_rows(prepared / 'dataset.jsonl')
    texts = {normalized_text(row['text']) for row in rows}
    if any(normalized_text(text) in texts for _, text in PROMPTS):
        raise ValueError('an inference probe already occurs in the dataset; choose an unseen probe before evaluation')
    by_id = {row['id']: row for row in rows}
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError('reference IDs must be unique')
    for identifier in reference_ids:
        if identifier not in by_id or by_id[identifier]['split'] not in ('train', 'validation'):
            raise ValueError('A/B reference must be an existing train/validation clip; final test stays untouched')
    if production_url:
        url = urlsplit(production_url)
        if (url.scheme not in ('http', 'https') or not url.netloc or url.username or url.password or
                url.query or url.fragment or url.path not in ('/v1/audio/speech', '/v1/aegis/speech')):
            raise ValueError('production-url must be the explicit full speech endpoint, without credentials/query')
    return {'status': 'plan-only', 'bindings': bindings,
            'probes': [{'id': name, 'text': text, 'purpose': 'unseen inference only'} for name, text in PROMPTS],
            'reference_ids': list(reference_ids), 'production_url': production_url,
            'output_format': 'mono PCM16 WAV at 24000 Hz through PiperUnitAdapter',
            'execution_requires': ['--execute and explicit owner authorization',
                                   'production comparison also requires its explicit URL and separate authorization'],
            'promotion': False}


def activity_metrics(pcm: bytes, arrivals: list[dict], total_seconds: float) -> dict:
    if len(pcm) % 2 or sum(chunk['bytes'] for chunk in arrivals) != len(pcm):
        raise ValueError('unaligned PCM or inconsistent delivery inventory')
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.int32)
    indices = np.flatnonzero(np.abs(samples) > ACTIVITY_THRESHOLD_PCM16)
    first_activity = int(indices[0]) if len(indices) else None
    delivered = 0
    playback_end = None
    available = playback = None
    for chunk in arrivals:
        if chunk['bytes'] % 2:
            raise ValueError('each delivery must contain complete PCM16 samples')
        count = chunk['bytes'] // 2
        start = max(chunk['seconds'], playback_end) if playback_end is not None else chunk['seconds']
        if first_activity is not None and available is None and delivered <= first_activity < delivered + count:
            available = chunk['seconds']
            playback = start + (first_activity - delivered) / RATE
        playback_end = start + count / RATE
        delivered += count
    seconds = len(samples) / RATE
    return {'first_chunk_wall_seconds': arrivals[0]['seconds'] if arrivals else None,
            'first_activity_available_wall_seconds': available,
            'leading_pcm_silence_seconds': first_activity / RATE if first_activity is not None else None,
            'estimated_first_audible_playback_wall_seconds': playback,
            'total_wall_seconds': total_seconds, 'audio_seconds': seconds,
            'real_time_factor': total_seconds / seconds if seconds else None,
            'peak_pcm16': int(np.max(np.abs(samples), initial=0)),
            'rms_pcm16': float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.,
            'pcm_rail_samples': int(np.count_nonzero((samples == -32768) | (samples == 32767))),
            'activity_threshold_pcm16': ACTIVITY_THRESHOLD_PCM16,
            'activity_definition': 'first sample above threshold; activity proxy, not a perceptual intelligibility judgment',
            'playback_assumption': 'immediate playback respecting arrivals, excluding client/device delay'}


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(pcm)
    path.chmod(0o600)


def capture(chunks, *, clock=time.perf_counter) -> tuple[bytes, dict]:
    started, cpu_started = clock(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts, arrivals, received = [], [], 0
    for block in chunks:
        if not isinstance(block, bytes) or len(block) % 2:
            raise ValueError('expected aligned PCM16 byte chunks')
        if not block:
            continue
        parts.append(block)
        received += len(block)
        arrivals.append({'seconds': clock() - started, 'bytes': len(block)})
        if received > RATE * 2 * 120:
            raise ValueError('evaluation exceeded 120 seconds of audio')
    elapsed = clock() - started
    pcm = b''.join(parts)
    if not pcm:
        raise ValueError('candidate generated empty audio')
    if not np.any(np.frombuffer(pcm, dtype='<i2')):
        raise ValueError('generated audio is entirely silent')
    return pcm, {**activity_metrics(pcm, arrivals, elapsed), 'chunks': arrivals,
                 'process_cpu_seconds': time.process_time() - cpu_started,
                 'host_max_rss_before_mib': before,
                 'host_max_rss_after_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                 'memory_note': 'Linux process lifetime peak RSS, includes model load/prior requests; not an isolated per-request peak'}


def export_candidate(bindings: dict, output: Path) -> tuple[Path, Path, dict]:
    model, config = output / 'AegisVoicev2.0-candidate.onnx', output / 'AegisVoicev2.0-candidate.onnx.json'
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-m', 'piper.train.export_onnx', '--checkpoint', bindings['checkpoint'],
               '--output-file', str(model)]
    pairs = [('checkpoint', 'checkpoint_sha256'), ('checkpoint_index', 'checkpoint_index_sha256'),
             ('run', 'run_sha256'), ('voice_config', 'voice_config_sha256'),
             ('experiment_config', 'experiment_config_sha256')]
    def verify():
        for key, digest in pairs:
            if sha256(Path(bindings[key])) != bindings[digest]:
                raise ValueError('source changed before/during export: ' + key)
        prepared = Path(bindings['prepared'])
        for filename, key in [('dataset.jsonl', 'dataset_sha256'), ('trainval.jsonl', 'trainval_sha256')]:
            if sha256(prepared / filename) != bindings[key]:
                raise ValueError('prepared input changed before/during export: ' + filename)
    verify()
    started = time.perf_counter()
    with (output / 'export.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    (output / 'export.log').chmod(0o600)
    verify()
    if not model.is_file() or not model.stat().st_size:
        raise ValueError('upstream export did not produce a model')
    model.chmod(0o600)
    shutil.copyfile(bindings['voice_config'], config)
    config.chmod(0o600)
    return model, config, {'command': command, 'wall_seconds': time.perf_counter() - started,
                           'model_sha256': sha256(model), 'config_sha256': sha256(config),
                           'device': 'cpu', 'private_export': True}


def production_chunks(url: str | None, authorization: str | None, text: str):
    if not url or not authorization or not authorization.strip():
        raise ValueError('production comparison requires explicit URL and separate authorization')
    payload = {'stream': True, 'response_format': 'pcm'}
    if urlsplit(url).path == '/v1/aegis/speech':
        payload.update(text=text, priority=50, interrupt_policy='enqueue')
    elif urlsplit(url).path == '/v1/audio/speech':
        payload.update(input=text)
    else:
        raise ValueError('unsupported production speech endpoint')
    request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=120) as response:
        if (response.headers.get('X-Aegis-Sample-Rate') != '24000' or
                response.headers.get('X-Aegis-Channels') != '1' or
                response.headers.get('X-Aegis-Audio-Format') != 'pcm_s16le'):
            raise ValueError('production response does not confirm the public mono/24000/PCM16 contract')
        pending = b''
        while True:
            block = response.read1(2400)
            if not block:
                break
            pending += block
            complete = len(pending) // 2 * 2
            if complete:
                yield pending[:complete]
                pending = pending[complete:]
        if pending:
            raise ValueError('production returned a truncated PCM16 sample')


def cancellation_probe(adapter, text: str) -> dict:
    cancelled = False
    stream = adapter.stream_unit(text, lambda: cancelled)
    started = time.perf_counter()
    try:
        first = next(stream, b'')
        cancelled = True
        requested = time.perf_counter()
        extra = list(stream)
    finally:
        stream.close()
    return {'first_chunk_bytes': len(first), 'bytes_after_cancellation': sum(map(len, extra)),
            'first_chunk_seconds': requested - started,
            'cancellation_completion_seconds': time.perf_counter() - requested,
            'scope': 'cancel after first block; does not abort or time cancellation inside a running ONNX calculation'}


def evaluate(plan: dict, output: Path, *, authorization: str,
             production_authorization=None, check_cancellation=False, adapter_factory=PiperUnitAdapter.load) -> dict:
    if not authorization or not authorization.strip():
        raise ValueError('explicit owner authorization required')
    if plan['production_url'] and not production_authorization:
        raise ValueError('explicit separate production authorization required')
    output = output.resolve()
    if output == REPOSITORY or REPOSITORY in output.parents or output.exists():
        raise ValueError('evaluation requires a new private directory outside the repository')
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    report = {'status': 'running', 'bindings': plan['bindings'], 'owner_authorization': authorization,
              'production_authorization': production_authorization if plan['production_url'] else None,
              'production_url': plan['production_url'], 'promoted': False,
              'runtime': 'ONNX CPUExecutionProvider, one intra/inter-op thread, public mono PCM16/24000',
              'normalization': 'disabled explicitly in PiperUnitAdapter.load; no limiter or effects applied',
              'limitations': ['first PCM follows full first-sentence inference; subsequent 50ms blocks are unpaced transport',
                              'local inference excludes HTTP, queueing and the existing API cancellation contract',
                              'metrics cannot approve identity, robotic texture, reverb, pronunciation or intelligibility',
                              'one measurement per prompt; no statistical latency guarantee or automatic promotion'],
              'samples': [], 'reference_pairs': []}
    def save():
        # Reuse the local ASR auditor on generated evaluation audio if desired.
        # Its explicit partition prevents this artifact from becoming training
        # input; it records technical readability, never content approval.
        generated = report['samples'] + [pair['candidate'] for pair in report['reference_pairs']]
        audit_rows = [{'id': row['kind'] + '.' + row['id'], 'text': row['text'],
                       'audio_filepath': row['waveform'], 'sha256': row['sha256'],
                       'duration': row['audio_seconds'], 'quality_status': 'accepted',
                       'content_review': 'pending', 'split': 'evaluation-only',
                       'purpose': 'candidate inference audit; never training input'}
                      for row in generated if row['status'] == 'rendered']
        audit_path = output / 'generated-evaluation.jsonl'
        audit_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in audit_rows))
        audit_path.chmod(0o600)
        report['generated_audit_manifest'] = {'path': str(audit_path), 'sha256': sha256(audit_path),
                                             'examples': len(audit_rows), 'asr_executed': False}
        path = output / 'evaluation.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        path.chmod(0o600)
    try:
        model, config, report['export'] = export_candidate(plan['bindings'], output)
        started = time.perf_counter()
        adapter = adapter_factory(model, config, report['export']['model_sha256'])
        adapter.max_audio_seconds = 120
        report['model_load_wall_seconds'] = time.perf_counter() - started
        def render(identifier, text, prefix):
            record = {'id': identifier, 'text': text, 'kind': prefix}
            try:
                pcm, metrics = capture(adapter.stream_unit(text))
                target = output / (prefix + '-' + identifier + '.wav')
                write_wav(target, pcm)
                record.update(status='rendered', waveform=str(target), sha256=sha256(target), **metrics)
                if prefix == 'candidate' and identifier == plan['probes'][0]['id']:
                    record['model_load_plus_first_chunk_seconds'] = report['model_load_wall_seconds'] + metrics['first_chunk_wall_seconds']
            except (ValueError, RuntimeError) as error:
                record.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            return record
        for probe in plan['probes']:
            candidate = render(probe['id'], probe['text'], 'candidate')
            report['samples'].append(candidate)
            if plan['production_url']:
                pcm, metrics = capture(production_chunks(plan['production_url'], production_authorization, probe['text']))
                target = output / ('production-' + probe['id'] + '.wav')
                write_wav(target, pcm)
                candidate['production_comparison'] = {**metrics, 'waveform': str(target), 'sha256': sha256(target),
                                                      'scope': 'functional API comparison; public transport includes queue/network overhead'}
            save()
        rows = {row['id']: row for row in read_rows(Path(plan['bindings']['prepared']) / 'dataset.jsonl')}
        for identifier in plan['reference_ids']:
            row = rows[identifier]
            source = Path(row['original_audio_filepath'])
            if row['split'] == 'test' or sha256(source) != row['original_audio_sha256']:
                raise ValueError('reference source changed or belongs to final test')
            target = output / ('target-original-' + identifier + '.wav')
            shutil.copyfile(source, target)
            target.chmod(0o600)
            pair = {'id': identifier, 'split': row['split'], 'text': row['text'],
                    'target_original': str(target), 'target_original_sha256': sha256(target),
                    'candidate': render(identifier, row['text'], 'reference'),
                    'purpose': 'same-text target/candidate A/B; supervised examples are not evidence of novel-text generalization'}
            report['reference_pairs'].append(pair)
            save()
        if check_cancellation:
            report['cancellation'] = cancellation_probe(adapter, dict(PROMPTS)['long'])
        failed = any(row['status'] == 'failed' for row in report['samples'])
        failed |= any(row['candidate']['status'] == 'failed' for row in report['reference_pairs'])
        report['status'] = 'evaluation-has-failures' if failed else 'awaiting-private-perceptual-review'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        save()
        raise
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'checkpoint-index', 'voice-config', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference-id', action='append', default=[])
    parser.add_argument('--production-url', help='explicit full /v1/audio/speech or /v1/aegis/speech endpoint')
    parser.add_argument('--production-authorization')
    parser.add_argument('--owner-authorization')
    parser.add_argument('--check-cancellation', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        plan = evaluation_plan(args.checkpoint, args.checkpoint_index, args.voice_config, args.prepared,
                               args.reference_id, args.production_url)
        if not args.execute:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        result = evaluate(plan, args.output, authorization=args.owner_authorization,
                          production_authorization=args.production_authorization,
                          check_cancellation=args.check_cancellation)
        print(json.dumps({'status': result['status'], 'report': str(args.output.resolve() / 'evaluation.json'),
                          'promoted': False}))
        if result['status'] != 'awaiting-private-perceptual-review':
            raise SystemExit(2)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
