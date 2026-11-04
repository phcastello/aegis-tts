"""Validation-only lossless chunking control with the retained native16 weights.

Reuses the qualified CUDA worker in one isolated process. Its local Backend
subclass changes only generate(text): lossless linguistic spans are synthesized
sequentially and concatenated without audio edits. Shared source files, weights,
training and the unsegmented quality executor remain unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import subprocess
import sys

from training import v2_fastpitch_gpu_quality as quality
from training import v2_fastpitch_benchmark as common
from training.f5_refine_segments import linguistic_chunks
from training import v2_qwen_cache_client as cache
from training import v2_qwen_cache_audit_reader as reader

SOURCE = Path(__file__).resolve()
OUTPUT = quality.PREPARATION / 'sentence-control-step16000-v2'
EXTERNAL_LEDGER = quality.ROOT / 'operations/qwen-reader-integration-20261104/external-ledger.PROPOSED.json'
CONTAINER = '08004c66f014e3e63fb6fd123946e4dbba2919ce29b4530848279917325faa75'
DOCKER_LOG_RESERVE = 64 * 1024**2
WEIGHT_SHA = '117c8d3548fa9c75d470468147afbbd1221a13faf622b8a11249c6238c6aa3eb'
QUALITY_SHA = '4b7c3ea312bd8677013953c0164ca4e669fe617f68bc1e0e5aff2351ec5afb7a'
MAX_CHARS, MAX_WORDS = 180, 32
CANDIDATE = 'fastpitch-step16000-cuda-bf16-linguistic-chunks'
SCHEMA = 'aegis.fastpitch.sentence-control.v1'


def chunks(text):
    parts = linguistic_chunks(text, max_chars=MAX_CHARS, max_words=MAX_WORDS)
    if not parts or ''.join(parts) != text or any(not part.strip() for part in parts):
        raise ValueError('Every normalized source character must occur once')
    return parts


def bindings():
    return {str(path): quality.sha(path) for path in
            (SOURCE, Path(quality.__file__), quality.REPO / 'training/f5_refine_segments.py',
             Path(cache.__file__), Path(reader.__file__), EXTERNAL_LEDGER, cache.PUBLICATION_PATH, cache.CHARGE_PATH)}


def original_stopped():
    value = json.loads(subprocess.check_output(['docker', 'inspect', '--size', CONTAINER], text=True, timeout=10))[0]
    if value['Id'] != CONTAINER or value['Image'] != reader.IMAGE or value['State']['Running'] is not False:
        raise RuntimeError('Exact original Qwen must remain stopped')
    return value


def external_budget(budget, cache_bytes, other_bytes, docker_rw):
    if any(type(n) is not int or n < 0 for n in (cache_bytes, other_bytes, docker_rw)):
        raise ValueError('Complete nonnegative external measurements required')
    total = budget['working_allocated_bytes'] + quality.PAIR_EXTRA + cache_bytes + other_bytes + docker_rw + DOCKER_LOG_RESERVE
    if total > 40_000_000_000 or docker_rw > 64 * 1024**2:
        raise RuntimeError('Full cache/external/Docker plus quality reserve exceeds budget')
    return dict(total_projected_bytes=total, full_vllm_physical_bytes=cache_bytes,
                full_other_external_bytes=other_bytes, full_docker_rw_bytes=docker_rw,
                docker_logs_reserved_bytes=DOCKER_LOG_RESERVE, cache_exclusion_credit_bytes=0)


def fresh_budget():
    # Quiet CPU audit completes and removes its own temporary container before
    # CUDA inference; no hashing or Docker auditor runs during generated audio.
    def guard():
        quality.worker_guard()
        quality.gpu_ownership(quality.gpu_pids(), os.getpid(), loaded=False)
        original_stopped()
    guard()
    snapshot = cache.quiet_snapshot(OUTPUT / 'external-cache-audit', guard)
    git = Path(subprocess.check_output(['git', '-C', str(quality.REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip())
    roots = sorted(Path('/var/tmp').glob('aegis*')) + [Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),
            quality.REPO, Path('/home/phc/.cache'), git]
    outside = cache.extra_external(EXTERNAL_LEDGER, roots)
    base = quality.preflight()
    info = original_stopped()
    result = external_budget(base, snapshot['physical_bytes'], outside['accounted_bytes'], info['SizeRw'])
    return dict(core=base, complete_cache_schema=snapshot['schema'],
                complete_cache_entries=snapshot['entries_count'], other_external=outside, **result)


def prepare():
    if quality.sha(quality.__file__) != QUALITY_SHA:
        raise ValueError('Qualified quality worker changed')
    options = argparse.Namespace(expected_step=16000,
        checkpoint=quality.ROOT / 'fastpitch/language/common-language-selection/best.pt',
        checkpoint_sha256=WEIGHT_SHA, output=OUTPUT,
        source=quality.ROOT / 'fastpitch/source', bigvgan=quality.ROOT / 'bigvgan')
    base = quality.planned(options)
    roster = {}
    for case in base['cases']:
        normalized = common.frontend('refinement')(case['text'])['tts_text']
        roster[case['id']] = dict(normalized_text=normalized, spans=chunks(normalized))
    if OUTPUT.exists():
        raise FileExistsError('Preserve previous control; do not overwrite')
    OUTPUT.mkdir(mode=0o700)
    quality.write(OUTPUT / 'base-plan.json', base)
    plan = dict(schema=SCHEMA, status='prepared_only', utc=quality.utc(),
        base_plan=str(OUTPUT / 'base-plan.json'), base_plan_sha256=quality.sha(OUTPUT / 'base-plan.json'),
        source_hashes=bindings(), candidate=CANDIDATE, roster=roster,
        policy=dict(order='Frozen refinement of whole input, then lossless linguistic_chunks, then original FastPitch tokenizer per span',
                    max_chars=MAX_CHARS, max_words=MAX_WORDS, waveform='Concatenate complete raw span waveforms; no silence, fades, gain, trim or truncation',
                    seed=quality.SEED, pace=1.0, device='CUDA', acoustic='BF16', parameters_vocoder='FP32'),
        hypothesis='Long input accounts for most native16 excess ASR edits; lossless shorter spans may improve clarity with identical weights.',
        advance='All7 valid; compare raw/spoken/CER and every case. Language surrogate threshold raw<=23/135 and spoken<=42/154; critical divergences need separate review. No auditory identity or pronunciation approval.',
        scope='Validation7 only. No final corpus, private target training, HTTP, streaming or TTFA claim.',
        implementation='Only process-local Backend subclass and row adapter compose the existing qualified quality worker; no shared source/weight mutation.')
    quality.write(OUTPUT / 'plan.json', plan)
    return OUTPUT / 'plan.json', quality.sha(OUTPUT / 'plan.json')


def load(path, expected):
    if quality.sha(path) != expected:
        raise ValueError('Control plan SHA differs')
    plan = quality.read(path)
    if (plan['schema'] != SCHEMA or plan['candidate'] != CANDIDATE
            or Path(path).parent != OUTPUT or plan['source_hashes'] != bindings()
            or quality.sha(quality.__file__) != QUALITY_SHA):
        raise ValueError('Frozen source/control differs')
    args = argparse.Namespace(plan_file=Path(plan['base_plan']),
                              plan_sha256=plan['base_plan_sha256'], gpu_authorized=True)
    base, output = quality.load_plan(args)
    if base['expected_step'] != 16000 or base['checkpoint']['sha256'] != WEIGHT_SHA or output != OUTPUT:
        raise ValueError('Only retained native16 is this control')
    if set(plan['roster']) != set(quality.CASE_IDS):
        raise ValueError('All7 cases required')
    for case in base['cases']:
        text = common.frontend('refinement')(case['text'])['tts_text']
        if plan['roster'][case['id']] != dict(normalized_text=text, spans=chunks(text)):
            raise ValueError('Lossless frozen span policy differs')
    if plan['policy'] != dict(order='Frozen refinement of whole input, then lossless linguistic_chunks, then original FastPitch tokenizer per span',
                             max_chars=180, max_words=32, waveform='Concatenate complete raw span waveforms; no silence, fades, gain, trim or truncation',
                             seed=quality.SEED, pace=1.0, device='CUDA', acoustic='BF16', parameters_vocoder='FP32'):
        raise ValueError('Frozen inference policy differs')
    return plan, args


def segmented_backend(base_class):
    class Segmented(base_class):
        def generate(self, text):
            parts = chunks(text)
            arrays, descriptions, count = [], [], 0
            for span in parts:
                self.torch.manual_seed(quality.SEED)
                values = super().generate(span)
                count += int(values.size)
                if count > quality.MAX_AUDIO_SECONDS * 24000:
                    raise ValueError('Combined audio exceeds bound; no truncation')
                arrays.append(values)
                descriptions.append(dict(text=span, samples=int(values.size)))
            self.control_segments = descriptions
            return self.np.concatenate(arrays)
    return Segmented


def verify_segments(rows, plan):
    if len(rows) != 7 or {row['case'] for row in rows} != set(quality.CASE_IDS):
        raise ValueError('Complete seven-case segmentation proof required')
    for row in rows:
        spans = row.get('generated_segments', [])
        if (row.get('candidate') != CANDIDATE or row.get('outcome') != 'ok'
                or [span['text'] for span in spans] != plan['roster'][row['case']]['spans']
                or any(type(span['samples']) is not int or span['samples'] <= 0 or span['samples'] % 256 for span in spans)
                or sum(span['samples'] for span in spans) != row['samples']):
            raise ValueError('No missing, extra, partial or relabeled generated spans')


def finalize_control_rows(rows, cases, error, original):
    completed = original(rows, cases, error)
    for row in completed:
        row['candidate'] = CANDIDATE
        row.setdefault('generated_segments', [])
    return completed


def execute(path, expected, worker=False):
    plan, args = load(path, expected)
    quality.worker_guard()
    quality.gpu_ownership(quality.gpu_pids(), os.getpid(), loaded=False)
    if not worker:
        if (OUTPUT / 'controller.json').exists() or (OUTPUT / 'result.json').exists():
            raise FileExistsError('Preserve existing execution')
        budget = fresh_budget()
        load(path, expected)
        quality.write(OUTPUT / 'controller.json', dict(status='exec_worker', utc=quality.utc(),
            identity=quality.process_identity(os.getpid()), owner=quality.worker_guard(), preflight=budget,
            plan_sha256=args.plan_sha256, extension_plan_sha256=expected,
            lifecycle='Same PID/start via exec; inherited qualified CUDA quality worker with local span adapter'))
        command = [sys.executable, '-m', 'training.v2_fastpitch_sentence_control', '--action', 'worker',
                   '--gpu-authorized', '--plan-file', str(path), '--plan-sha256', expected]
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                           CUDA_CACHE_PATH=str(quality.PREPARATION / 'cuda-cache'),
                           CUDA_CACHE_MAXSIZE=str(quality.CACHE_CAP))
        os.execve(sys.executable, command, environment)
    if quality.read(OUTPUT / 'controller.json').get('extension_plan_sha256') != expected:
        raise ValueError('Controller extension binding differs')
    # Both assignments are confined to this fresh inference process. Original
    # source bytes stay pinned and the ordinary CLI uses its original Backend.
    original_backend, original_measure, original_finalize = common.Backend, quality.measure, quality.finalize_rows
    common.Backend = segmented_backend(original_backend)
    def measure(backend, case, base_plan, output):
        backend.control_segments = []
        row = original_measure(backend, case, base_plan, output)
        row.update(candidate=CANDIDATE, sentence_control_plan_sha256=expected,
                   segmentation_policy=plan['policy'], generated_segments=backend.control_segments)
        return row
    quality.measure = measure
    quality.finalize_rows = lambda rows, cases, error=None: finalize_control_rows(rows, cases, error, original_finalize)
    quality.write(OUTPUT / 'extension-manifest.json', dict(plan=plan, plan_sha256=expected,
        base_plan_sha256=args.plan_sha256, source_hashes=bindings(), no_source_or_weights_mutated=True))
    try:
        code = quality.worker(args)
    finally:
        common.Backend, quality.measure, quality.finalize_rows = original_backend, original_measure, original_finalize
    load(path, expected)
    if code == 0:
        verify_segments(quality.read(OUTPUT / 'results.json'), plan)
    quality.write(OUTPUT / 'extension-complete.json', dict(utc=quality.utc(), returncode=code,
        plan_sha256=expected, source_hashes_stable=True, seven_complete_span_proofs=code == 0,
        scope='Parent must still verify process exit, GPU empty and all7 pinned ASR; no language/identity/latency approval'))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('plan', 'execute', 'worker'), default='plan')
    parser.add_argument('--plan-file', type=Path, default=OUTPUT / 'plan.json')
    parser.add_argument('--plan-sha256')
    parser.add_argument('--gpu-authorized', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == 'plan':
        import json
        path, digest = prepare()
        print(json.dumps(dict(status='prepared_only', plan=str(path), sha256=digest, torch_imported='torch' in sys.modules)))
        return 0
    if not args.gpu_authorized or not args.plan_sha256:
        raise ValueError('Explicit GPU authorization and sealed plan required')
    return execute(args.plan_file, args.plan_sha256, args.action == 'worker')


if __name__ == '__main__':
    raise SystemExit(main())
