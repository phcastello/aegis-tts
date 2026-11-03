"""Bounded, stdlib-only Piper validation supervision; never controls training/GPU.

Piper publishes a list of seven native probe outcomes and last-probes.json,
whose step AND checkpoint SHA bind a saved G+D state. This monitor freezes
that exact inode before CPU ASR. One pending state and one selected ASR state
are allowed; trainer last/mel-best/atomic temporary remain trainer-owned.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import wave

from training.f5_asr_metrics import METRIC_POLICY, plain_word_comparison, plain_word_tokens
from training.v2_language_monitor import physical_usage, ticks, utc
from training.v2_voxcpm_quality_monitor import process_identity, flag

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/var/tmp/aegis-v2-work')
PYTHON = '/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python'
TRAINER_SHA = '2e08e891f89a766eaaaff985c5482cc63d528f199aa453ff9520f985eeb64911'
SCORER_SHA = '4ee9f7da2cc8c733badb4d6d7c248a96ce0204a394fc26dc7652801a1410f241'
ASR_REVISION = '536b0662742c02347bc0e980a01041f333bce120'
TARGET_MANIFEST_SHA = 'b212a8eb20fe766af8d20fe741a20a07450022abd25a5bf20bf2ed84da7284a9'
TARGET_FINGERPRINT = '59568c022a7339165768322a456488eb2cd93bc2e3358f0e93dda99d07b6a99f'
VERSIONS = {'faster-whisper': '1.1.1', 'ctranslate2': '4.6.0', 'numpy': '1.26.4',
            'scipy': '1.17.1', 'soundfile': '0.14.0', 'av': '14.2.0'}
SETTINGS = dict(repository='Systran/faster-whisper-small', revision=ASR_REVISION,
                device='cpu', compute_type='int8', cpu_threads=2, num_workers=1,
                language='pt', beam_size=5, temperature=0, vad_filter=False,
                condition_on_previous_text=False, initial_prompt=None, trim=False, gain=False)
CP_LIMIT = 900 * 1024**2
BASE_BYTES = 845_888_150
WORKING_LIMIT = 40_000_000_000
FREE_RESERVE = 5 * 1024**3
AUX_RESERVE = 256 * 1024**2
WAV_LIMIT = 4 * 1024**2
KEEP_AUDIO = 6
MAX_JOBS = 512
HEX = re.compile(r'^[0-9a-f]{64}$')


class SafetyError(ValueError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sync_dir(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def owned_path(path, base, kind=None, missing=False):
    """Reject symlinks, foreign ownership and lexical/resolved path escapes."""
    path, base = Path(path).absolute(), Path(base).absolute()
    if '..' in path.parts or path.resolve(strict=False) != path or base.resolve(strict=False) != base:
        raise SafetyError('Symlink or noncanonical path: ' + str(path))
    try:
        relative = path.relative_to(base)
    except ValueError as exc:
        raise SafetyError('Path escapes private owner directory') from exc
    current = base
    for component in (None, *relative.parts):
        if component is not None:
            current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if missing:
                continue
            raise
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
            raise SafetyError('Expected owned real path: ' + str(current))
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise SafetyError('Nondirectory ancestor')
    if path.exists() and kind:
        info = path.lstat()
        if (kind == 'file' and not stat.S_ISREG(info.st_mode)) or (kind == 'directory' and not stat.S_ISDIR(info.st_mode)):
            raise SafetyError('Unexpected file type: ' + str(path))
    return path


def read(path, base, default=None):
    path = owned_path(path, base, 'file', missing=True)
    if not path.exists():
        return default
    if path.stat().st_size > 16 * 1024**2:
        raise SafetyError('Metadata exceeds bounded reader')
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value, base):
    path = owned_path(path, base, 'file', missing=True)
    temporary = owned_path(path.with_name(path.name + '.tmp'), base, 'file', missing=True)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path); sync_dir(path.parent)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_state(pid):
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(') ', 1)[1].split()[0]
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def trainer_binding(root, run, worker, previous=None):
    """Bind actual exec'ed Piper child to UID/PPID/start and the GPU worker."""
    if worker.get('status') != 'running':
        return None
    parent = process_identity(worker.get('worker_pid', -1))
    child = process_identity(worker.get('pid', -1))
    if not parent or not child or parent['uid'] != os.getuid() or child['uid'] != os.getuid() or child['ppid'] != parent['pid']:
        return None
    if flag(parent['command'], '-m') != 'training.v2_gpu_worker' or flag(parent['command'], '--root') != str(root):
        return None
    anchor = read(root / 'operations/gpu-window.json', root, {})
    if anchor.get('child_pid') != parent['pid'] or str(anchor.get('child_start')) != parent['start_ticks']:
        return None
    command = child['command']
    try:
        index = command.index('training.v2_piper_experiment') if flag(command, '-m') == 'training.v2_piper_experiment' else command.index(str(REPO / 'training/v2_piper_experiment.py'))
        action = command[index + 1]
    except (ValueError, IndexError):
        return None
    if action not in ('pilot', 'train') or flag(command, '--run') != str(run) or flag(command, '--device') != 'cuda' or '--gpu-authorized' not in command:
        return None
    identity = dict(**child, worker_pid=parent['pid'], worker_start_ticks=parent['start_ticks'],
                    boot_id=boot_id(), job_started_utc=worker.get('started_utc'),
                    job_command=worker.get('job', {}).get('command', []))
    if previous and previous.get('boot_id') == identity['boot_id'] and previous.get('job_started_utc') == identity['job_started_utc']:
        if any(previous.get(key) != identity[key] for key in ('pid', 'start_ticks', 'worker_pid', 'worker_start_ticks')):
            return None
    return identity


def owned_scorer(active):
    """Only our exact boot/UID/start/session and frozen input/output command."""
    process = process_identity(active.get('pid', -1))
    if not process or process['uid'] != os.getuid() or active.get('uid') != os.getuid():
        return False
    try:
        return (active.get('boot_id') == boot_id() and process['start_ticks'] == str(active.get('start_ticks'))
                and os.getpgid(process['pid']) == process['pid'] == os.getsid(process['pid'])
                and str(REPO / 'benchmarks/v2/asr_score.py') in process['command']
                and flag(process['command'], '--jobs') == active['jobs']
                and flag(process['command'], '--output') == active['output']
                and flag(process['command'], '--asr-root') == active['asr_root'])
    except (ProcessLookupError, PermissionError, KeyError):
        return False


def pause_child(active, paused, force=False):
    if not owned_scorer(active):
        raise SafetyError('Refusing signal to an unverified/reused ASR PID')
    current = process_state(active['pid'])
    actually_paused = current in ('T', 't') if current else bool(active.get('paused'))
    if not force and actually_paused == paused:
        active['paused'] = paused
        return False
    os.killpg(active['pid'], signal.SIGSTOP if paused else signal.SIGCONT)
    active['paused'] = paused
    return True


def checkpoint(run, step, digest, target):
    """Adopt only the pointer's exact step+SHA; never substitute newer weights."""
    if type(step) is not int or step < 0 or not HEX.fullmatch(digest):
        raise SafetyError('Invalid checkpoint pointer identity')
    target = owned_path(target, target.parent, 'file', missing=True)
    if target.exists():
        info = target.stat()
        if not 0 < info.st_size <= CP_LIMIT or sha(target) != digest:
            raise SafetyError('Existing frozen state differs from expected SHA/cap')
        return dict(step=step, sha256=digest, bytes=info.st_size, source='reconciled monitor inode',
                    binding='Published Piper pointer step+SHA and frozen exact file SHA; trainer validated full state')
    for label in ('last', 'best'):
        source = run / (label + '.pt'); sidecar = source.with_suffix('.pt.json')
        before = read(sidecar, run, {})
        if before.get('step') != step or before.get('sha256') != digest:
            continue
        try:
            owned_path(source, run, 'file'); info = source.lstat()
        except FileNotFoundError:
            continue
        if not 0 < info.st_size <= CP_LIMIT or before.get('bytes') != info.st_size:
            raise SafetyError('Unsafe/oversize Piper checkpoint or sidecar length mismatch')
        os.link(source, target, follow_symlinks=False)
        linked = target.stat()
        if ((linked.st_dev, linked.st_ino) == (info.st_dev, info.st_ino)
                and sha(target) == digest and read(sidecar, run) == before):
            sync_dir(target.parent)
            return dict(**before, source=str(source), binding='Exact published step+SHA, stable sidecar and frozen inode; no model load')
        target.unlink(); sync_dir(target.parent)
    return None


def checkpoint_inventory(run, jobs):
    """Account existing unique checkpoint inodes, including atomic temporaries."""
    paths = [run / name for name in ('last.pt', 'best.pt', 'last.pt.tmp', 'best.pt.tmp')]
    paths += [Path(job['directory']) / 'checkpoint.pt' for job in jobs if not job.get('removed_utc')]
    seen = set(); allocated = 0; largest = BASE_BYTES
    for path in paths:
        owned_path(path, run, 'file', missing=True)
        if not path.exists():
            continue
        info = path.stat(); key = (info.st_dev, info.st_ino)
        if not 0 < info.st_size <= CP_LIMIT:
            raise SafetyError('Checkpoint inventory exceeds per-file cap')
        largest = max(largest, info.st_size)
        if key not in seen:
            seen.add(key); allocated += info.st_blocks * 512
    return dict(allocated_bytes=allocated, unique_inodes=len(seen), largest_bytes=largest)


def budget_decision(usage, free_bytes, checkpoints):
    # Worst five distinct full states: trainer last, melbest, monitor bestASR,
    # one pending/active ASR candidate, trainer atomic temporary. Hardlinks that
    # coincide today are not presumed to coincide after the next publication.
    per_state = min(CP_LIMIT, checkpoints['largest_bytes'] + 16 * 1024**2)
    peak_states = 5 * per_state
    reserve = max(per_state, peak_states - checkpoints['allocated_bytes']) + AUX_RESERVE
    ok = (not usage.get('permission_errors') and usage['allocated_bytes'] + reserve <= WORKING_LIMIT
          and free_bytes >= FREE_RESERVE + reserve)
    return dict(ok=bool(ok), working_bytes=usage['allocated_bytes'], free_bytes=free_bytes,
                projected_peak_bytes=usage['allocated_bytes'] + reserve, reserve_bytes=reserve,
                full_states=5, per_state_estimate_bytes=per_state, checkpoint_inventory=checkpoints,
                auxiliary_reserve_bytes=AUX_RESERVE, working_limit_bytes=WORKING_LIMIT,
                free_reserve_bytes=FREE_RESERVE,
                explanation='5 distinct full states in worst case: last/melbest/bestASR/pending/atomic temporary; unique existing inodes subtracted once')


def pending_keys(jobs, best):
    return {job['key'] for job in jobs if (job.get('checkpoint') or (Path(job['directory']) / 'checkpoint.pt').exists()) and not job.get('checkpoint_removed_utc')
            and job['key'] != best and job.get('status') in ('waiting_checkpoint', 'ready', 'launching', 'running')}


def retention_keys(jobs, best, active, training_best):
    latest = sorted((job for job in jobs if not job.get('removed_utc')), key=lambda job: (job['step'], job['key']))[-KEEP_AUDIO:]
    return ({job['key'] for job in latest}
            | {job['key'] for job in jobs if job['step'] in (0, training_best)}
            | {key for key in (best, active) if key})


def unused_directory(path, base):
    owned_path(path, base, 'directory')
    entries = [path]
    for entry in path.iterdir():
        owned_path(entry, base, 'file')
        entries.append(entry)
    result = subprocess.run(['fuser', *map(str, entries)], capture_output=True, timeout=10)
    # fuser1 means no user; unsupported flags/errors must never authorize delete.
    return result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip()


def remove_directory(path, base):
    if path.parent != base or not unused_directory(path, base):
        return False
    for entry in path.iterdir():
        owned_path(entry, base, 'file').unlink()
    owned_path(path, base, 'directory').rmdir(); sync_dir(base)
    return True


def freeze(pointer_path, run, target, corpus, spoken, policy):
    """Freeze original list/jobs/pointer and complete seven-case denominator."""
    pointer = read(pointer_path, run)
    pointer_bytes = owned_path(pointer_path, run, 'file').read_bytes()
    if json.loads(pointer_bytes) != pointer:
        raise SafetyError('Probe pointer changed while reading')
    step = pointer.get('step'); digest = pointer.get('checkpoint_sha256', '')
    if type(step) is not int or step < 0 or not HEX.fullmatch(digest):
        raise SafetyError('Invalid last/best-probes identity')
    directory = Path(pointer['directory'])
    expected_name = 'initial' if step == 0 else f'step-{step:08d}'
    if directory != run / 'probes' / expected_name:
        raise SafetyError('Probe pointer escapes its exact native step directory')
    source = owned_path(directory / 'results.json', run, 'file')
    declared_path = owned_path(directory / 'asr-jobs.json', run, 'file')
    raw = source.read_bytes(); original_jobs = declared_path.read_bytes()
    rows = json.loads(raw); declared = json.loads(original_jobs)['jobs']
    if not isinstance(rows, list) or len(rows) != 7 or len({row['case'] for row in rows}) != 7 or {row['case'] for row in rows} != set(corpus):
        raise SafetyError('Expected all seven unique Piper validation outcomes')
    if len({row['case'] for row in declared}) != len(declared):
        raise SafetyError('Duplicate producer jobs')
    available = {row['case']: row for row in rows if row.get('wav')}
    if {row['case'] for row in declared} != set(available):
        raise SafetyError('Producer jobs do not cover exactly the available WAVs')
    declared = {row['case']: row for row in declared}
    for row in rows:
        case = row['case']
        if (row.get('step') != step or row.get('source_text') != corpus[case]['text']
                or row.get('effective_text') != corpus[case]['text'] or row.get('frontend') != 'native'
                or row.get('sampling') != dict(noise_scale=0., noise_w=0., length_scale=1.)):
            raise SafetyError('Piper native step/text/frontend/sampling mismatch')
    key = hashlib.sha256(raw + original_jobs + pointer_bytes
                         + json.dumps(policy, sort_keys=True).encode()).hexdigest()
    destination = target / f'step-{step:08d}-{key[:12]}'
    if destination.exists():
        return read(destination / 'source.json', target)
    staging = target / ('.snapshot-' + key[:12])
    if staging.exists():
        if not remove_directory(staging, target):
            raise SafetyError('Stale staging directory is active/unsafe')
    staging.mkdir(mode=0o700)
    try:
        jobs = []; hashes = {}; expected = {}
        candidate = f'piper-cadu-target-step{step}-{digest[:12]}'
        for case, row in available.items():
            wave_path = owned_path(row['wav'], run, 'file')
            item = declared[case]
            if wave_path != directory / (case + '.wav') or item.get('wav') != str(wave_path):
                raise SafetyError('Unexpected producer WAV path')
            texts = dict(verbatim=corpus[case]['text'], spoken=spoken[case], effective=corpus[case]['text'])
            if any(item.get('expected_' + kind) != text for kind, text in texts.items()):
                raise SafetyError('Producer expected-text policy differs')
            if wave_path.stat().st_size > WAV_LIMIT:
                raise SafetyError('WAV exceeds bounded validation duration')
            with wave.open(str(wave_path), 'rb') as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 24000, 'NONE') or not 0 < stream.getnframes() <= 60 * 24000:
                    raise SafetyError('Expected complete mono24kPCM16 validation WAV')
                if len(stream.readframes(stream.getnframes())) != stream.getnframes() * 2:
                    raise SafetyError('Truncated validation WAV')
            path = staging / (case + '.wav'); os.link(wave_path, path, follow_symlinks=False)
            hashes[case] = sha(path)
            if hashes[case] != row.get('wav_sha256'):
                raise SafetyError('Frozen WAV hash differs from producer')
            expected[case] = texts
            jobs.append(dict(candidate=candidate, case=case, cohort='validation-seven-native-zero-noise',
                             wav=str(destination / path.name), input_audio_sha256=hashes[case],
                             generation_outcome=row['outcome'], **{'expected_' + kind: text for kind, text in texts.items()}))
        (staging / 'source-results.json').write_bytes(raw)
        (staging / 'source-jobs.json').write_bytes(original_jobs)
        (staging / 'pointer.json').write_bytes(pointer_bytes)
        write(staging / 'jobs.json', dict(jobs=jobs), target)
        job = dict(key=key, step=step, checkpoint_sha256=digest, directory=str(destination),
                   status='waiting_checkpoint', policy={name: value for name, value in policy.items() if name != 'trainer_recipe'},
                   trainer_recipe_sha256=policy.get('trainer_recipe'), candidate=candidate, created_utc=utc(),
                   source_results_sha256=hashlib.sha256(raw).hexdigest(),
                   source_jobs_sha256=hashlib.sha256(original_jobs).hexdigest(),
                   pointer_sha256=hashlib.sha256(pointer_bytes).hexdigest(),
                   jobs_sha256=sha(staging / 'jobs.json'), audio_hashes=hashes, expected=expected,
                   outcomes={row['case']: row.get('outcome') for row in rows}, case_denominator=7,
                   synthesis_successes=sum(row.get('outcome') == 'ok' for row in rows), product=False)
        write(staging / 'source.json', job, target)
        for path in staging.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        sync_dir(staging); staging.rename(destination); sync_dir(target)
        return job
    finally:
        if staging.exists():
            remove_directory(staging, target)


def score_metrics(job, receipt, code):
    attempts = receipt.get('attempts', [])
    expected = job['audio_hashes']; successes = [row for row in attempts if row.get('success')]
    reasons = []
    if code not in (0, None):
        reasons.append('scorer_exit_failure')
    if not receipt.get('completed_utc') or receipt.get('deadline_reached'):
        reasons.append('missing_complete_receipt')
    if receipt.get('script_sha256') != job['policy']['scorer'] or receipt.get('model_receipt_sha256') != job['policy']['asr']:
        reasons.append('scorer_or_model_receipt_mismatch')
    if receipt.get('versions') != job['policy']['versions'] or receipt.get('metric_policy') != METRIC_POLICY:
        reasons.append('versions_or_normalization_mismatch')
    settings = receipt.get('settings', {})
    if any(settings.get(key) != value for key, value in SETTINGS.items()) or settings.get('niceness', -1) < 19:
        reasons.append('settings_mismatch')
    if (job['synthesis_successes'] != 7 or len(expected) != 7 or len(attempts) != 7
            or len(successes) != 7 or len({row.get('case') for row in attempts}) != 7):
        reasons.append('seven_case_denominator_incomplete')
    observed = {(row.get('case'), row.get('audio_sha256')) for row in attempts}
    if observed != set(expected.items()):
        reasons.append('input_audio_mismatch')
    for row in successes:
        case = row.get('case'); texts = job['expected'].get(case, {})
        key = dict(candidate=job['candidate'], case=case, audio_sha256=expected.get(case), expected=texts)
        if row.get('candidate') != job['candidate'] or row.get('key') != key or row.get('input_audio_sha256') != expected.get(case):
            reasons.append('input_job_identity_mismatch'); continue
        for kind, text in texts.items():
            result = row.get('comparisons', {}).get(kind, {})
            if not isinstance(row.get('hypothesis'), str) or result.get('reference_tokens') != plain_word_tokens(text):
                reasons.append('comparison_reference_mismatch'); continue
            if result.get('word_edits') != plain_word_comparison(text, row['hypothesis'])['word_edits']:
                reasons.append('word_edit_mismatch')
            if type(result.get('character_edits')) is not int or result['character_edits'] < 0 or result.get('reference_characters') != len(' '.join(plain_word_tokens(text))):
                reasons.append('character_metric_mismatch')
    metrics = dict(step=job['step'], key=job['key'], case_denominator=7, filtered=False,
                   synthesis_successes=job['synthesis_successes'], successful_cases=len(successes),
                   missing_cases=max(0, 7 - len(attempts)), asr_failures=len(attempts) - len(successes),
                   comparable=not reasons, reasons=sorted(set(reasons)), comparisons={}, product=False)
    if not reasons:
        for kind in ('verbatim', 'spoken', 'effective'):
            values = [row['comparisons'][kind] for row in successes]
            edits = sum(value['word_edits'] for value in values)
            words = sum(len(value['reference_tokens']) for value in values)
            chars = sum(value['reference_characters'] for value in values)
            char_edits = sum(value['character_edits'] for value in values)
            metrics['comparisons'][kind] = dict(word_edits=edits, reference_words=words,
                                               wer=edits / max(1, words), character_edits=char_edits,
                                               reference_characters=chars, cer=char_edits / max(1, chars))
    return metrics


class Monitor:
    def __init__(self, args):
        self.a = args; self.root = owned_path(args.root, args.root, 'directory')
        self.run = owned_path(args.run, self.root, 'directory')
        self.out = owned_path(self.run / 'quality-monitor-piper', self.run, 'directory', missing=True)
        self.out.mkdir(exist_ok=True, mode=0o700)
        self.probes = owned_path(self.out / 'probes', self.out, 'directory', missing=True)
        self.probes.mkdir(exist_ok=True, mode=0o700)
        self.scorer = REPO / 'benchmarks/v2/asr_score.py'
        self.files = dict(scorer=self.scorer, trainer=REPO / 'training/v2_piper_experiment.py',
                          corpus=REPO / 'benchmarks/v2/validation.pt-BR.json',
                          spoken=REPO / 'benchmarks/v2/spoken-validation.pt-BR.json',
                          metrics=REPO / 'training/f5_asr_metrics.py',
                          asr=args.asr_root / 'download-receipt.json')
        if sha(self.files['trainer']) != TRAINER_SHA or sha(self.scorer) != SCORER_SHA:
            raise SafetyError('Immutable trainer/scorer pin differs')
        asr = json.loads(self.files['asr'].read_text(encoding='utf-8'))
        if asr['repository'] != SETTINGS['repository'] or asr['revision'] != ASR_REVISION:
            raise SafetyError('ASR model revision differs')
        self.policy = {name: sha(path) for name, path in self.files.items()}
        self.policy.update(monitor=sha(__file__), versions=VERSIONS, native_frontend='native161',
                           sampling=dict(noise_scale=0., noise_w=0., length_scale=1.), case_denominator=7)
        self.corpus = {row['id']: row for row in json.loads(self.files['corpus'].read_text(encoding='utf-8'))}
        self.spoken = json.loads(self.files['spoken'].read_text(encoding='utf-8'))['cases']
        if len(self.corpus) != 7 or set(self.corpus) != set(self.spoken):
            raise SafetyError('Only fixed seven-case validation policy is supported')
        self.state = read(self.out / 'state.json', self.out, dict(jobs={}, best_asr=None))
        self.child = read(self.out / 'active.json', self.out); self.proc = None; self.running = True
        self.last_poll = self.last_budget = self.last_heartbeat = 0.
        self.log = logging.getLogger('piper-quality-' + str(os.getpid())); self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self.log.addHandler(RotatingFileHandler(self.out / 'monitor.log', maxBytes=1024**2, backupCount=1))
        write(self.out / 'recipe.json', dict(utc=utc(), pid=os.getpid(), start_ticks=ticks(os.getpid()),
              boot_id=boot_id(), command=sys.argv, policy=self.policy, run=str(self.run),
              poll_seconds=args.poll_seconds, heartbeat_seconds=1800, checkpoint_cap_bytes=CP_LIMIT,
              pending_checkpoint_cap=1, worst_distinct_full_states=5, scope='CPU2 nice19 pinned/offline ASR; no training/GPU controls or listening verdicts'), self.out)
        for source in self.probes.glob('step-*/source.json'):
            owned_path(source, self.probes, 'file'); job = read(source, self.probes)
            if Path(job['directory']) != source.parent or job['key'][:12] != source.parent.name.rsplit('-', 1)[-1]:
                raise SafetyError('Invalid persisted snapshot path/key')
            self.state['jobs'].setdefault(job['key'], job)
        if self.child:
            if owned_scorer(self.child):
                self.child['paused'] = process_state(self.child['pid']) in ('T', 't')
                self.state['jobs'][self.child['key']]['status'] = 'running'
                if (self.root / 'pause-asr').exists():
                    pause_child(self.child, True, force=True)
                write(self.out / 'active.json', self.child, self.out)
            # A dead/reused PID is reconciled by poll_child. Hashing a full
            # checkpoint during an active latency pause would disturb timing.
        self.reconcile_launch()

    def save(self):
        write(self.out / 'state.json', self.state, self.out)

    def review(self, kind, **values):
        path = self.out / 'review-needed.json'; record = read(path, self.out, dict(notices=[]))
        record['notices'] = [row for row in record['notices'] if row['kind'] != kind][-63:]
        record['notices'].append(dict(kind=kind, utc=utc(), **values))
        record['interpretation'] = 'Review only; ASR does not approve pronunciation/identity, and no automatic trainer control occurs'
        write(path, record, self.out); self.log.info(json.dumps(record['notices'][-1], ensure_ascii=False))

    def policy_ok(self):
        return all(sha(path) == self.policy[name] for name, path in self.files.items()) and sha(__file__) == self.policy['monitor']

    def budget_roots(self):
        return sorted(Path('/var/tmp').glob('aegis*')) + [Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'), REPO, Path('/home/phc/.cache')]

    def budget_ok(self, fresh=False):
        if (self.root / 'pause-asr').exists():
            return False
        if fresh or time.monotonic() - self.last_budget >= 300 or 'budget' not in self.state:
            usage = physical_usage(self.budget_roots())
            inventory = checkpoint_inventory(self.run, self.state['jobs'].values())
            self.state['budget'] = budget_decision(usage, shutil.disk_usage(self.root).free, inventory)
            self.last_budget = time.monotonic()
            write(self.out / 'budget-latest.json', self.state['budget'], self.out)
        return self.state['budget']['ok']

    def verify_job(self, job, with_checkpoint=True):
        directory = owned_path(job['directory'], self.probes, 'directory')
        if job['policy'] != self.policy or not self.policy_ok():
            raise SafetyError('Frozen scoring policy changed')
        for name, expected in [('source-results.json', job['source_results_sha256']), ('source-jobs.json', job['source_jobs_sha256']), ('pointer.json', job['pointer_sha256']), ('jobs.json', job['jobs_sha256'])]:
            if sha(owned_path(directory / name, self.probes, 'file')) != expected:
                raise SafetyError('Frozen source/job bytes changed')
        for case, expected in job['audio_hashes'].items():
            if sha(owned_path(directory / (case + '.wav'), self.probes, 'file')) != expected:
                raise SafetyError('Frozen audio bytes changed')
        if with_checkpoint:
            target = owned_path(directory / 'checkpoint.pt', self.probes, 'file')
            receipt = read(directory / 'checkpoint.json', self.probes, {})
            if receipt.get('step') != job['step'] or receipt.get('sha256') != job['checkpoint_sha256'] or sha(target) != job['checkpoint_sha256']:
                raise SafetyError('Exact checkpoint proof unavailable or changed')

    def observe(self):
        progress = read(self.run / 'progress.json', self.run, {})
        worker = read(self.root / 'worker-current.json', self.root, {})
        identity = trainer_binding(self.root, self.run, worker, self.state.get('trainer_identity'))
        if identity:
            self.state['trainer_identity'] = identity
        age = time.time() - (self.run / 'progress.json').stat().st_mtime if progress else None
        if identity and age is not None and age > 900:
            self.review('trainer_progress_stale', step=progress.get('step'), age_seconds=age)
        if progress and not identity:
            self.review('trainer_not_verified_running', step=progress.get('step'), worker_status=worker.get('status'))
        if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in progress.values()):
            self.review('nonfinite_progress', step=progress.get('step'))
        binding = identity or self.state.get('trainer_identity')
        if binding:
            for done in sorted((self.root / 'done').glob('*.json'), reverse=True):
                item = read(done, self.root, {})
                if item.get('job', {}).get('command') == binding['job_command'] and item.get('started_utc') == binding['job_started_utc']:
                    self.review('trainer_exit', status=item.get('status'), returncode=item.get('returncode'), receipt=str(done)); break
            for queued in (self.root / 'queue').glob('*.json'):
                if read(queued, self.root, {}).get('command') != binding['job_command']:
                    continue
                log = self.root / 'logs' / (queued.stem + '.log')
                if log.exists():
                    owned_path(log, self.root, 'file')
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size - 32768)); tail = stream.read().decode(errors='replace')
                    if re.search(r'\b(nan|nonfinite|out of memory|traceback)\b', tail, re.I):
                        self.review('trainer_log_error', log=str(log))
        paused = (self.root / 'pause-asr').exists()
        if not paused and not self.budget_ok():
            self.review('budget_blocks_new_capture_or_ASR', budget=self.state['budget'])
        free = shutil.disk_usage(self.root).free
        if free < FREE_RESERVE:
            self.review('volume_reserve_low', free_bytes=free)
        status = dict(utc=utc(), progress=progress, trainer_running=bool(identity), trainer_identity=identity,
                      progress_age_seconds=age, active_asr=self.child, paused=paused,
                      budget=self.state.get('budget'), free_bytes=free)
        status['pause_acknowledged'] = bool(paused and (not self.child
            or ticks(self.child['pid']) != str(self.child['start_ticks'])
            or (owned_scorer(self.child) and process_state(self.child['pid']) in ('T', 't'))))
        write(self.out / 'status.json', status, self.out)
        if paused:
            write(self.out / 'pause-ack.json', dict(utc=utc(), acknowledged=status['pause_acknowledged'], active_asr=self.child), self.out)
        if time.monotonic() - self.last_heartbeat >= 1800:
            write(self.out / 'heartbeat.json', status, self.out); self.log.info(json.dumps(dict(event='heartbeat', **status)))
            self.last_heartbeat = time.monotonic()

    def discover(self):
        if not self.policy_ok():
            raise SafetyError('Pinned monitor/trainer/scoring policy changed')
        if not self.budget_ok():
            return
        recipe = read(self.run / 'recipe.json', self.run, {})
        if not recipe:
            return
        if (recipe.get('code_sha256') != TRAINER_SHA or recipe.get('frontend') != 'native161'
                or recipe.get('train_count') != 1025 or recipe.get('validation_count') != 55
                or recipe.get('data_manifest_sha256') != self.a.expected_manifest_sha256
                or recipe.get('data_fingerprint') != self.a.expected_fingerprint):
            raise SafetyError('Unexpected immutable Piper target recipe')
        if self.state.get('trainer_recipe_sha256') not in (None, sha(self.run / 'recipe.json')):
            raise SafetyError('Piper recipe changed within monitor run')
        self.state['trainer_recipe_sha256'] = sha(self.run / 'recipe.json')
        policy = dict(self.policy, trainer_recipe=self.state['trainer_recipe_sha256'])
        for name in ('best-probes.json', 'last-probes.json'):
            pointer = self.run / name
            if not pointer.exists():
                continue
            if len(self.state['jobs']) >= MAX_JOBS:
                self.review('bounded_history_full', max_jobs=MAX_JOBS); break
            job = freeze(pointer, self.run, self.probes, self.corpus, self.spoken, policy)
            # Recipe is an additional generation binding, not a scorer-policy
            # change; retain it separately from the shared fixed ASR policy.
            job['trainer_recipe_sha256'] = policy['trainer_recipe']; job['policy'] = self.policy
            self.state['jobs'].setdefault(job['key'], job)
        observed = {job['step'] for job in self.state['jobs'].values()}
        missed = []
        for directory in (self.run / 'probes').glob('step-*'):
            match = re.fullmatch(r'step-(\d{8})', directory.name)
            if match and int(match[1]) not in observed:
                owned_path(directory, self.run, 'directory')
                missed.append(int(match[1]))
        if missed:
            self.review('intermediate_probe_checkpoint_identity_unavailable', steps=sorted(missed),
                        reason='No received published pointer SHA for these retained intermediate probes; no invented checkpoint or ASR-only selection')
        self.capture(); self.save()

    def capture(self):
        jobs = list(self.state['jobs'].values())
        pending = pending_keys(jobs, self.state.get('best_asr'))
        if len(pending) > 1:
            raise SafetyError('More than one retained pending checkpoint; manual reconciliation required')
        for job in sorted(jobs, key=lambda item: (item['step'], item['key'])):
            if job.get('removed_utc') or job['status'] != 'waiting_checkpoint':
                continue
            if job['synthesis_successes'] != 7 or len(job['audio_hashes']) != 7:
                job['status'] = 'generation_failed'; self.review('seven_probe_generation_incomplete', step=job['step'], outcomes=job['outcomes']); continue
            if pending and job['key'] not in pending:
                continue
            if not self.budget_ok(fresh=True):
                self.review('capture_budget_unavailable', step=job['step'], budget=self.state['budget']); return
            target = Path(job['directory']) / 'checkpoint.pt'
            receipt = checkpoint(self.run, job['step'], job['checkpoint_sha256'], target)
            if receipt:
                write(target.parent / 'checkpoint.json', receipt, self.probes)
                job['checkpoint'] = receipt; job['status'] = 'ready'; pending.add(job['key']); self.save(); return
            last = read(self.run / 'last.pt.json', self.run, {})
            best = read(self.run / 'best.pt.json', self.run, {})
            if last.get('step', -1) > job['step'] and (best.get('step'), best.get('sha256')) != (job['step'], job['checkpoint_sha256']):
                job['status'] = 'checkpoint_unavailable'
                self.review('exact_checkpoint_unavailable', step=job['step'], expected_sha256=job['checkpoint_sha256'], last=last, best=best)
            else:
                self.review('exact_checkpoint_publication_pending', step=job['step'], expected_sha256=job['checkpoint_sha256'])

    def reconcile_launch(self):
        intent = read(self.out / 'launching.json', self.out)
        if not intent or self.child:
            return
        matches = []
        if intent.get('boot_id') == boot_id():
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                process = process_identity(int(path.name))
                if not process:
                    continue
                candidate = dict(intent, pid=process['pid'], start_ticks=process['start_ticks'])
                if owned_scorer(candidate):
                    matches.append(candidate)
        if len(matches) > 1:
            raise SafetyError('Multiple matching scorer sessions; refuse duplicate launch')
        if matches:
            self.child = matches[0]
            self.child['paused'] = process_state(self.child['pid']) in ('T', 't')
            self.state['jobs'][intent['key']]['status'] = 'running'
            if (self.root / 'pause-asr').exists():
                pause_child(self.child, True, force=True)
            write(self.out / 'active.json', self.child, self.out)
        else:
            if (self.root / 'pause-asr').exists():
                return
            self.complete(intent, None)
            self.review('interrupted_launch_reconciled_no_retry', key=intent['key'])
        (self.out / 'launching.json').unlink(missing_ok=True); self.save()

    def start(self):
        if not self.running or self.child or (self.root / 'pause-asr').exists() or (self.out / 'launching.json').exists():
            return
        jobs = [job for job in self.state['jobs'].values() if job['status'] == 'ready' and job.get('checkpoint')]
        if not jobs:
            return
        if len(jobs) > 1:
            raise SafetyError('Pending state cap exceeded before ASR')
        if not self.budget_ok(fresh=True):
            return
        job = jobs[0]; self.verify_job(job); directory = Path(job['directory'])
        if (directory / 'scores.json').exists():
            self.complete(dict(key=job['key']), None); return
        command = ['nice', '-n', '19', self.a.python, str(self.scorer), '--jobs', str(directory / 'jobs.json'),
                   '--output', str(directory / 'scores.json'), '--asr-root', str(self.a.asr_root), '--deadline-seconds', '31536000']
        active = dict(key=job['key'], jobs=str(directory / 'jobs.json'), output=str(directory / 'scores.json'),
                      asr_root=str(self.a.asr_root), command=command, boot_id=boot_id(), uid=os.getuid(),
                      monitor_pid=os.getpid(), monitor_start_ticks=ticks(os.getpid()), paused=False,
                      active_seconds=0., last_monotonic=time.monotonic(), started_utc=utc())
        job['status'] = 'launching'; write(self.out / 'launching.json', active, self.out); self.save()
        if not self.running or (self.root / 'pause-asr').exists():
            job['status'] = 'ready'; (self.out / 'launching.json').unlink(); self.save(); return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        with (directory / 'asr.log').open('ab') as stream:
            self.proc = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        active.update(pid=self.proc.pid, start_ticks=ticks(self.proc.pid)); self.child = active
        job['status'] = 'running'; write(self.out / 'active.json', active, self.out)
        (self.out / 'launching.json').unlink(); self.save()
        if (self.root / 'pause-asr').exists():
            pause_child(active, True); write(self.out / 'active.json', active, self.out)

    def complete(self, active, code):
        job = self.state['jobs'][active['key']]
        if job['status'] in ('complete', 'failed'):
            return
        directory = Path(job['directory']); receipt = read(directory / 'scores.json', self.probes, {})
        metrics = score_metrics(job, receipt, code)
        try:
            self.verify_job(job)
        except (SafetyError, FileNotFoundError) as exc:
            metrics['comparable'] = False; metrics['reasons'].append('frozen_input_or_checkpoint_invalid')
            self.review('completion_integrity_failure', step=job['step'], error=str(exc))
        metrics['scores_sha256'] = sha(directory / 'scores.json') if (directory / 'scores.json').exists() else None
        job.update(status='complete' if metrics['comparable'] else 'failed', metrics=metrics, finished_utc=utc(), exit_code=code)
        previous = self.state['jobs'].get(self.state.get('best_asr'), {})
        old_wer = previous.get('metrics', {}).get('comparisons', {}).get('verbatim', {}).get('wer', math.inf)
        if metrics['comparable'] and job.get('checkpoint') and metrics['comparisons']['verbatim']['wer'] < old_wer:
            self.state['best_asr'] = job['key']
        write(directory / 'metrics.json', metrics, self.probes)
        self.log.info(json.dumps(dict(event='ASR_complete', **metrics)))
        self.review('ASR_ready_review_only' if metrics['comparable'] else 'ASR_incomplete', step=job['step'], metrics=metrics)
        self.save()

    def poll_child(self):
        if not self.child:
            return
        active = self.child; code = self.proc.poll() if self.proc and self.proc.pid == active['pid'] else None
        if code is not None or ticks(active['pid']) != str(active['start_ticks']):
            if (self.root / 'pause-asr').exists():
                return
            self.complete(active, code); self.child = self.proc = None
            (self.out / 'active.json').unlink(missing_ok=True); self.save(); return
        if not owned_scorer(active):
            self.review('ASR_identity_unverified_no_signal', pid=active['pid']); return
        now = time.monotonic()
        if not active.get('paused'):
            active['active_seconds'] += max(0., now - active['last_monotonic'])
        active['last_monotonic'] = now
        pause_child(active, (self.root / 'pause-asr').exists())
        if active['active_seconds'] > self.a.timeout:
            if owned_scorer(active):
                os.killpg(active['pid'], signal.SIGKILL)
            self.review('ASR_timeout_no_retry', key=active['key'])
        write(self.out / 'active.json', active, self.out)
        log = Path(self.state['jobs'][active['key']]['directory']) / 'asr.log'
        if log.exists() and owned_path(log, self.probes, 'file').stat().st_size > 2 * 1024**2:
            with log.open('r+b') as stream:
                stream.seek(-1024**2, 2); tail = stream.read(); stream.seek(0); stream.write(tail); stream.truncate()

    def cleanup(self):
        if (self.root / 'pause-asr').exists():
            return
        jobs = list(self.state['jobs'].values()); best = self.state.get('best_asr')
        active = self.child['key'] if self.child else None
        pending = pending_keys(jobs, best)
        if len(pending) > 1:
            raise SafetyError('Pending cap exceeded; no automatic deletion')
        keep_cp = pending | {key for key in (best, active) if key}
        training_best = read(self.run / 'best.pt.json', self.run, {}).get('step')
        keep = retention_keys(jobs, best, active, training_best) | keep_cp
        for job in jobs:
            directory = Path(job['directory'])
            if job.get('removed_utc') or not directory.exists():
                continue
            owned_path(directory, self.probes, 'directory')
            if not unused_directory(directory, self.probes):
                continue
            if (directory / 'checkpoint.pt').exists() and job['key'] not in keep_cp:
                owned_path(directory / 'checkpoint.pt', self.probes, 'file').unlink()
                (directory / 'checkpoint.json').unlink(missing_ok=True)
                job['checkpoint_removed_utc'] = utc(); job.pop('checkpoint', None)
            if job['key'] in keep:
                continue
            if remove_directory(directory, self.probes):
                job['removed_utc'] = utc()
                if job['status'] == 'waiting_checkpoint':
                    job['status'] = 'expired_unavailable'
                self.log.info(json.dumps(dict(event='cleanup', key=job['key'], directory=str(directory))))
        self.save()

    def shutdown(self):
        if self.child and owned_scorer(self.child):
            pause_child(self.child, True, force=True); write(self.out / 'active.json', self.child, self.out)
        self.save()

    def loop(self):
        while self.running:
            try:
                self.poll_child()
                if not self.child:
                    self.reconcile_launch()
                if time.monotonic() - self.last_poll >= self.a.poll_seconds:
                    self.observe(); self.discover(); self.cleanup(); self.save(); self.last_poll = time.monotonic()
                if not self.a.once:
                    self.start()
            except Exception as exc:
                self.log.exception('monitor_iteration_failed'); self.review('monitor_error', error=repr(exc)[:512])
            if self.a.once:
                return
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--python', default=PYTHON)
    parser.add_argument('--asr-root', type=Path, default=Path('/var/tmp/aegis-v2-evaluation-20261102/asr'))
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--expected-manifest-sha256', default=TARGET_MANIFEST_SHA)
    parser.add_argument('--expected-fingerprint', default=TARGET_FINGERPRINT)
    parser.add_argument('--once', action='store_true', help='Observe/freeze/reconcile only; never launch ASR')
    parser.add_argument('--plan', action='store_true', help='Print proposed operation; create no files/processes')
    args = parser.parse_args()
    if min(args.poll_seconds, args.timeout) <= 0:
        parser.error('Positive cadence/deadline required')
    if not HEX.fullmatch(args.expected_manifest_sha256) or not HEX.fullmatch(args.expected_fingerprint):
        parser.error('Manifest/fingerprint SHA256 pins required')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.umask(0o077)
    if args.plan:
        print(json.dumps(dict(status='PROPOSED_NOT_EXECUTED', run=str(args.run), poll_seconds=args.poll_seconds,
              heartbeat_seconds=1800, checkpoint_cap_bytes=CP_LIMIT, max_pending_checkpoints=1,
              worst_distinct_full_states=5, asr='CPU2 nice19 offline fixed scorer/model',
              expected_manifest_sha256=args.expected_manifest_sha256, expected_fingerprint=args.expected_fingerprint,
              trainer_sha256=TRAINER_SHA, scorer_sha256=SCORER_SHA), indent=2)); return
    owned_path(args.run, args.root, 'directory')
    if any((parent / '.git').exists() for parent in (args.run, *args.run.parents)):
        raise SafetyError('Private monitor output cannot be under Git')
    out = owned_path(args.run / 'quality-monitor-piper', args.run, 'directory', missing=True)
    out.mkdir(exist_ok=True, mode=0o700)
    lock_path = owned_path(out / 'monitor.lock', out, 'file', missing=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor = Monitor(args)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(monitor, 'running', False))
        try:
            monitor.loop()
        finally:
            monitor.shutdown()


if __name__ == '__main__':
    main()
