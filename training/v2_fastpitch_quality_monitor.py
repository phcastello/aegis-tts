"""Bounded CPU-only native FastPitch validation probes; never a latency benchmark.

The monitor imports no tensor framework. It freezes one atomically published
mel packet, then runs the existing CPU decoder/ASR in an identified session.
Only native text-only mels are decoded. No trainer, GPU queue or final test is
read or controlled. ASR notices request review and never reject a model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from training.v2_language_monitor import physical_usage, ticks

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/var/tmp/aegis-v2-work')
PYTHON = '/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python'
MAX_PACKET_BYTES = 64 * 1024**2
PROBE_RESERVE_BYTES = 192 * 1024**2
WORKING_LIMIT = 40_000_000_000
FREE_RESERVE = 5 * 1024**3


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


class SnapshotChanged(RuntimeError):
    pass


def freeze_snapshot(run, target):
    """Hardlink one complete inode; reject metadata-before-packet publication.

    The trainer publishes validation-latest.json before validation-mels.pt.
    Stable metadata and packet mtime order avoid labeling the previous packet
    with a new step. The decoder additionally verifies the actual saved step
    before constructing BigVGAN.
    """
    run, target = Path(run), Path(target)
    metadata = run / 'validation-latest.json'
    source = run / 'validation-mels.pt'
    before = metadata.read_bytes()
    metadata_stat = metadata.stat()
    source_stat = source.lstat()
    if source.is_symlink() or source_stat.st_uid != os.getuid():
        raise ValueError('Validation packet must be an owned real file')
    if source_stat.st_size > MAX_PACKET_BYTES:
        raise ValueError('Validation packet exceeds the declared memory/disk bound')
    if source_stat.st_mtime_ns < metadata_stat.st_mtime_ns:
        raise SnapshotChanged('Metadata is newer than its native mel packet')
    os.link(source, target)
    try:
        linked = target.stat()
        if (linked.st_dev, linked.st_ino) != (source_stat.st_dev, source_stat.st_ino):
            raise SnapshotChanged('Validation packet replaced before hardlink')
        if metadata.read_bytes() != before:
            raise SnapshotChanged('Validation metadata replaced while snapshotting')
        data = json.loads(before)
        step = data['step']
        if not isinstance(step, int) or step < 0:
            raise ValueError('Invalid validation step')
        return dict(step=step, sha256=sha(target), bytes=linked.st_size,
                    source_inode=linked.st_ino,
                    source_metadata_sha256=hashlib.sha256(before).hexdigest())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def retention_keys(jobs, best, active, keep=6):
    newest = sorted(jobs, key=lambda j: (j['step'], j['key']))[-keep:]
    retained = {job['key'] for job in newest}
    retained.update(k for k in (best, active) if k)
    return retained


def owned_group(active):
    """Prove boot, UID, PID start, session and unique decoder input marker."""
    pid = active['pid']
    try:
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != active['boot_id']:
            return False
        if ticks(pid) != str(active['start_ticks']):
            return False
        if Path(f'/proc/{pid}').stat().st_uid != os.getuid():
            return False
        if os.getpgid(pid) != pid or os.getsid(pid) != pid:
            return False
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        return (str(active['snapshot']) in command
                and any(arg.endswith('v2_fastpitch_decode_probe.py') for arg in command))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def set_paused(active, pause):
    if bool(active.get('paused')) == pause:
        return False
    if not owned_group(active):
        raise RuntimeError('Refuse to signal an unverified CPU decoder group')
    os.killpg(active['pid'], signal.SIGSTOP if pause else signal.SIGCONT)
    active['paused'] = pause
    return True


class Monitor:
    def __init__(self, args):
        self.a = args
        self.root, self.run = args.root.resolve(), args.run.resolve()
        self.out = self.run / 'quality-monitor'
        self.probes = self.out / 'probes'
        self.probes.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state = read(self.out/'state.json', dict(jobs={}, best_asr=None))
        self.child = read(self.out/'active.json')
        self.proc = None
        self.running = True
        self.last_poll = self.last_budget = self.last_heartbeat = 0.
        self.decoder = REPO/'training/v2_fastpitch_decode_probe.py'
        self.scorer = REPO/'benchmarks/v2/asr_score.py'
        self.policy = dict(decoder_sha256=sha(self.decoder), scorer_sha256=sha(self.scorer),
                           validation_manifest_sha256=sha(args.inputs/'validation.jsonl'))
        self.policy_sha = hashlib.sha256(json.dumps(self.policy, sort_keys=True).encode()).hexdigest()
        self.log = logging.getLogger('fastpitch-quality')
        self.log.setLevel(logging.INFO)
        handler = RotatingFileHandler(self.out/'monitor.log', maxBytes=1024**2, backupCount=1)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        self.log.addHandler(handler)
        write(self.out/'recipe.json', dict(utc=utc(), pid=os.getpid(), start_ticks=ticks(os.getpid()),
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              command=sys.argv, code_sha256=sha(__file__), policy=self.policy,
              first_step=args.first_step, every_steps=args.every_steps,
              snapshot_poll_seconds=args.poll_seconds, child_pause_poll_seconds=1,
              timeout_active_seconds=args.timeout, heartbeat_seconds=1800,
              budget_poll_seconds=300, probe_reserve_bytes=PROBE_RESERVE_BYTES,
              maximum_packet_bytes=MAX_PACKET_BYTES, pause_file=str(self.root/'pause-asr'),
              inputs=str(args.inputs), vocoder=str(args.vocoder), python=args.python,
              keep_latest=args.keep, keep_best_asr=True,
              cpu=dict(threads=2, nice=19, cuda_visible_devices=''),
              scope='Native validation mel -> CPU BigVGAN -> pinned local ASR. No product latency, test-final access, auditory claim or automatic rejection.',
              review_file=str(self.out/'review-needed.json')))
        self.reconcile()

    def event(self, name, **values):
        self.log.info(json.dumps(dict(event=name, **values), ensure_ascii=False))

    def save(self):
        write(self.out/'state.json', self.state)

    def review(self, kind, **values):
        # Separate from recovery-watch's notices: no competing read/modify/write.
        path = self.out/'review-needed.json'
        receipt = read(path, dict(notices=[]))
        receipt['notices'] = [n for n in receipt['notices'] if n['kind'] != kind]
        receipt['notices'].append(dict(kind=kind, utc=utc(), **values))
        receipt['notices'] = receipt['notices'][-100:]
        receipt['interpretation'] = 'Review only; ASR does not approve or reject pronunciation, identity or a model.'
        write(path, receipt)
        self.event('review_needed', kind=kind)

    def reconcile(self):
        for path in self.probes.glob('step-*/source.json'):
            job = read(path)
            if job and Path(job['directory']).resolve() == path.parent.resolve():
                self.state['jobs'].setdefault(job['key'], job)
        if self.child and not owned_group(self.child):
            self.complete(self.child, None)
            self.child = None
            (self.out/'active.json').unlink(missing_ok=True)
        self.save()

    def budget_roots(self):
        return (sorted(Path('/var/tmp').glob('aegis*'))
                + [Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'), REPO, Path('/home/phc/.cache')])

    def budget_ok(self):
        budget = self.state.get('budget', {})
        return (bool(budget) and not budget.get('permission_errors')
                and budget['allocated_bytes'] + PROBE_RESERVE_BYTES <= WORKING_LIMIT
                and shutil.disk_usage(self.root).free >= FREE_RESERVE + PROBE_RESERVE_BYTES)

    def observe(self):
        pause = (self.root/'pause-asr').exists()
        now = time.monotonic()
        if not pause and now-self.last_budget >= 300:
            self.state['budget'] = dict(utc=utc(), **physical_usage(self.budget_roots()))
            self.last_budget = now
            write(self.out/'budget-latest.json', self.state['budget'])
            if not self.budget_ok():
                self.review('budget_blocks_new_probe', budget=self.state['budget'],
                            free_bytes=shutil.disk_usage(self.root).free,
                            estimated_extra_bytes=PROBE_RESERVE_BYTES)
        status = dict(utc=utc(), progress=read(self.run/'progress.json', {}),
                      active_cpu_probe=self.child, paused=pause,
                      next_probe_min_step=self.a.first_step, every_steps=self.a.every_steps,
                      working_bytes=self.state.get('budget', {}).get('allocated_bytes'),
                      free_bytes=shutil.disk_usage(self.root).free,
                      completed_probes=sum(j['status']=='complete' for j in self.state['jobs'].values()))
        write(self.out/'status.json', status)
        if now-self.last_heartbeat >= 1800:
            write(self.out/'heartbeat.json', status)
            self.last_heartbeat = now
            self.event('heartbeat', step=status['progress'].get('step'), paused=pause)

    def discover(self):
        if (self.root/'pause-asr').exists() or not self.budget_ok():
            return
        current = dict(decoder_sha256=sha(self.decoder), scorer_sha256=sha(self.scorer),
                       validation_manifest_sha256=sha(self.a.inputs/'validation.jsonl'))
        if current != self.policy:
            self.review('scoring_policy_changed', before=self.policy, after=current)
            return
        data = read(self.run/'validation-latest.json', {})
        step = data.get('step', -1)
        if step < self.a.first_step:
            return
        milestone = self.a.first_step + ((step-self.a.first_step)//self.a.every_steps)*self.a.every_steps
        if any(j['milestone']==milestone and j['policy_sha256']==self.policy_sha for j in self.state['jobs'].values()):
            return
        staging = self.probes/f'.snapshot-{os.getpid()}'
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(mode=0o700)
        try:
            packet = freeze_snapshot(self.run, staging/'input-mels.pt')
            if packet['step'] != step:
                raise SnapshotChanged('Validation step changed before snapshot')
            key = hashlib.sha256(json.dumps(dict(**packet, policy=self.policy_sha), sort_keys=True).encode()).hexdigest()
            directory = self.probes/f'step-{step:07d}-{key[:12]}'
            job = dict(key=key, milestone=milestone, step=step, status='pending',
                       directory=str(directory), snapshot=packet,
                       policy_sha256=self.policy_sha, policy=self.policy, created_utc=utc(),
                       checkpoint_observed=read(self.run/'checkpoint.json', {}))
            write(staging/'source.json', job)
            if directory.exists():
                shutil.rmtree(staging)
            else:
                staging.rename(directory)
            self.state['jobs'].setdefault(key, job)
            self.save()
            self.event('native_packet_frozen', step=step, milestone=milestone, sha256=packet['sha256'])
        except SnapshotChanged as error:
            self.event('snapshot_retry', error=str(error))
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def start(self):
        if not self.running or self.child or (self.root/'pause-asr').exists() or not self.budget_ok():
            return
        pending = sorted((j for j in self.state['jobs'].values() if j['status']=='pending'), key=lambda j:j['step'])
        if not pending:
            return
        job = pending[0]
        if (sha(self.decoder) != self.policy['decoder_sha256']
                or sha(self.scorer) != self.policy['scorer_sha256']
                or sha(self.a.inputs/'validation.jsonl') != self.policy['validation_manifest_sha256']):
            self.review('scoring_policy_changed_before_launch')
            return
        directory = Path(job['directory'])
        if directory.is_symlink() or directory.resolve().parent != self.probes.resolve():
            raise ValueError('Unexpected frozen probe directory')
        snapshot = directory/'input-mels.pt'
        if sha(snapshot) != job['snapshot']['sha256'] or job['policy_sha256'] != self.policy_sha:
            raise ValueError('Frozen quality packet or policy changed')
        command = ['nice', '-n', '19', self.a.python, str(self.decoder),
                   '--run', str(self.run), '--snapshot', str(snapshot), '--expected-step', str(job['step']),
                   '--inputs', str(self.a.inputs), '--vocoder', str(self.a.vocoder),
                   '--output', str(directory/'decoded'), '--asr', '--pause-file', str(self.root/'pause-asr')]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        # Recheck immediately before mutation/launch, including monitor SIGINT.
        if not self.running or (self.root/'pause-asr').exists():
            return
        with (directory/'probe.log').open('ab') as output:
            self.proc = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        self.child = dict(key=job['key'], pid=self.proc.pid, start_ticks=ticks(self.proc.pid),
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                          snapshot=str(snapshot), command=command, paused=False,
                          started_utc=utc(), last_monotonic=time.monotonic(), active_seconds=0.)
        write(self.out/'active.json', self.child)
        job['status'] = 'running'
        self.save()
        self.event('cpu_probe_started', pid=self.proc.pid, step=job['step'])

    def complete(self, active, exit_code):
        job = self.state['jobs'].get(active['key'])
        if not job:
            self.review('missing_active_job', active=active)
            return
        if job['status']=='complete':
            # A crash after state commit but before active.json removal must
            # not duplicate a completed observation in metrics-history.jsonl.
            return
        decoded = Path(job['directory'])/'decoded'
        done = read(decoded/'completed.json', {})
        results = read(decoded/'results.json', {})
        receipt = read(decoded/'asr.json', {})
        attempts = receipt.get('attempts', [])
        expected = {(r['id'], r['sha256']) for r in results.get('results', [])}
        observed = {(r.get('case'), r.get('audio_sha256')) for r in attempts if r.get('success')}
        valid = (exit_code in (0, None) and done.get('cuda_initialized') is False
                 and done.get('step')==job['step'] and done.get('count')==4
                 and done.get('decoder_code_sha256')==job['policy']['decoder_sha256']
                 and done.get('validation_input_sha256')==job['policy']['validation_manifest_sha256']
                 and results.get('step')==job['step']
                 and results.get('mel_snapshot_sha256')==job['snapshot']['sha256']
                 and len(expected)==4 and expected==observed and len(attempts)==4
                 and receipt.get('completed_utc') and receipt.get('script_sha256')==job['policy']['scorer_sha256'])
        if valid:
            valid = all(Path(row['path']).resolve().parent==decoded.resolve()
                        and sha(row['path'])==row['sha256'] for row in results['results'])
        job['last_exit_code'] = exit_code
        job['finished_utc'] = utc()
        if valid:
            edits = sum(a['comparisons']['verbatim']['word_edits'] for a in attempts)
            words = sum(len(a['comparisons']['verbatim']['reference_tokens']) for a in attempts)
            metrics = dict(step=job['step'], key=job['key'], policy_sha256=job['policy_sha256'],
                           word_edits=edits, reference_words=words, wer=edits/max(1, words),
                           source_sha256=job['snapshot']['sha256'], asr_receipt_sha256=sha(decoded/'asr.json'),
                           summary=read(decoded/'asr-summary.json'), utc=utc(),
                           scope='CPU native validation diagnostic, not product latency or auditory assessment')
            job['status'], job['metrics'] = 'complete', metrics
            previous = self.state['jobs'].get(self.state.get('best_asr'), {})
            if (previous.get('policy_sha256') != job['policy_sha256']
                    or metrics['wer'] < previous.get('metrics', {}).get('wer', float('inf'))):
                self.state['best_asr'] = job['key']
            with (self.out/'metrics-history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush(); os.fsync(stream.fileno())
            self.review('native_asr_ready', step=job['step'], wer=metrics['wer'],
                        scores=str(decoded/'asr.json'), interpretation='Review ASR differences against audio; no automatic rejection or identity judgment.')
        else:
            job['status'] = 'failed'
            self.review('cpu_probe_failed', step=job['step'], exit_code=exit_code,
                        successful_cases=len(observed), decoded_count=done.get('count'), directory=str(decoded))
        self.save()

    def poll_child(self):
        if not self.child:
            return
        active = self.child
        code = self.proc.poll() if self.proc and self.proc.pid==active['pid'] else None
        if code is not None or ticks(active['pid']) != str(active['start_ticks']):
            self.complete(active, code)
            self.child = self.proc = None
            (self.out/'active.json').unlink(missing_ok=True)
            return
        now = time.monotonic()
        if not active['paused']:
            active['active_seconds'] += max(0., now-active['last_monotonic'])
        active['last_monotonic'] = now
        pause = (self.root/'pause-asr').exists()
        changed = set_paused(active, pause)
        if changed:
            self.event('cpu_probe_paused' if pause else 'cpu_probe_resumed', pid=active['pid'])
        if active['active_seconds'] > self.a.timeout:
            if owned_group(active):
                os.killpg(active['pid'], signal.SIGKILL)
            self.review('cpu_probe_timeout', active_seconds=active['active_seconds'], active=active)
        if changed or int(now)%10==0:
            write(self.out/'active.json', active)
        log = Path(self.state['jobs'][active['key']]['directory'])/'probe.log'
        if log.stat().st_size > 1024**2:
            with log.open('r+b') as stream:
                stream.seek(-512*1024, 2); tail=stream.read()
                stream.seek(0); stream.write(tail); stream.truncate()

    def cleanup(self):
        if (self.root/'pause-asr').exists():
            return
        jobs = list(self.state['jobs'].values())
        keep = retention_keys(jobs, self.state.get('best_asr'), self.child['key'] if self.child else None, self.a.keep)
        for job in jobs:
            path = Path(job['directory'])
            if job['key'] in keep or not path.exists():
                continue
            if path.is_symlink() or path.resolve().parent != self.probes.resolve() or path.stat().st_uid != os.getuid():
                raise ValueError('Unsafe quality-monitor cleanup path')
            if job['status']=='running':
                continue
            files = [str(p) for p in path.rglob('*') if p.is_file()]
            active = subprocess.run(['fuser', *files], capture_output=True, timeout=10) if files else None
            if active and active.returncode != 1:
                self.review('retained_probe_in_use', directory=str(path))
                continue
            allocated = physical_usage([path])['allocated_bytes']
            shutil.rmtree(path)
            job['removed_utc'] = utc()
            if job['status']=='pending':
                job['status']='expired_unscored'
            with (self.out/'cleanup.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(utc=utc(), directory=str(path), allocated_bytes=allocated,
                         source_sha256=job['snapshot']['sha256'], reason='Outside latest retained probes and best ASR; no fuser use'))+'\n')
            self.event('old_probe_removed', step=job['step'], allocated_bytes=allocated)

    def loop(self):
        while self.running:
            try:
                self.poll_child()
                now = time.monotonic()
                if now-self.last_poll >= self.a.poll_seconds:
                    self.observe(); self.discover(); self.cleanup(); self.save()
                    self.last_poll = now
                if not self.a.once:
                    self.start()
            except Exception as error:
                self.log.exception('quality_monitor_iteration_failed')
                self.review('monitor_error', error=repr(error))
            if self.a.once:
                return
            time.sleep(1)

    def pause_for_shutdown(self):
        # Run after any in-flight launch/poll returns. A signal handler alone
        # could miss a child created just after it ran. Unconditionally STOP
        # the verified group rather than trusting an interrupted pause flag.
        if self.child and owned_group(self.child):
            os.killpg(self.child['pid'], signal.SIGSTOP)
            self.child['paused'] = True
            self.child['paused_reason'] = 'monitor_stopped; reattach before further CPU work'
            write(self.out/'active.json', self.child)
        self.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT/'fastpitch/language')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--inputs', type=Path, default=ROOT/'portuguese/fastpitch-input-v1')
    p.add_argument('--vocoder', type=Path, default=ROOT/'bigvgan')
    p.add_argument('--python', default=PYTHON)
    p.add_argument('--first-step', type=int, default=4000)
    p.add_argument('--every-steps', type=int, default=2000)
    p.add_argument('--poll-seconds', type=float, default=30.)
    p.add_argument('--timeout', type=float, default=1800.)
    p.add_argument('--keep', type=int, default=6)
    p.add_argument('--once', action='store_true', help='One observation/snapshot pass; never launch a new CPU probe')
    args = p.parse_args()
    if min(args.first_step, args.every_steps, args.poll_seconds, args.timeout, args.keep) <= 0:
        p.error('Step cadence, polls, timeout and retention must be positive')
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    directory = args.run/'quality-monitor'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory/'monitor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor = Monitor(args)
    def stop(*_):
        # Keep the detached CPU session's identity for a subsequent reattach;
        # never signal the trainer or start a new probe after monitor SIGINT.
        monitor.running = False
        if monitor.child:
            write(monitor.out/'active.json', monitor.child)
        monitor.save()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        monitor.loop()
    finally:
        monitor.pause_for_shutdown()


if __name__ == '__main__':
    main()
