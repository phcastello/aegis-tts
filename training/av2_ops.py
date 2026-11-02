#!/usr/bin/env python3
"""Private production snapshot and guarded, unlimited-duration exclusive GPU run.

Run snapshot during CPU preparation. Run all GPU operations in ONE child command;
production is restored in finally only after that command exits or fails. No
training deadline, service recreation, image change, volume deletion or pruning.
The owner has authorized stopping only the existing aegis-tts container.
"""
from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import wave


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def private_dir(path):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError('private directory must not be a symlink')
    path = path.resolve()
    if any((p / '.git').exists() for p in (path, *path.parents)):
        raise ValueError('operational artifacts must be outside Git')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077:
        raise ValueError('private directory must be owned by this user and mode 0700')
    return path


def write(path, data):
    raw = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode()
    tmp = path.with_name(path.name + '.new')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def run(args):
    return subprocess.check_output(args, text=True)


def inspect(container):
    return json.loads(run(['docker', 'inspect', container]))[0]


def signature(info):
    # Config and HostConfig include the original command, environment, image tag,
    # healthcheck, binds, device requests, ports, restart, shm and security opts.
    result = {k: info[k] for k in ('Id', 'Image', 'Name', 'Config', 'HostConfig')}
    result['Mounts'] = sorted(info['Mounts'], key=lambda mount: mount['Destination'])
    return result


def gpu():
    return run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']).strip()


def gpu_pids():
    return run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']).strip()


def reference_hashes(info):
    result = {}
    for mount in info['Mounts']:
        if mount['Destination'] != '/protected':
            continue
        for path in sorted(Path(mount['Source']).rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as inp:
                    for block in iter(lambda: inp.read(1024 * 1024), b''):
                        digest.update(block)
                result[str(path)] = digest.hexdigest()
    return result


def snapshot(directory, repo, container):
    if (directory / 'snapshot.json').exists():
        raise RuntimeError('snapshot exists; select a new private directory')
    info = inspect(container)
    if not info['State']['Running']:
        raise RuntimeError('production must be running at snapshot')
    write(directory / 'snapshot.json', info)
    write(directory / 'image-inspect.json', json.loads(run(['docker', 'image', 'inspect', info['Image']])))
    write(directory / 'reference-hashes.json', reference_hashes(info))
    for source in ('.env', 'docker-compose.yml'):
        path = repo / source
        if path.is_file():
            write(directory / (source.lstrip('.') + '.private'), path.read_bytes())
    write(directory / 'preparation.json', {'utc': utc(), 'gpu': gpu(), 'disk': run(['df', '-B1', str(directory)]), 'container_id': info['Id'], 'image_id': info['Image'], 'restart_count': info['RestartCount']})
    print(json.dumps({'snapshot': str(directory), 'id': info['Id'], 'image': info['Image']}), flush=True)


def assert_original(directory):
    expected = json.loads((directory / 'snapshot.json').read_text())
    current = inspect(expected['Id'])
    if signature(current) != signature(expected):
        raise RuntimeError('container identity, image or configuration changed; refusing operation')
    if reference_hashes(current) != json.loads((directory / 'reference-hashes.json').read_text()):
        raise RuntimeError('protected reference hashes changed; refusing operation')
    return current


def request(base, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {'Content-Type': 'application/json'}
    return urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=timeout)


def verify(directory, base, label):
    info = assert_original(directory)
    deadline = time.monotonic() + 900  # startup readiness only, never training
    while True:
        try:
            with request(base, '/health/ready', timeout=5) as response:
                assert response.status == 200
            break
        except Exception:
            if time.monotonic() > deadline:
                raise RuntimeError('original production did not become ready in 15 min')
            time.sleep(3)
    for path in ('/health', '/ready', '/v1/aegis/queue'):
        with request(base, path, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError('health check failed: ' + path)
            response.read()
    text = 'A gaveta azul contém três chaves e um bilhete de Joana.'
    started = time.monotonic()
    with request(base, '/v1/aegis/speech', {'text': text, 'stream': False, 'response_format': 'wav', 'interrupt_policy': 'enqueue'}) as response:
        wav = response.read()
    wav_latency = time.monotonic() - started
    with wave.open(io.BytesIO(wav)) as audio:
        if audio.getframerate() != 24000 or audio.getnchannels() != 1 or audio.getnframes() < 2400:
            raise RuntimeError('real WAV synthesis failed')
    write(directory / (label + '.wav'), wav)
    started = time.monotonic()
    arrivals, body = [], bytearray()
    ttfa = None
    with request(base, '/v1/aegis/speech', {'text': text, 'stream': True, 'response_format': 'pcm', 'interrupt_policy': 'enqueue'}) as response:
        if response.headers.get_content_type() != 'application/octet-stream':
            raise RuntimeError('PCM response has wrong content type')
        rate = int(response.headers.get('X-Aegis-Sample-Rate', '24000'))
        while block := response.read1(4096):
            elapsed = time.monotonic() - started
            arrivals.append({'seconds': elapsed, 'bytes': len(block)})
            body.extend(block)
            if ttfa is None:
                samples = array.array('h', bytes(body[:len(body)//2*2]))
                if sys.byteorder != 'little':
                    samples.byteswap()
                if any(abs(v) > 32 for v in samples):
                    ttfa = elapsed
    total = time.monotonic() - started
    if len(body) < 4800 or len(body) % 2 or ttfa is None:
        raise RuntimeError('real PCM streaming is empty, silent or unaligned')
    pcm_wav = io.BytesIO()
    with wave.open(pcm_wav, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(body)
    write(directory / (label + '-stream.wav'), pcm_wav.getvalue())
    current = assert_original(directory)
    result = {'utc': utc(), 'container_id': current['Id'], 'image_id': current['Image'], 'running': current['State']['Running'], 'health': current['State'].get('Health', {}).get('Status'), 'restart_count': current['RestartCount'], 'gpu': gpu(), 'wav_latency_s': wav_latency, 'stream': {'ttfa_threshold': 'first received block with any PCM sample abs > 32', 'ttfa_s': ttfa, 'total_s': total, 'audio_s': len(body) / 2 / rate, 'rtf': total / (len(body) / 2 / rate), 'sample_rate': rate, 'bytes': len(body), 'arrivals': arrivals}}
    write(directory / (label + '.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'stream'}), flush=True)
    return result


def restore(directory, base):
    current = assert_original(directory)
    if not current['State']['Running']:
        # No candidate process may overlap original model residency.
        deadline = time.monotonic() + 90
        while gpu_pids():
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU is occupied after child exit; inspect process ownership before restoring')
            time.sleep(2)
        subprocess.run(['docker', 'start', current['Id']], check=True)
    return verify(directory, base, 'restored')


def terminate_group(child):
    # Child command owns its session/process group; no unrelated process is killed.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=15)
    # Reap any surviving child descendants as well.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def watchdog(directory, base):
    """Emergency rollback only when the guard process disappears; no timer limit."""
    while True:
        record = json.loads((directory / 'gpu-window.json').read_text())
        if record.get('status') == 'production_restored':
            return
        if process_start(record['owner_pid']) == record['owner_start']:
            time.sleep(2)
            continue
        record['watchdog_owner_lost_utc'] = utc()
        write(directory / 'gpu-window.json', record)
        pid = record.get('child_pid')
        if pid and process_start(pid) == record.get('child_start'):
            try:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 30
                while process_start(pid) == record.get('child_start') and time.monotonic() < deadline:
                    time.sleep(1)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            write(directory.parent / 'gpu-window-authorized.json', {'owner_authorized': True, 'status': 'emergency_closed', 'utc': utc()})
            restore(directory, base)
            record['restored_utc'] = utc()
            record['status'] = 'production_restored'
        except BaseException as exc:
            record['status'] = 'restoration_failed_requires_attention'
            record['watchdog_restoration_error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write(directory / 'gpu-window.json', record)
        return


def guarded(directory, base, command):
    if not command:
        raise ValueError('run requires a command after --')
    current = assert_original(directory)
    if not current['State']['Running']:
        raise RuntimeError('expected original production to be running before guarded run')
    record = {'started_utc': utc(), 'container_id': current['Id'], 'command': command, 'training_deadline': None, 'owner_pid': os.getpid(), 'owner_start': process_start(os.getpid())}
    write(directory / 'gpu-window.json', record)
    with (directory / 'watchdog.log').open('ab') as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--directory', str(directory), '--base-url', base, 'watchdog'], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    child = None
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[sig] = signal.signal(sig, interrupted)
    try:
        record['stop_started_utc'] = utc()
        write(directory / 'gpu-window.json', record)
        subprocess.run(['docker', 'stop', '--time', '45', current['Id']], check=True)
        if gpu_pids():
            raise RuntimeError('GPU still occupied after stopping only TTS')
        record['stopped_utc'] = utc()
        write(directory / 'gpu-window.json', record)
        write(directory.parent / 'gpu-window-authorized.json', {'owner_authorized': True, 'training_deadline': None, 'exclusive_aegisvoice_v2_gpu': True, 'production_stopped': True, 'container_id': current['Id'], 'container_stopped_utc': record['stopped_utc'], 'restore_only_after_all_gpu_phases': True, 'status': 'active'})
        child = subprocess.Popen(command, start_new_session=True)
        record['child_pid'] = child.pid
        record['child_start'] = process_start(child.pid)
        write(directory / 'gpu-window.json', record)
        code = child.wait()  # Unlimited training and evaluation window.
        record['child_returncode'] = code
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Further stop signals cannot interrupt restoration halfway through.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                terminate_group(child)
            record['gpu_work_ended_utc'] = utc()
            write(directory / 'gpu-window.json', record)
            write(directory.parent / 'gpu-window-authorized.json', {'owner_authorized': True, 'training_deadline': None, 'status': 'gpu_phases_closed', 'gpu_work_ended_utc': record['gpu_work_ended_utc']})
            restore(directory, base)
            record['restored_utc'] = utc()
            record['status'] = 'production_restored'
        except BaseException as exc:
            record['restoration_error'] = f'{type(exc).__name__}: {exc}'
            record['status'] = 'restoration_failed_requires_attention'
            raise
        finally:
            write(directory / 'gpu-window.json', record)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'verify', 'restore', 'run', 'watchdog'])
    parser.add_argument('--directory', required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--container', default='aegis-tts', choices=['aegis-tts'])
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.umask(0o077)
    directory = private_dir(args.directory)
    if args.action == 'snapshot':
        snapshot(directory, args.repo.resolve(), args.container)
    elif args.action == 'verify':
        verify(directory, args.base_url, 'baseline')
    elif args.action == 'restore':
        restore(directory, args.base_url)
    elif args.action == 'watchdog':
        watchdog(directory, args.base_url)
    elif args.action == 'run':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        guarded(directory, args.base_url, command)


if __name__ == '__main__':
    main()
