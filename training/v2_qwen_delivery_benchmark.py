"""Original-container Qwen baseline; default is a stdlib-only validation plan.

Execution requires an explicit root-review receipt and the real GPU worker.
Never av2_ops.restore/verify, container recreation or the end of the GPU window.
No CLI seed/frontend/codec override; no ASR. Final cohort reads remain gated.
"""
from __future__ import annotations
import argparse
import datetime as dt
import http.client
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

REPO = Path('/home/phc/aegis-v2-lab')
ROOT = Path('/var/tmp/aegis-v2-work')
OPS = ROOT / 'operations'
CONTAINER = '08004c66f014e3e63fb6fd123946e4dbba2919ce29b4530848279917325faa75'
IMAGE = 'sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0'
PROFILE = 'AegisVoicev1.0-Q00-W04-P07-M02'
RUNTIME = 'stable-codec22x22-onset-conservative-v1-startup320ms'
BASE = 'http://127.0.0.1:8001'
ENDPOINT = BASE + '/v1/audio/speech'
PROTOCOL_SHA = '60761d5e648bf17da49512f1707fe8656c082850b500d04e361316ead4510cbc'
METRIC_SHA = '997eed71e5066509dc7569e31e663733ab0977468c53bdd3b350a194a12f743f'
MODEL_ROOT = Path('/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base')
MODEL_REVISION = '5d83992436eae1d760afd27aff78a71d676296fc'
MODEL_PINS = {'model.safetensors': '180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6',
              'speech_tokenizer/model.safetensors': '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'}
PREPARATION = OPS / 'qwen-baseline-fair-protocol-20261104'
CACHE_BASELINE = PREPARATION / 'cache-original-baseline.json'
PRODUCTION_FILES = ('default_reference_gateway.py', 'voice_pipeline.py', 'runtime_config.py',
                    'run-vllm-with-default-reference.sh', 'aegis_speech/service.py')
CACHE_GROWTH_RESERVE = 512 * 1024**2  # Planned allocation, not a cache quota.
DOCKER_RW_CAP = 64 * 1024**2
DOCKER_LOG_CAP = 64 * 1024**2
TRANSIENT_MARGIN = DOCKER_RW_CAP  # Included in EXTRA_RESERVE, never added to its preflight reservation twice.
EXTRA_RESERVE = 512 * 1024**2 + CACHE_GROWTH_RESERVE + DOCKER_RW_CAP + DOCKER_LOG_CAP
PHASE_TIMEOUT = 3600  # Operational ceiling, including startup; no performance claim.
STOP_GRACE = 10
STOP_COMMAND_TIMEOUT = 20  # Below av2_ops watchdog's30s worker grace.
INSPECT_TIMEOUT = 10
REFERENCE_PROOF_SECONDS = 30  # Checked between local chunks; not an OS-I/O hard deadline.
sys.path.insert(0, str(REPO))
from training import av2_ops as ops
from training import v2_delivery_protocol as protocol
from training import v2_voxcpm_gateway_probe as wire
from training import v2_qwen_cache_audit_reader as cache_reader
from training import v2_qwen_cache_client as cache_client


def review_guard(args):
    """Before final-cohort read/hash in execute, and before every mutation."""
    if not args.gpu_authorized or not getattr(args, 'review', None):
        raise RuntimeError('Explicit --gpu-authorized and --review required before execution')
    path = Path(args.review)
    if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise RuntimeError('Root-review receipt must be an own private regular file')
    value = protocol.read(path)
    if (value.get('status') != 'approved_for_qwen_baseline'
            or value.get('container_id') != CONTAINER or value.get('image_id') != IMAGE
            or value.get('executor_sha256') != protocol.sha(__file__)
            or value.get('plan_sha256') != protocol.sha(args.run / 'plan.json')
            or value.get('cache_budget_policy') != 'frozen-existing-cache-exclusion-new-or-changed-files-counted'
            or value.get('resume_supplemental_budget_required') is not True):
        raise RuntimeError('Root-review receipt does not approve this exact source/plan/budget contract')
    return value


def model_contract(info, hash_payloads=False):
    """Metadata-only in plan; exact payload SHA only before start/after stop."""
    mounts = [m for m in info['Mounts'] if m['Destination'] == '/models/Qwen3-TTS-12Hz-0.6B-Base']
    if len(mounts) != 1 or mounts[0].get('RW') is not False or Path(mounts[0]['Source']).resolve() != MODEL_ROOT:
        raise RuntimeError('Expected exact original local read-only model mount')
    env = {s.partition('=')[0]: s.partition('=')[2] for s in info['Config'].get('Env', [])}
    if env.get('HF_HUB_OFFLINE') != '1' or env.get('TRANSFORMERS_OFFLINE') != '1':
        raise RuntimeError('Original offline loader flags differ')
    rows = []
    metadata_root = MODEL_ROOT / '.cache/huggingface/download'
    names = {str(path.relative_to(metadata_root))[:-len('.metadata')]: path
             for path in metadata_root.rglob('*.metadata')}
    if not set(MODEL_PINS) <= names.keys():
        raise RuntimeError('Local model/codec download metadata missing')
    actual = {str(path.relative_to(MODEL_ROOT)) for path in MODEL_ROOT.rglob('*')
              if path.is_file() and '.cache' not in path.relative_to(MODEL_ROOT).parts}
    if actual != set(names):
        raise RuntimeError('Untracked/alternate local model loader artifact; refuse ambiguous precedence')
    for name, metadata in sorted(names.items()):
        path = MODEL_ROOT / name
        lines = metadata.read_text().splitlines()
        if (len(lines) < 2 or lines[0] != MODEL_REVISION or len(lines[1]) not in (40, 64)
                or any(char not in '0123456789abcdef' for char in lines[1])
                or (name in MODEL_PINS and lines[1] != MODEL_PINS[name])):
            raise RuntimeError('Original local model revision/etag changed')
        expected = lines[1]
        if path.is_symlink() or not path.resolve().is_relative_to(MODEL_ROOT.resolve()):
            raise RuntimeError('Local model payload escapes original read-only model directory')
        before = path.stat()
        if hash_payloads:
            if len(expected) == 64:
                observed = protocol.sha(path)
            else:
                digest = hashlib.sha1(('blob %d\0' % before.st_size).encode())
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        digest.update(block)
                observed = digest.hexdigest()
            if observed != expected:
                raise RuntimeError('Exact local model/codec/config payload hash differs')
        after = path.stat()
        signature = lambda s: [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]
        if signature(before) != signature(after):
            raise RuntimeError('Local model changed during verification')
        rows.append(dict(path=str(path), revision=lines[0], expected_digest=expected,
                         digest_kind='sha256' if len(expected) == 64 else 'git-blob-sha1',
                         metadata_sha256=protocol.sha(metadata), signature=signature(after), payload_hashed=hash_payloads))
    return dict(loader_path='/models/Qwen3-TTS-12Hz-0.6B-Base', source=str(MODEL_ROOT), read_only_mount=True,
                offline=True, artifacts=rows,
                historical_limit='Nov3 snapshot bound this mount/path; it did not archive model payload SHA. Local HF metadata predates that run.')


def flag(command, name):
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def owner_identity(check_cpu=True):
    me = protocol.identity(os.getpid())
    parent = protocol.identity(os.getppid())
    anchor = protocol.read(OPS / 'gpu-window.json')
    if (me['uid'] != os.getuid() or parent['uid'] != os.getuid()
            or me['ppid'] != parent['pid']
            or flag(parent['command'], '-m') != 'training.v2_gpu_worker'
            or flag(parent['command'], '--root') != str(ROOT)
            or anchor.get('container_id') != CONTAINER
            or anchor.get('child_pid') != parent['pid']
            or str(anchor.get('child_start')) != parent['start_ticks']):
        raise RuntimeError('Actual immediate GPU-worker identity/window differs')
    owner = protocol.identity(anchor['owner_pid'])
    if (owner['uid'] != os.getuid() or owner['start_ticks'] != anchor['owner_start']
            or flag(owner['command'], '-m') != 'training.av2_ops'
            or flag(owner['command'], '--directory') != str(OPS)):
        raise RuntimeError('Original av2_ops owner is absent or changed')
    if check_cpu:
        if (ROOT / 'finish-gpu').exists():
            raise RuntimeError('Do not enter a baseline while the GPU window is finishing')
        if not (ROOT / 'pause-asr').exists():
            raise RuntimeError('Fixed CPU load requires ASR pause')
        from training.v2_voxcpm_optimized_probe import require_idle_cpu
        require_idle_cpu()
    return dict(self=me, worker=parent, guard_owner=owner,
                boot_id=me['boot_id'], container_id=anchor.get('container_id'))


def same_owner(expected):
    try:
        current = owner_identity(check_cpu=False)
        for role in ('self', 'worker', 'guard_owner'):
            if any(current[role][key] != expected[role][key]
                   for key in ('pid', 'ppid', 'start_ticks', 'uid', 'boot_id')):
                return False
        return True
    except (OSError, ValueError, RuntimeError, KeyError):
        return False


def original(running=None, references=False):
    # Never enter ops.inspect/assert_original: their Docker CLI is unbounded.
    current = json.loads(subprocess.check_output(['docker', 'inspect', CONTAINER], text=True,
                                                timeout=INSPECT_TIMEOUT))[0]
    frozen = protocol.read(OPS / 'snapshot.json')
    if (current['Id'] != CONTAINER or current['Image'] != IMAGE
            or frozen['Id'] != CONTAINER or frozen['Image'] != IMAGE
            or ops.signature(current) != ops.signature(frozen)):
        raise RuntimeError('Original container/image/configuration/mounts changed')
    if running is not None and current['State']['Running'] is not running:
        raise RuntimeError('Unexpected original container running state')
    if references:
        begun = time.monotonic()
        expected = protocol.read(OPS / 'reference-hashes.json')
        protected = [Path(m['Source']) for m in current['Mounts'] if m['Destination'] == '/protected' and m.get('RW') is False]
        if len(protected) != 1 or len(expected) != 77:
            raise RuntimeError('Expected exact original77 protected artifact bindings')
        paths = {str(path): path for path in protected[0].rglob('*') if path.is_file()}
        if set(paths) != set(expected):
            raise RuntimeError('Protected artifact roster changed')
        deadline = begun + REFERENCE_PROOF_SECONDS
        for name, path in sorted(paths.items()):
            if not path.resolve().is_relative_to(protected[0].resolve()):
                raise RuntimeError('Protected artifact escaped original mount')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Protected artifact proof exceeded operational deadline between chunks')
                    block = stream.read(1024**2)
                    if not block:
                        break
                    digest.update(block)
            if digest.hexdigest() != expected[name]:
                raise RuntimeError('Protected reference hashes changed')
        current['_qwen_reference_proof'] = dict(artifact_count=77, elapsed_seconds=time.monotonic() - begun,
            chunk_deadline_seconds=REFERENCE_PROOF_SECONDS, limitation='Local OS reads may block; chunk checks are not an I/O hard quota')
    return current


def parse_cgroups(raw):
    result = {}
    for line in raw.splitlines():
        hierarchy, controllers, path = line.split(':', 2)
        result[(hierarchy, controllers)] = path.rstrip('/') or '/'
    return result


def member_cgroup(actual, anchor):
    # Require the full Docker ID in the anchor; '/' or a broad host subtree is
    # never proof. Permit child cgroups used by vLLM's distributed workers.
    relevant = [(key, path) for key, path in anchor.items() if CONTAINER in path]
    return bool(relevant) and all(actual.get(key) == path or actual.get(key, '').startswith(path + '/')
                                  for key, path in relevant)


def cgroups(pid):
    return parse_cgroups((Path('/proc') / str(int(pid)) / 'cgroup').read_text())


def gpu_binding(container_process, allow_empty=False, verify_container=True):
    if verify_container:
        current = original(running=True)
        if current['State']['Pid'] != container_process['pid']:
            raise RuntimeError('Container init PID changed')
    now = protocol.identity(container_process['pid'])
    if now['start_ticks'] != container_process['start_ticks']:
        raise RuntimeError('Container init PID was recycled')
    anchor = cgroups(container_process['pid'])
    if protocol.identity(container_process['pid'])['start_ticks'] != now['start_ticks']:
        raise RuntimeError('Container init PID changed during cgroup lookup')
    snapshot = protocol.metrics().gpu_snapshot()
    actors = {}
    for row in snapshot['processes']:
        pid = int(row[0])
        actor = protocol.identity(pid)
        groups = cgroups(pid)
        if protocol.identity(pid)['start_ticks'] != actor['start_ticks']:
            raise RuntimeError('GPU actor PID changed during cgroup lookup')
        if not member_cgroup(groups, anchor):
            raise RuntimeError('GPU PID outside exact original Docker cgroup: ' + str(pid))
        actors[str(pid)] = dict(identity=actor, cgroups=[list(k) + [v] for k, v in groups.items()])
    if not actors and not allow_empty:
        raise RuntimeError('No original-container GPU process at measurement')
    return dict(snapshot=snapshot, gpu_actors=actors,
                container_init=now, cgroup_scope='Exact Docker container; init PID need not own a CUDA context')


def initial_auth():
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    if (value.get('status') != 'active' or value.get('container_id') != CONTAINER
            or any(value.get(key) is not True for key in
                   ('owner_authorized', 'exclusive_aegisvoice_v2_gpu', 'production_stopped'))):
        raise RuntimeError('Expected existing production-off experimental authorization')
    return value


def phase_write(saved, owner, phase):
    current = protocol.read(ROOT / 'gpu-window-authorized.json')
    if phase == 'qwen_baseline_starting':
        if current != saved:
            raise RuntimeError('Experimental authorization changed before baseline transition')
    elif phase == 'qwen_baseline_active':
        claimed = current.get('qwen_baseline', {})
        if (current.get('status') != 'qwen_baseline_starting'
                or claimed.get('pid') != owner['self']['pid'] or claimed.get('start_ticks') != owner['self']['start_ticks']):
            raise RuntimeError('Starting baseline authorization changed; refuse overwrite')
    else:
        raise ValueError('Unsupported Qwen authorization transition')
    value = dict(saved, status=phase, production_stopped=False,
                 exclusive_aegisvoice_v2_gpu=False,
                 qwen_baseline=dict(pid=owner['self']['pid'], start_ticks=owner['self']['start_ticks'],
                                    worker_pid=owner['worker']['pid'], owner_pid=owner['guard_owner']['pid'],
                                    container_id=CONTAINER, image_id=IMAGE))
    protocol.save(ROOT / 'gpu-window-authorized.json', value)


def baseline_guard(owner, container_process):
    if not same_owner(owner):
        raise RuntimeError('Guard ownership lost; hand recovery to av2_ops watchdog')
    if (ROOT / 'finish-gpu').exists() or not (ROOT / 'pause-asr').exists():
        raise RuntimeError('Baseline requires the same unfinished GPU window and ASR pause')
    from training.v2_voxcpm_optimized_probe import require_idle_cpu
    require_idle_cpu()
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    claimed = value.get('qwen_baseline', {})
    if (value.get('status') != 'qwen_baseline_active' or value.get('production_stopped') is not False
            or value.get('exclusive_aegisvoice_v2_gpu') is not False
            or claimed.get('pid') != owner['self']['pid']
            or claimed.get('start_ticks') != owner['self']['start_ticks']):
        raise RuntimeError('Explicit Qwen-online baseline authorization differs')
    return gpu_binding(container_process)


def headers_contract(headers, speech=False):
    if headers.get('x-aegis-tts-runtime-variant') != RUNTIME or headers.get('x-aegis-tts-dsp-variant') != 'stable':
        raise ValueError('Protected WORLD/22x22/onset/startup320 runtime differs')
    if speech and (headers.get('x-aegis-voice-profile') != PROFILE
                   or headers.get('x-aegis-sample-rate') != '24000'
                   or headers.get('x-aegis-channels') != '1'):
        raise ValueError('Protected speech profile/PCM contract differs')


def request_count():
    status, _, body = wire.complete_request(8001, '/metrics')
    if status != 200:
        raise ValueError('Original request counters unavailable')
    rows = {line.split()[0]: float(line.split()[1]) for line in body.decode().splitlines()
            if line and not line.startswith('#')}
    value = rows.get('aegis_tts_requests_total')
    if value is None or not math.isfinite(value) or value < 0 or value != int(value):
        raise ValueError('Invalid actual request counter')
    return int(value)


def completion(row):
    headers_contract(row.get('headers', {}), speech=True)
    identifier = row['headers'].get('x-aegis-request-id')
    if not identifier:
        raise ValueError('No actual request ID')
    status, _, body = wire.complete_request(8001, '/v1/aegis/speech/' + identifier)
    result = wire.status_receipt(row, status, body)
    if result.get('status') != 'completed':
        raise ValueError('HTTP200 PCM did not complete')
    status, _, body = wire.complete_request(8001, '/v1/aegis/speech/' + identifier + '/events')
    if len(body) > 65536:
        raise ValueError('SSE evidence exceeds bound')
    names = [line.removeprefix('event:').strip() for line in body.decode().splitlines() if line.startswith('event:')]
    if status != 200 or 'accepted' not in names or 'completed' not in names or any(x in names for x in ('failed', 'cancelled')):
        raise ValueError('Actual SSE completion contract failed')
    return dict(status=result, sse=body.decode())


def call(args, case, repetition, warmup=False, native=False):
    target = BASE + '/v1/aegis/speech' if native else ENDPOINT
    payload = dict(text=case['text'], stream=True, response_format='pcm') if native else dict(input=case['text'], stream=True, response_format='pcm')
    pcm, row = protocol.metrics().receive(target, payload, args.timeout)
    row.update(candidate=args.candidate, case=case['id'], size=case.get('size', 'warmup'), repetition=repetition,
               condition='first-external-after-ready' if warmup and repetition == 0 else 'common-external-warmup' if warmup else 'warm',
               final_test=args.final_test, source_text=case['text'], seed=40733133,
               seed_scope='Unmodified gateway Q00/stage0 seed40733133; protected stage1 codec seed41 unchanged. No payload seed.',
               scope='Original Qwen HTTP PCM, common onset/FIFO rules, no hardware playback delay')
    if row.get('outcome') == 'ok' and row.get('eof_ms', 0) > args.timeout * 1000:
        row.update(outcome='failed_deadline', error='EOF exceeded overall deadline')
    if row.get('outcome') == 'ok':
        try:
            row['gateway_completion'] = completion(row)
        except Exception as error:
            row.update(outcome='failed_gateway_completion', error=repr(error))
    if pcm and len(pcm) % 2 == 0 and not warmup and (repetition == 0 or row['outcome'] != 'ok'):
        protocol.output_guard(args.run, len(pcm) + 4096)
        path = args.run / (case['id'] + '-r%02d.wav' % repetition)
        protocol.write_pcm(path, pcm)
        row.update(wav=str(path), wav_sha256=protocol.sha(path))
    # Qwen does not expose effective normalized text in status. Do not invent
    # it using Vox/F5 normalization; jobs retain verbatim and frozen spoken.
    return row


def post_contracts(args):
    result = {}
    for route, payload in [('/v1/audio/speech', dict(input='Esta fala inédita verifica o arquivo completo.', stream=False, response_format='wav')),
                           ('/v1/aegis/speech', dict(text='Outra fala inédita verifica o arquivo nativo.', stream=False, response_format='wav'))]:
        status, headers, body = wire.complete_request(8001, route, 'POST', payload, 180)
        if status != 200:
            raise ValueError('Actual WAV request failed')
        headers_contract(headers, speech=True)
        if headers.get('content-type', '').split(';')[0] != 'audio/wav':
            raise ValueError('Actual WAV content type differs')
        result[route] = wire.wav_contract(body)
        path = args.run / ('post-' + ('openai' if route == '/v1/audio/speech' else 'native') + '-complete.wav')
        protocol.output_guard(args.run, len(body) + 4096)
        path.write_bytes(body)
        result[route].update(wav=str(path), wav_sha256=protocol.sha(path))
        result[route]['completion'] = completion(dict(case='post_wav_contract', repetition=0, headers=headers))
    row = call(args, dict(id='post_native_pcm', text='Esta frase confirma o PCM nativo completo.'), 0, True, True)
    row['condition'] = 'post-matrix-native-contract'
    if row['outcome'] != 'ok':
        raise ValueError('Actual native PCM/SSE failed')
    result['native_pcm'] = row
    connection = None
    try:
        connection, response = wire.request(8001, '/v1/aegis/speech', 'POST',
            dict(text='Esta solicitação verifica cancelamento durante a fala. ' * 12, stream=True, response_format='pcm'), 180)
        if response.status != 200:
            raise ValueError('Actual cancellation request failed')
        headers_contract({key.lower(): value for key, value in response.getheaders()}, speech=True)
        if response.getheader('Content-Type', '').split(';')[0].strip().lower() not in ('audio/pcm', 'audio/raw', 'application/octet-stream'):
            raise ValueError('Cancellation response is not actual raw PCM')
        identifier = response.getheader('X-Aegis-Request-Id')
        if not identifier:
            raise ValueError('Actual cancellation request has no request ID')
        first = response.read1(4096)
        if len(first) % 2:
            first += response.read(1)
        if len(first) < 2 or len(first) % 2 or first[:4] == b'RIFF':
            raise ValueError('Cancel must follow actual valid PCM')
        status, _, body = wire.complete_request(8001, '/v1/aegis/speech/' + identifier, 'DELETE')
        if status != 200:
            raise ValueError('Actual DELETE cancellation failed')
        result['cancel'] = dict(request_id=identifier, first_pcm_bytes=len(first), delete=json.loads(body))
    finally:
        if connection is not None:
            connection.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status, _, body = wire.complete_request(8001, '/v1/aegis/speech/' + identifier)
        state = json.loads(body).get('status') if status == 200 else None
        if state == 'cancelled':
            result['cancel']['final_status'] = json.loads(body)
            break
        time.sleep(.2)
    else:
        raise ValueError('Cancellation did not finish')
    status, _, body = wire.complete_request(8001, '/v1/aegis/speech/' + identifier + '/events')
    events = [line.removeprefix('event:').strip() for line in body.decode().splitlines() if line.startswith('event:')]
    if status != 200 or len(body) > 65536 or 'accepted' not in events or 'cancelled' not in events or any(name in events for name in ('completed', 'failed')):
        raise ValueError('Actual SSE cancellation contract failed')
    result['cancel']['sse'] = body.decode()
    row = call(args, dict(id='post_cancel_next', text='Esta nova frase confirma a retomada depois do cancelamento.'), 0, True)
    row['condition'] = 'post-matrix-next-after-cancel'
    if row['outcome'] != 'ok':
        raise ValueError('New generation after cancellation failed')
    result['next_after_cancel'] = row
    for route in ('/health', '/ready', '/v1/aegis/queue'):
        status, _, body = wire.complete_request(8001, route)
        if status != 200:
            raise ValueError('Post-contract health/readiness/queue failed')
        if route.endswith('/queue'):
            queue = json.loads(body)
            if queue.get('active_request_id') is not None or queue.get('queued_requests') != 0 or queue.get('queued_segments') != 0:
                raise ValueError('Actual post-contract queue is not empty')
            result[route] = queue
        else:
            result[route] = status
    return result


class CgroupSampler:
    """Same500ms cadence, resolving actor ownership while every PID is live."""
    def __init__(self, path, container_process):
        self.path = Path(path)
        self.container_process = container_process
        self.stop = threading.Event()
        self.thread = None
        self.peak = None
        self.error = None
        self.samples = 0

    def capture(self, stream):
        row = gpu_binding(self.container_process, allow_empty=True, verify_container=False)
        row['utc'] = protocol.utc()
        for device in row['snapshot']['devices']:
            self.peak = max(self.peak or 0, float(device[3]))
        protocol.append_row(stream, row)
        self.samples += 1
        if self.path.stat().st_size > 8 * 1024**2:
            raise RuntimeError('Bounded GPU sampler log exceeded8MiB')

    def loop(self):
        try:
            with self.path.open('x') as stream:
                self.capture(stream)
                while not self.stop.wait(.5):
                    self.capture(stream)
        except Exception as error:
            self.error = repr(error)

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(15)
        if self.thread.is_alive():
            self.error = self.error or 'GPU sampler did not stop'


def production_code(plan, run, label):
    """Only five small source files, never model/reference payloads."""
    output = run / ('production-code-' + label)
    output.mkdir()
    observed = {}
    for name in PRODUCTION_FILES:
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['docker', 'cp', CONTAINER + ':/gateway/' + name, str(path)],
                       check=True, capture_output=True, timeout=30)
        observed[name] = protocol.sha(path)
    if observed != plan['settings']['production_code_sha256']:
        raise RuntimeError('Actual original-container gateway source changed')
    return observed


def docker_size():
    value = json.loads(subprocess.check_output(['docker', 'inspect', '--size', CONTAINER], text=True, timeout=30))[0]
    if value['Id'] != CONTAINER or value['Image'] != IMAGE:
        raise RuntimeError('Docker size query resolved another container/image')
    return int(value.get('SizeRw', 0))


def cache_roots(info):
    roots = [Path(mount['Source']).resolve() for mount in info['Mounts'] if mount['Destination'] == '/cache']
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError('Exact original production-cache volume required')
    return roots


def capture_cache(info, run=None, guard=None):
    if run is None or guard is None:
        raise RuntimeError('CPU readonly auditor requires explicit owned artifact path/guard; no host fallback')
    if cache_roots(info) != [Path(cache_reader.HOST_CACHE).resolve()]:
        raise RuntimeError('Original production cache source differs')
    return cache_client.quiet_snapshot(run, guard)


def cache_fee(baseline, current, metadata_only=False):
    # Same temporal/closed-link policy for hash and metadata snapshots.
    return cache_client.fee(baseline, current)


def cache_physical_bytes(snapshot):
    return snapshot['physical_bytes']


def reader_phase_guard(owner, quiet=False):
    if not same_owner(owner): raise RuntimeError('CPU cache auditor owner lost')
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    if value.get('container_id') != CONTAINER or value.get('owner_authorized') is not True:
        raise RuntimeError('CPU cache auditor authorization differs')
    if value.get('status') not in ('active', 'qwen_baseline_starting', 'qwen_baseline_active', 'budget_reconciliation_required'):
        raise RuntimeError('CPU cache auditor phase closed')
    if quiet:
        original(running=False); protocol.gpu_set([])
    elif value.get('status') != 'active':
        claim = value.get('qwen_baseline', {})
        if claim.get('pid') != owner['self']['pid'] or claim.get('start_ticks') != owner['self']['start_ticks']:
            raise RuntimeError('Metadata reader phase belongs to another job')
    return value


def phase_budget(run, info, extra, baseline, owner=None, snapshot=None, external_ledger=None, core_usage=None):
    """Union/dedup first, subtract only proven immutable pre-existing cache."""
    from training.v2_language_monitor import physical_usage
    caches = cache_roots(info)
    common = Path(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip())
    roots = sorted(Path('/var/tmp').glob('aegis*')) + [Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),
             REPO, Path('/home/phc/.cache'), common]
    if any(cache == root.resolve() or cache.is_relative_to(root.resolve()) or root.resolve().is_relative_to(cache)
           for cache in caches for root in roots):
        raise RuntimeError('Core inventory/cache scope overlap; no duplicate credit permitted')
    working = physical_usage(roots) if core_usage is None else core_usage
    outside = cache_client.extra_external(external_ledger, roots)
    current = snapshot if snapshot is not None else capture_cache(info, run / ('cache-quiet-' + str(time.time_ns())),
                    lambda: reader_phase_guard(owner, quiet=True))
    measured_fee = cache_fee(baseline, current)
    charge = cache_client.external_charge()
    cache_physical = measured_fee['physical_bytes']
    size_rw = docker_size()
    free = __import__('shutil').disk_usage(run).free
    rw_growth = size_rw  # All writable-layer bytes are new/uncertain; no late-baseline exemption.
    total = working['allocated_bytes'] + measured_fee['accounted_bytes'] + outside['accounted_bytes'] + rw_growth + DOCKER_LOG_CAP
    if (working['permission_errors'] or rw_growth > DOCKER_RW_CAP
            or total + extra > 40_000_000_000 or free < 5 * 1024**3 + extra):
        raise RuntimeError('Qwen40GB/5GiB budget: new/changed cache and uncertain hardlinks/Docker growth remain counted')
    return dict(working_union=working, cache_fee=measured_fee, external_charge=charge, extra_external=outside, production_cache_roots=[str(path) for path in caches],
                cache_snapshot=current, cache_physical_bytes=cache_physical,
                cache_accounted_bytes=measured_fee['accounted_bytes'],
                temporary_auditor_metadata_margin_within_extra_bytes=DOCKER_RW_CAP,
                docker_size_rw_bytes=size_rw, docker_size_rw_growth_bytes=rw_growth,
                docker_log_growth_reserved_bytes=DOCKER_LOG_CAP,
                total_accounted_bytes=total, extra_reserved_bytes=extra, free_bytes=free,
                resume_reconciliation='Fresh mainV3 reconciliation must add new/changed cache and Docker-growth supplement; immutable baseline remains excluded')


class SafetyMonitor:
    """Lightweight stat/free checks only; no inventory/hashing during timings."""
    def __init__(self, run, before, info, baseline, phase_deadline, auditor=None):
        self.run = run
        self.before = before
        self.info = info
        self.auditor = auditor
        self.baseline = baseline
        self.initial_free = before['free_bytes']
        self.phase_deadline = phase_deadline
        self.stop = threading.Event()
        self.thread = None
        self.error = None

    def measure(self):
        begun = time.monotonic()
        if self.auditor is None: raise RuntimeError('Approved persistent CPU cache reader required; no host fallback')
        snapshot = self.auditor.snapshot()
        fee = cache_fee(self.baseline, snapshot, metadata_only=True)
        free = __import__('shutil').disk_usage(self.run).free
        physical_loss = self.initial_free - free
        cache_physical_delta = fee['physical_bytes'] - self.before['cache_physical_bytes']
        outside_cache_growth = max(0, physical_loss - cache_physical_delta)
        cache_fee_growth = max(0, fee['accounted_bytes'] - self.before['cache_accounted_bytes'])
        growth = outside_cache_growth + cache_fee_growth
        return dict(utc=protocol.utc(), free_bytes=free, physical_free_loss_bytes=physical_loss,
                    cache_physical_delta_bytes=cache_physical_delta, cache_physical_bytes=fee['physical_bytes'],
                    cache_accounted_bytes=fee['accounted_bytes'], outside_cache_growth_bytes=outside_cache_growth,
                    cache_fee_growth_bytes=cache_fee_growth, combined_growth_bytes=growth,
                    estimated_total_bytes=self.before['total_accounted_bytes'] + growth,
                    transient_margin_bytes=TRANSIENT_MARGIN,
                    changed_or_uncertain_paths=len(fee['charged_paths']),
                    metadata_scan_seconds=time.monotonic() - begun, payloads_hashed=False)

    def loop(self):
        while not self.stop.wait(5):
            try:
                protocol.output_guard(self.run)
                if time.monotonic() >= self.phase_deadline:
                    raise RuntimeError('Qwen baseline exceeded the one-hour operational phase ceiling')
                row = self.measure()
                protocol.save(self.run / 'safety-metadata-latest.json', row)
                if (row['free_bytes'] < 5 * 1024**3 + TRANSIENT_MARGIN
                        or row['combined_growth_bytes'] + TRANSIENT_MARGIN > EXTRA_RESERVE
                        or row['estimated_total_bytes'] + TRANSIENT_MARGIN > 40_000_000_000):
                    raise RuntimeError('Phase metadata-cache fee/free-space/40GB reserve exceeded')
            except Exception as error:
                self.error = repr(error)
                protocol.save(self.run / 'safety-stop.json', dict(utc=protocol.utc(), error=self.error))
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(6)
        if self.thread.is_alive():
            self.error = self.error or 'Cache metadata watcher did not stop; local filesystem reads have no hard I/O deadline'


@contextmanager
def cleanup_signals():
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def interrupted(number, _):
        for value in previous:
            signal.signal(value, signal.SIG_IGN)
        raise KeyboardInterrupt('Qwen baseline interrupted by signal' + str(number))
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def complete_matrix(rows, plan, error=None):
    expected = [(case, repetition) for repetition in range(plan['repetitions']) for case in plan['cohort']['cases']]
    by_key = {(row['case'], row['repetition']): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate measured condition')
    return [by_key.get((case['id'], repetition), dict(candidate=plan['candidate'], case=case['id'],
            size=case['size'], repetition=repetition, condition='warm', final_test=plan['cohort']['final_test'],
            outcome='not_attempted', error=error or 'not attempted')) for case, repetition in expected]


def worker_alive(owner):
    try:
        actual = protocol.identity(owner['worker']['pid'])
        return all(actual[key] == owner['worker'][key] for key in ('pid', 'start_ticks', 'uid', 'boot_id'))
    except (OSError, ValueError, KeyError):
        return False


def reconciliation_pending(args, owner, saved_auth, claimed, supplement=None):
    if not same_owner(owner):
        raise RuntimeError('Owner lost; do not overwrite watchdog reconciliation state')
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    if (value.get('qwen_baseline') != claimed
            or value.get('status') not in ('qwen_baseline_starting', 'qwen_baseline_active', 'budget_reconciliation_required')):
        raise RuntimeError('Baseline authorization changed before reconciliation state')
    pending = dict(status='root_fresh_gate_required', executor_sha256=protocol.sha(__file__),
                   plan_sha256=protocol.sha(args.run / 'plan.json'), run=str(args.run),
                   original_experimental_authorization=saved_auth,
                   cache_baseline=str(args.cache_baseline), cache_baseline_sha256=protocol.sha(args.cache_baseline),
                   qwen_evidence_complete=supplement is not None,
                   supplement_path=str(supplement) if supplement else None,
                   supplement_sha256=protocol.sha(supplement) if supplement else None,
                   consumer='A new independently reviewed fresh gate must account cache fees/Docker extras and six checkpoint slots; oldV3 does not consume this receipt')
    protocol.save(ROOT / 'gpu-window-authorized.json', dict(saved_auth, status='budget_reconciliation_required',
        production_stopped=True, exclusive_aegisvoice_v2_gpu=False, qwen_baseline=claimed,
        gpu_empty=True, budget_reconciliation=pending))
    return pending


def cleanup_transaction(args, owner, saved_auth, transaction_started, plan):
    if not transaction_started:
        return dict(status='no_container_transition', experimental_authorization_restored=False)
    # On owner loss the av2_ops watchdog owns production restoration. Do not
    # race its restore/start/verify or reopen experimental authorization.
    if not same_owner(owner):
        return dict(status='owner_lost_handed_to_watchdog', experimental_authorization_restored=False,
                    reconciliation_complete=False)
    value = protocol.read(ROOT / 'gpu-window-authorized.json')
    claimed = value.get('qwen_baseline', {})
    if (claimed.get('pid') != owner['self']['pid'] or claimed.get('start_ticks') != owner['self']['start_ticks']
            or value.get('status') not in ('qwen_baseline_starting', 'qwen_baseline_active')):
        raise RuntimeError('Baseline authorization changed; refuse cleanup/overwrite')
    try:
        current = original()
        running = current['State']['Running']
    except subprocess.TimeoutExpired:
        if not same_owner(owner):
            return dict(status='owner_lost_after_inspect_timeout_handed_to_watchdog',
                        experimental_authorization_restored=False, reconciliation_complete=False)
        protocol.save(args.run / 'cleanup-inspect-timeout.json', dict(inspect_timeout_seconds=INSPECT_TIMEOUT,
            recovery='Bounded stop of previously proven exact original ID; no cgroup/other process action'))
        running = True  # State unknown; exact ID was proven before this transaction.
    if running:
        subprocess.run(['docker', 'stop', '--time', str(STOP_GRACE), CONTAINER], check=True, capture_output=True,
                       timeout=STOP_COMMAND_TIMEOUT)
    if not same_owner(owner):
        return dict(status='owner_lost_after_stop_handed_to_watchdog', experimental_authorization_restored=False,
                    reconciliation_complete=False)
    deadline = time.monotonic() + 90
    while True:
        if not same_owner(owner):
            return dict(status='owner_lost_while_waiting_for_gpu_handed_to_watchdog', experimental_authorization_restored=False,
                        reconciliation_complete=False)
        try:
            gpu = protocol.gpu_set([])
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU did not empty after stopping the exact original container')
            time.sleep(1)
    stopped = original(running=False)
    reconciliation_pending(args, owner, saved_auth, claimed)
    stopped = original(running=False, references=True)
    source_proof = production_code(plan, args.run, 'after')
    baseline = protocol.read(args.cache_baseline)
    budget = phase_budget(args.run, stopped, 0, baseline, owner=owner, external_ledger=args.external_ledger)
    model_after = model_contract(stopped, hash_payloads=True)
    model_before = protocol.read(args.run / 'model-before.json')
    if model_after != model_before:
        raise RuntimeError('Actual local model/revision/signature changed across the baseline')
    if protocol.source_bindings(plan['source_hashes']) != plan['source_hashes']:
        raise RuntimeError('Frozen source/evidence changed; keep baseline authorization closed')
    supplement = args.run / 'resume-budget-supplement.json'
    protocol.save(supplement, budget)
    if not same_owner(owner):
        return dict(status='owner_lost_after_stop_handed_to_watchdog', experimental_authorization_restored=False,
                    stopped=True, gpu_empty=True, post_budget=budget, reconciliation_complete=False)
    pending = reconciliation_pending(args, owner, saved_auth, claimed, supplement)
    return dict(status='original_stopped_gpu_empty_budget_reconciliation_required', stopped=True, gpu_empty=True,
                experimental_authorization_restored=False, actual_gpu=gpu, reconciliation_required=True,
                source_proof=source_proof, protected_reference_proof=stopped.get('_qwen_reference_proof'),
                post_budget=budget, model_after=model_after, reconciliation_complete=False,
                qwen_evidence_complete=True, pending=pending)


@cleanup_signals()
def execute(args, plan):
    review = review_guard(args)
    owner = owner_identity()
    initial = original(running=False)
    saved_auth = initial_auth()
    protocol.gpu_set([])
    initial = original(running=False, references=True)
    if args.cache_baseline is None:
        raise RuntimeError('A separately frozen pre-existing-cache manifest is required')
    if args.cache_baseline.resolve() != CACHE_BASELINE.resolve():
        raise RuntimeError('Use the one immutable original-cache baseline; never rebase previous new cache into exclusions')
    baseline = protocol.read(args.cache_baseline)
    model_before = model_contract(initial, hash_payloads=True)
    protocol.save(args.run / 'model-before.json', model_before)
    production_code(plan, args.run, 'before')
    before = phase_budget(args.run, initial, EXTRA_RESERVE, baseline, owner=owner, external_ledger=args.external_ledger)
    sources = protocol.source_bindings(plan['source_hashes'])
    if sources != plan['source_hashes']:
        raise RuntimeError('Frozen executor/protocol sources changed')
    manifest = dict(plan_sha256=protocol.sha(args.run / 'plan.json'), plan=plan, owner=owner,
                    review_sha256=protocol.sha(args.review), preflight=before,
                    status='prepared_to_start_original', first_external='warmups[0]; no restore/verify/smoke POST beforehand')
    protocol.save(args.run / 'manifest.json', manifest)
    transaction_started = False
    rows, warmups = [], []
    error = cleanup_error = None
    cleanup = {}
    sampler = safety = stat_auditor = None
    try:
        stat_auditor = cache_client.Auditor(args.run / 'cache-stat-auditor', lambda: reader_phase_guard(owner), 'stat-stdio')
        stat_auditor.__enter__()
        owner_identity()
        initial_auth()
        original(running=False)
        protocol.gpu_set([])
        with wire.defer_spawn_signals():
            phase_write(saved_auth, owner, 'qwen_baseline_starting')
            transaction_started = True
        started = time.perf_counter()
        phase_deadline = time.monotonic() + PHASE_TIMEOUT
        spawn_utc = protocol.utc()
        subprocess.run(['docker', 'start', CONTAINER], check=True, capture_output=True, timeout=90)
        now = original(running=True)
        container_process = protocol.identity(now['State']['Pid'])
        with SafetyMonitor(args.run, before, initial, baseline, phase_deadline, stat_auditor) as safety, CgroupSampler(args.run / 'gpu-cgroup.jsonl', container_process) as sampler:
            polls = []
            deadline = time.monotonic() + 900
            while True:
                if not same_owner(owner):
                    raise RuntimeError('Original av2_ops owner lost during startup')
                now = original(running=True)
                try:
                    status, headers, _ = wire.complete_request(8001, '/ready', timeout=5)
                except (OSError, http.client.HTTPException):
                    status, headers = None, {}
                polls.append(dict(elapsed_seconds=time.perf_counter() - started, status=status))
                if status is not None:
                    headers_contract(headers)
                if status == 200:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Original Qwen readiness exceeded15minutes')
                time.sleep(1)
            ready_seconds = time.perf_counter() - started
            if not same_owner(owner):
                raise RuntimeError('Owner lost at readiness; do not overwrite watchdog authorization')
            phase_write(saved_auth, owner, 'qwen_baseline_active')
            baseline_guard(owner, container_process)
            if request_count() != 0:
                raise RuntimeError('Unexpected external synthesis before common warmups[0]')
            protocol.save(args.run / 'startup.json', dict(container_id=CONTAINER, image_id=IMAGE,
                docker_started_at=now['State'].get('StartedAt'), spawn_utc=spawn_utc, ready_utc=protocol.utc(),
                spawn_to_ready_seconds=ready_seconds, container_init=container_process, polls=polls,
                internal_warmup='Unmodified readiness warmup Aegis. through protected upstream; failed retry count not invented',
                cold_scope='New original-container processes; existing OS/CUDA/model caches retained; no external POST before warmups[0]'))
            for repetition in range(3):
                baseline_guard(owner, container_process)
                warmups.append(call(args, dict(id='common_warmup', text=protocol.WARMUP), repetition, True))
                protocol.save(args.run / 'warmups.json', warmups)
                if warmups[-1]['outcome'] != 'ok':
                    raise RuntimeError('Common external warmup failed')
            with (args.run / 'results.jsonl').open('x') as stream:
                for repetition in range(3):
                    for case in plan['cohort']['cases']:
                        baseline_guard(owner, container_process)
                        try:
                            row = call(args, case, repetition)
                        except Exception as failure:
                            row = dict(candidate=args.candidate, case=case['id'], size=case['size'], repetition=repetition,
                                       condition='warm', final_test=args.final_test, outcome='failed', error=repr(failure))
                        rows.append(row)
                        protocol.append_row(stream, row)
                        protocol.save(args.run / 'summary.json', protocol.metrics().summarize(rows))
            protocol.assert_matrix(rows, plan['cohort'], 3)
            if request_count() != 3 + len(rows):
                raise RuntimeError('Concurrent/unaccounted external requests')
            if any(row['outcome'] != 'ok' for row in rows):
                raise RuntimeError('Delivery matrix contains failed observations')
            baseline_guard(owner, container_process)
            protocol.save(args.run / 'contracts.json', post_contracts(args))
            baseline_guard(owner, container_process)
        if sampler.error or not sampler.samples or safety.error:
            raise RuntimeError('Sampling/safety failed: ' + str(sampler.error or safety.error))
    except BaseException as failure:
        error = repr(failure)
    finally:
        # A later stop signal must not interrupt this ownership cleanup.
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        try:
            if stat_auditor: stat_auditor.close()
        except BaseException as failure:
            cleanup_error = repr(failure)
        try:
            cleanup = cleanup_transaction(args, owner, saved_auth, transaction_started, plan)
        except BaseException as failure:
            cleanup_error = cleanup_error or repr(failure)
            # Keep qwen_baseline_* / production_stoppedFalse authorization.
            # Experimental guards therefore remain closed for subsequent jobs.
        completed = complete_matrix(rows, plan, error or cleanup_error)
        jobs = [protocol.asr_job(row, next(case for case in plan['cohort']['cases'] if case['id'] == row['case']),
                plan['cohort']['spoken'][row['case']], args.run) for row in completed if row['repetition'] == 0]
        protocol.save(args.run / 'asr-jobs.json', dict(jobs=jobs, expected_conditions=plan['cohort']['case_denominator'], filtered=False,
            policy='Parent ASR only after actual Docker stop and GPU-empty proof; no filtered successful subset'))
        protocol.save(args.run / 'results-complete.json', completed)
        try:
            same_sources = protocol.source_bindings(plan['source_hashes']) == plan['source_hashes']
        except Exception as source_error:
            same_sources = False
            error = error or repr(source_error)
        success = (not error and not cleanup_error and same_sources
                   and cleanup.get('qwen_evidence_complete') is True
                   and cleanup.get('status') == 'original_stopped_gpu_empty_budget_reconciliation_required'
                   and all(row['outcome'] == 'ok' for row in completed))
        result = dict(status='completed_pending_budget_reconciliation' if success else 'failed_requires_root_review', utc=protocol.utc(),
            error=error, cleanup_error=cleanup_error, cleanup=cleanup, sources_unchanged=same_sources,
            expected_observations=plan['cohort']['case_denominator'] * 3, observations=len(rows),
            successes=sum(row['outcome'] == 'ok' for row in completed), failures=sum(row['outcome'] != 'ok' for row in completed),
            sampled_peak_gpu_mib=sampler.peak if sampler else None,
            gpu_samples=sampler.samples if sampler else 0, asr_executed=False,
            gpu_proof_limitation='500ms live cgroup samples plus per-call checks may miss shorter overlaps',
            resume_budget_reconciliation_required=True, worker_left_running=worker_alive(owner),
            worker_not_stopped_by_executor=True, finish_gpu_created=False)
        protocol.save(args.run / 'execution.json', result)
    if not success:
        raise RuntimeError('Original Qwen baseline failed; inspect private execution/cleanup receipts')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    protocol.common_arguments(parser)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--review', type=Path, help='Explicit root-review receipt for this exact source/plan; execute only')
    parser.add_argument('--cache-baseline', type=Path, help='Separately frozen pre-existing-cache manifest; required for execute')
    parser.add_argument('--external-ledger', type=Path, help='Root-reviewed exact full NV/other-cache/owned-temp scope; execute requires it')
    parser.add_argument('--freeze-cache', action='store_true', help='Read-only cache freeze in an owned GPU-empty window; no Docker start/stop')
    args = parser.parse_args()
    os.umask(0o077)
    if args.freeze_cache:
        if args.action != 'plan' or args.final_test or not args.gpu_authorized or args.cache_baseline is None:
            raise RuntimeError('Cache freeze requires plan-mode, authorization and explicit output; no final cohort')
        freeze_owner = owner_identity()
        initial_auth()
        info = original(running=False)
        protocol.gpu_set([])
        destination = args.cache_baseline.absolute()
        if destination != CACHE_BASELINE or destination.exists() or destination.resolve() != destination:
            raise ValueError('Freeze the one original-cache baseline once; never overwrite/rebase it')
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        protocol.save(destination, capture_cache(info, destination.parent / ('cache-freeze-audit-' + str(time.time_ns())),
            lambda: reader_phase_guard(freeze_owner, quiet=True)))
        print(json.dumps(dict(status='readonly_cache_frozen_no_container_started', path=str(destination))))
        return
    # Review precedes execute's final-cohort read/hash. make_plan also invokes
    # the shared decision/roster guard before any final read, including plan.
    if args.action == 'execute':
        review_guard(args)
    if args.candidate != 'qwen-warm' or args.repetitions != 3 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Exact Qwen baseline candidate/three repetitions/finite timeout required')
    if protocol.sha(protocol.__file__) != PROTOCOL_SHA or protocol.sha(protocol.METRIC_SOURCE) != METRIC_SHA:
        raise ValueError('Frozen delivery protocol607 or metric997 changed')
    snapshot = protocol.read(OPS / 'snapshot.json')
    references = protocol.read(OPS / 'reference-hashes.json')
    if snapshot['Id'] != CONTAINER or snapshot['Image'] != IMAGE or len(references) != 77:
        raise ValueError('Expected exact original container/image and77 preserved artifact bindings')
    production = {name: protocol.sha(PREPARATION / 'production-source' / name) for name in PRODUCTION_FILES}
    charge = cache_client.external_charge()
    cache_sha = protocol.sha(args.cache_baseline) if args.cache_baseline is not None else None
    settings = dict(container_id=CONTAINER, image_id=IMAGE, voice_profile=PROFILE, runtime_variant=RUNTIME,
                    endpoint=ENDPOINT, timeout=args.timeout, seeds=dict(gateway_stage0=40733133, codec_stage1=41),
                    production_code_sha256=production, protected_artifact_count=77,
                    model_metadata_contract=model_contract(snapshot, hash_payloads=False),
                    cache_baseline_path=str(args.cache_baseline.absolute()) if args.cache_baseline else None,
                    cache_baseline_sha256=cache_sha, external_cache_charge=charge,
                    cache_reader_schema=cache_reader.SCHEMA, cache_goal_cutoff_ns=cache_reader.GOAL_START_NS,
                    extra_external_ledger=str(args.external_ledger) if args.external_ledger else None,
                    extra_external_ledger_sha256=protocol.sha(args.external_ledger) if args.external_ledger else None,
                    resource_bounds=dict(output_cap_bytes=protocol.OUTPUT_CAP, cache_growth_reserve_bytes=CACHE_GROWTH_RESERVE,
                        cache_reserve_enforcement='Planned allocation, not quota; fail at combined growth/free-space/output/40GB bounds without changing model/inference',
                        phase_timeout_seconds=PHASE_TIMEOUT, phase_stop_poll_seconds=5,
                        transient_margin_in_reservation_bytes=TRANSIENT_MARGIN,
                        cache_monitor='Metadata signatures/full changed-file fee; before.total+max(0,freeLoss-cachePhysicalDelta)+max(0,cacheFee-cacheFee0)',
                        post_cleanup_authorization='budget_reconciliation_required; oldV3 ignores supplement; new reviewed fresh gate mandatory',
                        docker_inspect_timeout_seconds=INSPECT_TIMEOUT,
                        original_stop_grace_seconds=STOP_GRACE, docker_stop_command_timeout_seconds=STOP_COMMAND_TIMEOUT,
                        docker_rw_cap_bytes=DOCKER_RW_CAP, docker_log_reserved_bytes=DOCKER_LOG_CAP,
                        extra_phase_reserve_bytes=EXTRA_RESERVE, cache_accounting='pre-existing byte-identical cache excluded; new/changed full files counted; uncertain links counted'),
                    payload='input/stream/response_format only; no protected controls',
                    lifecycle='Exact original Docker start/stop within existing av2_ops worker window; root-review receipt required')
    sources = [Path(__file__), Path(ops.__file__), Path(wire.__file__), OPS / 'snapshot.json', OPS / 'reference-hashes.json']
    sources += [Path(cache_reader.__file__), Path(cache_client.__file__), cache_client.CHARGE_PATH]
    sources += [REPO / 'training' / name for name in ('v2_gpu_worker.py', 'v2_language_monitor.py', 'v2_voxcpm_optimized_probe.py')]
    sources += [PREPARATION / 'production-source' / name for name in PRODUCTION_FILES]
    if args.cache_baseline is not None:
        sources.append(args.cache_baseline)
    if args.external_ledger is not None:
        sources += [args.external_ledger, cache_client.PUBLICATION_PATH]
    plan = protocol.make_plan(args, 'original-qwen-http', settings, sources)
    if protocol.plan_or_verify(args, plan):
        execute(args, plan)
    print(json.dumps(dict(status='executed' if args.action == 'execute' else 'prepared_only', run=str(args.run))))


if __name__ == '__main__':
    main()
