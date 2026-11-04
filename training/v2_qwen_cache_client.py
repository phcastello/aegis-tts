"""Owned bounded CPU-only auditor client. Import never launches Docker."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import time
import uuid
from training import v2_qwen_cache_audit_reader as reader

ORIGINAL = '08004c66f014e3e63fb6fd123946e4dbba2919ce29b4530848279917325faa75'
CHARGE_PATH = Path('/var/tmp/aegis-v2-work/operations/cache-metadata-readonly-cpu-20261104/outside-cache-conservative-charge.json')
CHARGE_SHA = 'c661aeceead032c9bf46bd27cb0a1dbf02d85422b35a9e0d4651e2e71da28ff1'
CHARGE_FLOOR = 21_049_344
NV_ROOT = Path('/home/phc/.nv')
PUBLICATION_PATH = CHARGE_PATH.parent.parent / '097f-main-resume-publication-20261104.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_charge():
    if sha(CHARGE_PATH) != CHARGE_SHA: raise RuntimeError('Published outside-cache charge changed')
    value = json.loads(CHARGE_PATH.read_text())
    if (value.get('cache_root') != reader.HOST_CACHE or value.get('goal_start_epoch') * 10**9 != reader.GOAL_START_NS
            or value.get('charged_new_or_uncertain_cache_bytes') != CHARGE_FLOOR
            or value.get('payload_read') is not False):
        raise RuntimeError('Exact published external-charge scope required')
    return dict(path=str(CHARGE_PATH), sha256=CHARGE_SHA, minimum_accounted_bytes=CHARGE_FLOOR,
                metadata_sha256=value['complete_metadata_sha256'])


def fee(baseline, current):
    result = reader.accounted_fee(baseline, current)
    result['measured_accounted_bytes'] = result['accounted_bytes']
    result['accounted_bytes'] = max(result['accounted_bytes'], CHARGE_FLOOR)
    result['published_charge_floor_bytes'] = CHARGE_FLOOR
    return result


def extra_external(ledger_path, core_roots):
    """Fresh full-byte external trees; no baseline credits outside vllm."""
    if ledger_path is None: raise RuntimeError('Exact reviewed external-ledger metadata required')
    path = Path(ledger_path)
    if path.is_symlink() or path.stat().st_uid != os.getuid() or path.stat().st_size > 65536:
        raise RuntimeError('Unsafe external-ledger specification')
    value = json.loads(path.read_text())
    if (value.get('schema') != 'aegis.external-cache-ledger.v1' or value.get('goal_start_epoch') != 1793686024
            or value.get('temporary_scope_complete') is not True):
        raise RuntimeError('Complete goal-scoped temporary ledger required')
    if value.get('source_receipt') != str(PUBLICATION_PATH) or value.get('source_receipt_sha256') != sha(PUBLICATION_PATH):
        raise RuntimeError('Exact root097f external-scope publication required')
    publication=json.loads(PUBLICATION_PATH.read_text())
    published=publication['other_external_whole_roots']
    required_temporary={name for name in published['roots'] if Path(name).is_relative_to('/tmp')}
    if (publication.get('production_cache_fee_bytes')!=CHARGE_FLOOR
            or not required_temporary <= set(value['owned_temporary_paths'])
            or value.get('minimum_additional_external_bytes',0)<published['allocated_bytes']):
        raise RuntimeError('Published whole external paths/charge omitted')
    parent = Path(reader.HOST_CACHE).parent
    roots = [NV_ROOT, *(item for item in parent.iterdir() if item.name != Path(reader.HOST_CACHE).name)]
    for item in value['owned_temporary_paths']:
        temporary = Path(item)
        if not temporary.is_absolute() or not temporary.is_relative_to('/tmp') or temporary.is_symlink():
            raise RuntimeError('Temporary ledger must contain exact owned /tmp paths')
        if temporary.exists() and temporary.stat().st_uid != os.getuid():
            raise RuntimeError('Foreign temporary path in ledger')
        roots.append(temporary)
    selected = []; already = []
    for item in roots:
        if item.is_symlink(): raise RuntimeError('External cache scope contains symlink; reconcile explicitly')
        if any(item == root.resolve() or item.is_relative_to(root.resolve()) for root in core_roots): already.append(str(item))
        else: selected.append(item)
    seen = set(); allocated = 0; missing = []
    stack = list(selected)
    while stack:
        item = stack.pop()
        try: details = item.lstat()
        except FileNotFoundError:
            missing.append(str(item)); continue
        key = (details.st_dev, details.st_ino)
        if key in seen: continue
        seen.add(key); allocated += details.st_blocks * 512
        if stat.S_ISDIR(details.st_mode):
            with os.scandir(item) as entries: stack.extend(Path(entry.path) for entry in entries)
    usage = dict(allocated_bytes=allocated, missing_paths=missing, permission_errors=[],
                 policy='Full allocated blocks incl symlink/special inodes, no symlink traversal, inode dedup')
    minimum = value.get('minimum_additional_external_bytes', 0)
    if type(minimum) is not int or minimum < 299008 + 2646016:
        raise RuntimeError('Ledger omitted published full NV/other production-cache charge')
    return dict(specification_path=str(path), specification_sha256=sha(path), roots=[str(v) for v in selected],
                already_in_core=already, inventory=usage, measured_full_bytes=usage['allocated_bytes'],
                accounted_bytes=max(usage['allocated_bytes'], minimum), minimum_published_bytes=minimum)


def save(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600); temporary.replace(path)


class Auditor:
    """One unique temporary container, no original-container command target."""
    def __init__(self, run, guard, mode='hash'):
        self.run = Path(run); self.guard = guard; self.mode = mode
        self.name = 'aegis-qwen-cache-' + uuid.uuid4().hex[:24]
        self.plan = reader.proposal(mode, self.name, self.run / 'latest.private.json')
        self.container = None; self.validated = False; self.process = None
        self.pending = b''; self.stderr = b''; self.selector = None; self.requests = 0
        self.error = None; self.proof = None
        self.closed = False

    def inspect(self):
        return json.loads(subprocess.check_output(['docker', 'inspect', self.container], text=True, timeout=10))[0]

    def validate(self, value):
        config, host = value['Config'], value['HostConfig']
        env = dict(item.split('=', 1) for item in config.get('Env', []) if '=' in item)
        labels = config.get('Labels', {})
        normalize = lambda values: sorted(str(v).upper().removeprefix('CAP_') for v in (values or []))
        expected_mounts = {'/input-cache': str(Path(reader.HOST_CACHE).resolve()), '/audit/reader.py': str(Path(reader.__file__).resolve())}
        mounts = value.get('Mounts', [])
        if (value.get('Id') != self.container or self.container == ORIGINAL or value.get('Image') != reader.IMAGE
                or config.get('User') != '0:0' or config.get('Entrypoint') != ['python3']
                or config.get('Cmd') != ['-I', '-S', '-B', '-u', '/audit/reader.py', '--reader-mode', self.mode, '--root', '/input-cache']
                or labels.get('aegis.root.audit-name') != self.name
                or labels.get('aegis.root.source-sha256') != self.plan['source_sha256']
                or labels.get('aegis.root.audit') != 'qwen-cache-goal-accounting'
                or host.get('Runtime') != 'runc' or host.get('NetworkMode') != 'none'
                or host.get('ReadonlyRootfs') is not True or host.get('Privileged') is not False
                or normalize(host.get('CapAdd')) != ['DAC_READ_SEARCH'] or normalize(host.get('CapDrop')) != ['ALL']
                or host.get('Devices') or host.get('DeviceRequests') or host.get('PortBindings')
                or host.get('LogConfig', {}).get('Type') != 'none'
                or host.get('Memory') != 512 * 1024**2 or host.get('NanoCpus') != 500_000_000
                or host.get('PidsLimit') != 32
                or 'no-new-privileges:true' not in (host.get('SecurityOpt') or [])
                or env.get('NVIDIA_VISIBLE_DEVICES') != 'void' or env.get('CUDA_VISIBLE_DEVICES') != ''
                or len(mounts) != 2 or {m['Destination'] for m in mounts} != set(expected_mounts)
                or any(m.get('Type') != 'bind' or m.get('RW') is not False
                       or str(Path(m['Source']).resolve()) != expected_mounts[m['Destination']] for m in mounts)):
            raise RuntimeError('Temporary CPU auditor identity/configuration differs; no start')
        return dict(container_id=self.container, image_id=value['Image'], runtime='runc', network='none',
                    mounts_readonly=True, root0=True, cap_read_search_only=True, gpu_devices=False,
                    source_sha256=self.plan['source_sha256'], name=self.name)

    def __enter__(self):
        self.run.mkdir(parents=True, exist_ok=False, mode=0o700)
        save(self.run / 'plan.json', self.plan)
        try:
            self.guard()
            created = subprocess.run(self.plan['create_argv'], check=True, capture_output=True, text=True, timeout=10)
            identifier = created.stdout.strip()
            if not re.fullmatch('[0-9a-f]{64}', identifier) or identifier == ORIGINAL:
                raise RuntimeError('Invalid new auditor ID; original container never a cleanup target')
            self.container = identifier
            self.proof = self.validate(self.inspect()); self.validated = True
            self.guard()
            self.process = subprocess.Popen(['docker', 'start', '-a', '-i', self.container],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ, 'stdout')
            self.selector.register(self.process.stderr, selectors.EVENT_READ, 'stderr')
            if self.mode == 'stat-stdio':
                ready = self.read_frame(20)
                if ready.get('ready') is not True or ready.get('schema') != reader.SCHEMA or ready.get('mode') != self.mode:
                    raise RuntimeError('CPU auditor readiness differs')
            return self
        except BaseException:
            self.close()
            raise

    def read_frame(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if b'\n' in self.pending:
                raw, self.pending = self.pending.split(b'\n', 1)
                if len(raw) + 1 > reader.MAX_FRAME: raise RuntimeError('CPU auditor frame cap exceeded')
                return json.loads(raw)
            if len(self.pending) > reader.MAX_FRAME: raise RuntimeError('CPU auditor frame cap exceeded')
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('CPU auditor response deadline')
            events = self.selector.select(min(.5, remaining))
            for key, _ in events:
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    self.selector.unregister(key.fileobj)
                    if key.data == 'stdout': raise RuntimeError('Incomplete CPU auditor stdout/EOF')
                    continue
                if key.data == 'stdout': self.pending += block
                else:
                    self.stderr += block
                    if len(self.stderr) > 65536: raise RuntimeError('CPU auditor stderr cap exceeded')

    def snapshot(self):
        self.guard()
        identifier = None
        if self.mode == 'stat-stdio':
            self.requests += 1; identifier = str(self.requests)
            self.process.stdin.write(json.dumps(dict(op='snapshot', request_id=identifier)).encode() + b'\n')
            self.process.stdin.flush()
        value = self.read_frame(20 if self.mode == 'stat-stdio' else reader.SCAN_SECONDS + 10)
        if (value.get('schema') != reader.SCHEMA or value.get('complete') is not True
                or value.get('root') != '/input-cache' or value.get('cutoff_unix_ns') != reader.GOAL_START_NS
                or value.get('fd_anchored') is not True or value.get('metadata_roster_passes') != 2
                or value.get('hashes_requested') is not (self.mode == 'hash')
                or (identifier is not None and value.get('request_id') != identifier)):
            raise RuntimeError('Incomplete/wrong-scope CPU cache snapshot')
        if self.mode == 'hash' and self.process.wait(timeout=5) != 0:
            raise RuntimeError('Hash auditor exited unsuccessfully')
        if self.mode == 'hash': reader.accounted_fee(value,value)
        return value

    def close(self):
        if self.closed: return
        self.closed = True
        failure = None
        try:
            if self.validated and self.container and self.container != ORIGINAL:
                # Exact created ID/config was proven before start. No name lookup
                # or original-container action, even if owner subsequently dies.
                subprocess.run(['docker', 'stop', '--time', '1', self.container], check=True, capture_output=True, timeout=5)
                subprocess.run(['docker', 'rm', '-v', self.container], check=True, capture_output=True, timeout=5)
            elif self.container:
                failure = 'Created but unvalidated temporary auditor retained for root inspection; never started'
        except BaseException as error:
            failure = repr(error)
        finally:
            if self.selector: self.selector.close()
            if self.process:
                for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                    if stream: stream.close()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: failure = failure or 'Attached CPU auditor CLI did not exit'
            self.error = failure
            save(self.run / 'receipt.json', dict(container_id=self.container, validated=self.validated, proof=self.proof,
                 cleanup_error=failure, removed=bool(self.validated and not failure), request_count=self.requests,
                 stderr_bytes=len(self.stderr), source_sha256=self.plan['source_sha256']))
        if failure: raise RuntimeError('CPU auditor cleanup failed; root review required')

    def __exit__(self, *_): self.close()


def quiet_snapshot(run, guard):
    with Auditor(run, guard, 'hash') as auditor:
        result = auditor.snapshot()
    return result
