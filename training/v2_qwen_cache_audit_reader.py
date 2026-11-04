"""CPU-only cache reader proposal for the pinned image; default prints a plan.

No Docker operation is executed by this module. Root reviews/owns any future
temporary auditor. Reader modes use stdlib only and require an actual read-only
/input-cache mount, runc's GPU-free device scope, and explicit offline/no-CUDA
environment. Host tests call snapshot only on synthetic temporary files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import sys
import time

IMAGE = 'sha256:8aef74c67fe5238ce1e2b8781b9da3aedc73784014dd38bc1c1553f741fe93a0'
HOST_CACHE = '/home/phc/aegis-tts/cache/vllm-omni'
GOAL_START_NS = 1793686024 * 1_000_000_000  #2026-11-03 06:07:04 UTC.
MAX_ENTRIES = 50_000
MAX_FRAME = 32 * 1024**2
SCAN_SECONDS = 180
IDLE_SECONDS = 60
SCHEMA = 'aegis.qwen.cache-reader.v3'


def signature(value):
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_blocks, value.st_nlink]


def eligible_before_goal(row, cutoff_ns=GOAL_START_NS):
    value = row.get('metadata_signature')
    return (isinstance(value, list) and len(value) == 7 and not row.get('uncertain_metadata')
            and all(isinstance(part, int) and not isinstance(part, bool) for part in value[3:5])
            and value[3] <= cutoff_ns and value[4] <= cutoff_ns)


def snapshot(root, with_hashes, cutoff_ns=GOAL_START_NS, scan_seconds=SCAN_SECONDS):
    """FD-anchored no-follow traversal plus a second complete metadata roster."""
    root = Path(root); begun = time.monotonic(); deadline = begun + scan_seconds
    errors = []; symlinks = set()
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('Cache audit exceeded entry/chunk deadline')
    def error(path, failure):
        if len(errors) < 20: errors.append(dict(path=str(path), type=type(failure).__name__, error=str(failure)[:512]))
    def scan(hashes):
        entries = {}
        def visit(descriptor, relative):
            check(); before = os.fstat(descriptor); listing = sorted(os.listdir(descriptor))
            def add(name, details, digest=None, uncertain=False):
                if len(entries) >= MAX_ENTRIES: raise ValueError('Cache audit entry bound exceeded')
                row = dict(kind='file' if stat.S_ISREG(details.st_mode) else 'directory',
                    metadata_signature=signature(details), device=details.st_dev, inode=details.st_ino,
                    bytes=details.st_size, allocated_bytes=details.st_blocks*512, links=details.st_nlink,
                    sha256=digest, uncertain_metadata=uncertain)
                row['eligible_before_goal'] = eligible_before_goal(row, cutoff_ns); entries[name] = row
            add(relative, before)
            for name in listing:
                check(); label = str(Path(relative) / name)
                try:
                    observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if stat.S_ISLNK(observed.st_mode):
                        symlinks.add(label); raise ValueError('Symlink cache entry denied')
                    if stat.S_ISDIR(observed.st_mode):
                        child = os.open(name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=descriptor)
                        try:
                            if signature(os.fstat(child)) != signature(observed): raise ValueError('Directory identity changed before traversal')
                            visit(child, label)
                            if signature(os.stat(name,dir_fd=descriptor,follow_symlinks=False)) != signature(os.fstat(child)):
                                raise ValueError('Directory entry changed during traversal')
                        finally: os.close(child)
                    elif stat.S_ISREG(observed.st_mode):
                        digest = None
                        if hashes:
                            child = os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=descriptor)
                            try:
                                if signature(os.fstat(child)) != signature(observed): raise ValueError('File identity changed before hashing')
                                value=hashlib.sha256()
                                while True:
                                    check(); block=os.read(child,1024**2)
                                    if not block: break
                                    value.update(block)
                                digest=value.hexdigest()
                                if signature(os.fstat(child)) != signature(observed): raise ValueError('File changed while hashing')
                            finally: os.close(child)
                        after=os.stat(name,dir_fd=descriptor,follow_symlinks=False)
                        if signature(after) != signature(observed): raise ValueError('File entry changed during scan')
                        add(label,after,digest)
                    else: raise ValueError('Unsupported nonregular cache entry')
                except TimeoutError: raise
                except (OSError,ValueError) as failure: error(label,failure)
            if listing != sorted(os.listdir(descriptor)) or signature(os.fstat(descriptor)) != signature(before):
                error(relative,ValueError('Directory roster/metadata changed during scan'))
        descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try: visit(descriptor,'.')
        finally: os.close(descriptor)
        return entries
    entries=scan(with_hashes); verified=scan(False)
    if set(entries)!=set(verified): error('.',ValueError('Second metadata roster differs'))
    for name,row in entries.items():
        if verified.get(name,{}).get('metadata_signature') != row['metadata_signature']:
            error(name,ValueError('Entry changed across complete metadata verification'))
    grouped={}
    for name,row in entries.items(): grouped.setdefault((row['device'],row['inode']),[]).append((name,row))
    physical=eligible=uncertain_links=0
    for names in grouped.values():
        row=names[0][1]; physical+=max(item['allocated_bytes'] for _,item in names)
        if len({tuple(item['metadata_signature']) for _,item in names})!=1:
            error(names[0][0],ValueError('Inconsistent hardlink metadata'))
        closed=row['kind']=='directory' or row['links']==len(names)
        if not closed: uncertain_links+=1
        if row['kind']=='file' and closed and all(item['eligible_before_goal'] for _,item in names): eligible+=row['allocated_bytes']
    complete=not errors
    return dict(schema=SCHEMA,complete=complete,root=str(root),cutoff_unix_ns=cutoff_ns,
        policy='Regular pregoal mtime AND ctime + closed links + invariance only; all directory/postgoal/uncertain bytes charged',
        hashes_requested=with_hashes,entries=entries,errors=errors,skipped_symlinks=len(symlinks),entries_count=len(entries),
        physical_bytes=physical,eligible_before_goal_candidate_bytes=eligible if complete else 0,
        charged_post_goal_or_uncertain_bytes=physical-eligible if complete else None,
        uncertain_external_hardlink_inodes=uncertain_links,scan_seconds=time.monotonic()-begun,
        fd_anchored=True,metadata_roster_passes=2,
        deadline_limitation='Entry/chunk checks, not a hard OS-I/O quota; concurrent later writes are checked at subsequent bounded polls')


def accounted_fee(baseline, current):
    """Shared future consumer: post-goal files never become baseline credits."""
    for frame in (baseline, current):
        if (frame.get('schema') != SCHEMA or frame.get('complete') is not True
                or frame.get('cutoff_unix_ns') != GOAL_START_NS or frame.get('fd_anchored') is not True
                or frame.get('metadata_roster_passes') != 2):
            raise ValueError('Complete exact goal-cutoff cache frame required')
    if baseline.get('hashes_requested') is not True:
        raise ValueError('Baseline requires quiet full-content hash proof')
    if baseline.get('root') != current.get('root'):
        raise ValueError('Exact cache reader root must match')
    grouped = {}
    for path, row in current['entries'].items():
        name = Path(path)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe relative cache artifact path')
        if (row.get('kind') not in ('file', 'directory') or not isinstance(row.get('metadata_signature'), list)
                or len(row['metadata_signature']) != 7):
            raise ValueError('Unknown cache metadata is not an exclusion proof')
        value = row['metadata_signature']
        if (not all(isinstance(part, int) and not isinstance(part, bool) for part in value)
                or value[0] != row['device'] or value[1] != row['inode'] or value[2] != row['bytes']
                or value[5] * 512 != row['allocated_bytes'] or value[6] != row['links']
                or row['allocated_bytes'] < 0 or row['links'] < 1):
            raise ValueError('Cache stat/allocation signature mismatch')
        grouped.setdefault((row['device'], row['inode']), []).append((path, row))
    physical = excluded = 0
    charged_paths = []
    for names in grouped.values():
        sample = names[0][1]
        if len({tuple(row['metadata_signature']) for _,row in names}) != 1:
            raise ValueError('Inconsistent hardlink observations')
        physical += sample['allocated_bytes']
        def unchanged(path, row):
            old = baseline['entries'].get(path, {})
            if not eligible_before_goal(old) or not eligible_before_goal(row): return False
            if old.get('kind') != row['kind']: return False
            if row['kind'] == 'directory': return False  # Root's conservative ledger charges every directory block.
            if old.get('metadata_signature') != row['metadata_signature']: return False
            if not isinstance(old.get('sha256'), str) or len(old['sha256']) != 64: return False
            return not current.get('hashes_requested') or old['sha256'] == row.get('sha256')
        internal_links = sample['kind'] == 'directory' or sample['links'] == len(names)
        if internal_links and all(unchanged(path, row) for path, row in names):
            old_blocks = min(baseline['entries'][path]['allocated_bytes'] for path, _ in names)
            excluded += min(old_blocks, sample['allocated_bytes'])
        else:
            charged_paths.extend(path for path, _ in names)
    return dict(physical_bytes=physical, excluded_goal_original_bytes=excluded,
                accounted_bytes=physical - excluded, charged_paths=charged_paths,
                goal_start_unix_ns=GOAL_START_NS, post_goal_files_always_charged=True)


def mount_unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda found: chr(int(found.group(1), 8)), value)


def reader_guard(root):
    if Path(root) != Path('/input-cache') or os.geteuid() != 0:
        raise RuntimeError('Reader requires exact input-cache mount and root0 in temporary auditor')
    mounts = [line.split(' - ', 1)[0].split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
    exact = [parts for parts in mounts if mount_unescape(parts[4]) == '/input-cache']
    if len(exact) != 1 or 'ro' not in exact[0][5].split(','):
        raise RuntimeError('Cache mount must actually be read-only')
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Explicit no-GPU environment required')
    if any(Path('/dev').glob('nvidia*')) or 'torch' in sys.modules:
        raise RuntimeError('GPU device or Torch detected in CPU auditor')
    effective = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('CapEff:'))
    if int(effective, 16) != (1 << 2):  #CAP_DAC_READ_SEARCH, read/search only; no write override.
        raise RuntimeError('Expected only DAC_READ_SEARCH capability for mixed-UID cache read access')
    for line in Path('/proc/net/route').read_text().splitlines()[1:]:
        if line.split() and line.split()[1] == '00000000':
            raise RuntimeError('Auditor unexpectedly has a default network route')
    return dict(read_only_cache=True, uid=0, torch_imported=False, cuda_devices=False,
                runtime_expected='runc', network_expected='none', proof='Host must independently inspect image/runtime/mounts/device configuration before start')


def encode_frame(value):
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    if len(payload) > MAX_FRAME:
        raise ValueError('Reader frame exceeds32MiB; no partial exclusion')
    return payload


def emit(value):
    sys.stdout.buffer.write(encode_frame(value)); sys.stdout.buffer.flush()


def command_lines(stream):
    """Bounded pipe reader; an orphan idle auditor exits without host mutation."""
    pending = b''
    with selectors.DefaultSelector() as selector:
        selector.register(stream.fileno(), selectors.EVENT_READ)
        while True:
            if b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                if len(line) + 1 > 4096: raise ValueError('Input command bound exceeded')
                yield line
                continue
            if len(pending) > 4096: raise ValueError('Input command bound exceeded')
            if not selector.select(IDLE_SECONDS): raise TimeoutError('CPU cache auditor idle60s; no persistent orphan')
            block = os.read(stream.fileno(), 4097 - len(pending))
            if not block:
                if pending: raise ValueError('Incomplete input command at EOF')
                return
            pending += block


def proposal(mode, name, output):
    if mode not in ('hash', 'stat-stdio') or not re.fullmatch(r'aegis-qwen-cache-[a-z0-9-]{1,48}', name):
        raise ValueError('Explicit safe temporary auditor name/mode required')
    source = str(Path(__file__).resolve())
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    create = ['docker', 'create', '--pull', 'never', '--name', name, '--label', 'aegis.root.audit=qwen-cache-goal-accounting',
              '--label', 'aegis.root.audit-name=' + name, '--label', 'aegis.root.source-sha256=' + source_sha,
              '--runtime', 'runc', '--network', 'none', '--read-only', '--user', '0:0', '--cap-drop', 'ALL',
              '--cap-add', 'DAC_READ_SEARCH',
              '--security-opt', 'no-new-privileges:true', '--pids-limit', '32', '--memory', '512m', '--cpus', '0.5',
              '--log-driver', 'none', '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--env', 'CUDA_VISIBLE_DEVICES=',
              '--mount', 'type=bind,src=' + HOST_CACHE + ',dst=/input-cache,readonly',
              '--mount', 'type=bind,src=' + source + ',dst=/audit/reader.py,readonly', '--entrypoint', 'python3', '-i',
              IMAGE, '-I', '-S', '-B', '-u', '/audit/reader.py', '--reader-mode', mode, '--root', '/input-cache']
    return dict(status='proposal_only_no_docker_operation', schema=SCHEMA, image=IMAGE,
                source_sha256=source_sha, cutoff_unix_ns=GOAL_START_NS,
                create_argv=create, inspect_argv=['docker', 'inspect', '<CREATED_FULL_ID>'], start_argv=['docker', 'start', '-a', '-i', '<CREATED_FULL_ID>'],
                cleanup_argv=['docker', 'rm', '<CREATED_FULL_ID>'], private_stdout=str(output), stdout_cap_bytes=MAX_FRAME,
                stderr_cap_bytes=64 * 1024, operation_timeout_seconds=10, scan_timeout_seconds=SCAN_SECONDS,
                persistent_idle_timeout_seconds=IDLE_SECONDS,
                actual_execution=False, requirements=['Root must inspect exact image/labels/runtime=runc/network=none/cacheRO/codeRO/user0/devices empty and only DAC_READ_SEARCH before start',
                    'Capture stdout/stderr privately with bounded pipes; reject incomplete/error/partial frame or missing EOF/trailer',
                    'After timeout stop/remove ONLY this audited temporary container, never original production',
                    'Bind create stdout64hex ID to image/owner labels before start/cleanup; name collision fails, never remove pre-existing name',
                    'Retain only latest metadata frame, bounded summaries and quiet pre/post hash manifests; never append all persistent frames',
                    'After owner loss reader exits on EOF/idle60s; root must remove its recorded stopped temp ID, never a same-name unknown container',
                    'DAC_READ_SEARCH is a proposed reviewed read/search capability, not DAC_OVERRIDE; no chmod/chown or automatic capability fallback',
                    'If daemon/userns cannot grant mixed-UID read access, unreadable input fails closed; no partial baseline credit',
                    'Future Qwen/gate must use this reader consistently; host reader8df fails on root-owned cache subtree',
                    'No image build/pull/env install/model refs mounts; Docker writable/log growth still accounted'],
                persistent_protocol='ready frame; each stdin JSON {op:snapshot,request_id:string} yields one bounded metadata-only frame; {op:quit} exits; no unsolicited scan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader-mode', choices=('hash', 'stat-stdio'))
    parser.add_argument('--root', type=Path, default=Path('/input-cache'))
    parser.add_argument('--mode', choices=('hash', 'stat-stdio'), default='hash', help='Plan mode only')
    parser.add_argument('--audit-name', default='aegis-qwen-cache-proposed-20261104')
    parser.add_argument('--private-output', type=Path, default=Path('/var/tmp/aegis-v2-work/operations/qwen-cache-reader-proposal-20261104/stdout.private.json'))
    args = parser.parse_args()
    if args.reader_mode is None:
        print(json.dumps(proposal(args.mode, args.audit_name, args.private_output), ensure_ascii=False, indent=2)); return
    guard = reader_guard(args.root)
    if args.reader_mode == 'hash':
        value = snapshot(args.root, with_hashes=True); value['reader_guard'] = guard; emit(value)
        raise SystemExit(0 if value['complete'] else 2)
    emit(dict(schema=SCHEMA, ready=True, mode='stat-stdio', reader_guard=guard))
    for line in command_lines(sys.stdin.buffer):
        command = json.loads(line)
        if command.get('op') == 'quit':
            emit(dict(schema=SCHEMA, stopped=True)); return
        if command.get('op') != 'snapshot' or not isinstance(command.get('request_id'), str) or len(command['request_id']) > 128:
            raise ValueError('Invalid metadata request')
        value = snapshot(args.root, with_hashes=False); value['request_id'] = command['request_id']; emit(value)
        if not value['complete']: raise SystemExit(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as failure:
        emit(dict(schema=SCHEMA, complete=False, error_type=type(failure).__name__, error=str(failure)[:2048]))
        raise SystemExit(2)
