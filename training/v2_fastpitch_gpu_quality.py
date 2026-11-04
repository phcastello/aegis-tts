"""Validation7 CUDA quality only. Planning/imports are stdlib-only; never runs ASR.

The GPU-worker launches one controller per checkpoint. The controller execs its
neural worker in the same PID, preserving the existing immediate-parent guard.
Two queue jobs therefore create two fresh CUDA processes, with no detached child.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
import wave

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/var/tmp/aegis-v2-work')
BASELINE = ROOT / 'fastpitch/common-cpu-language-control-step10000/manifest.json'
PREPARATION = ROOT / 'fastpitch/gpu-language-quality-preparation-20261104'
SEED = 40733133
CANONICAL_10000 = 'd23db54a42d25c27c369af43c8bc8606ab86c533116dcddb93ea642c7a4b5058'
CASE_IDS = ('v_command', 'v_reply', 'v_pressure', 'v_acronyms', 'v_question', 'v_sibilants', 'v_long')
OUTPUT_CAP = 72_000_000
CACHE_CAP = 256 * 1024**2
WORKER_LOG_CAP = 32 * 1024**2
PAIR_EXTRA = 2 * OUTPUT_CAP + CACHE_CAP + 2 * WORKER_LOG_CAP
RSS_CAP = 6 * 1024**3
WALL_SECONDS = 900
MAX_AUDIO_SECONDS = 60
MAX_TOKENS = 2048
LOG_CAP = 1024**2
PINS = {
    'training/fastpitch_experiment.py': '453c3806b384c0af8aa0fdc1cc79b97282b6940f971cc84ac01ed35c1011705d',
    'training/v2_fastpitch_benchmark.py': '0e108b2dacceb924019a6419dfe9cd89e65457700f123a230a8956d55538d17e',
    'training/v2_fastpitch_common_cpu.py': 'ae66058314c5b153b3344af4b4f24c56ca742cd295292341a8b6e7e1ee2ca42f',
    'training/v2_fastpitch_pace_control.py': '67bee5515049dae75d61898783efd86090e2bb26555fec917f007299de8cd13e',
    'training/f5_refine_text.py': '9528bcb5b9a41f68778a4f9ae0956af9edebc6c0365b604ea60dcbbcc6734c12',
    'gateway/aegis_speech/normalization.py': '01fbd3eb902637479ae6022bdac65b6a54b9a2ffd85f8628323034237e3c9ee0',
    'benchmarks/v2/benchmark.py': '997eed71e5066509dc7569e31e663733ab0977468c53bdd3b350a194a12f743f',
    'benchmarks/v2/asr_score.py': '4ee9f7da2cc8c733badb4d6d7c248a96ce0204a394fc26dc7652801a1410f241',
    'training/v2_gpu_worker.py': '22fd880706139618e80fa608d6eaa881b177f273c8cbf89a4e62b0809e7927d4',
    'training/v2_language_monitor.py': 'f086f62f650f4a033004df160f3a69d77b98676ac4d3088eed67fb4e2d092135',
    'benchmarks/v2/validation.pt-BR.json': '16c54bbc362229f0edde145506d3b79db451c65f61ca98d69806ebf8d975a247',
    'benchmarks/v2/spoken-validation.pt-BR.json': 'fc6b79a709d87c05a2538832fec49af6de125b7109ce527855df3b6dcb857011',
}


class QualityAbort(BaseException):
    """Resource/owner failure must escape the per-case Exception handler."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def safe(path, directory=False):
    path = Path(path).absolute()
    if path.resolve() != path or '..' in path.parts:
        raise ValueError('Canonical path without symlinks required: ' + str(path))
    info = path.lstat()
    if info.st_uid != os.getuid() or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError('Expected own regular artifact: ' + str(path))
    return path


def read(path):
    path = safe(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError('Oversize JSON metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if len(encoded) > 4 * 1024**2:
        raise ValueError('Oversize output metadata')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def event(output, value):
    path = Path(output) / 'events.jsonl'
    encoded = (json.dumps(dict(utc=utc(), **value), allow_nan=False) + '\n').encode()
    if len(encoded) > LOG_CAP:
        raise ValueError('Oversize event')
    if path.exists() and path.stat().st_size + len(encoded) > LOG_CAP:
        os.replace(path, path.with_suffix('.previous.jsonl'))
    with path.open('ab') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_step(step):
    if step not in (10000, 16000):
        raise ValueError('Only exact validation checkpoints10000/16000 are supported')
    return step


def validate_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Exact lowercase SHA256 required')
    return value


def signature(path):
    info = safe(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]


def checkpoint_binding(path, expected, expected_sha):
    validate_step(expected)
    validate_sha(expected_sha)
    path = safe(path)
    if expected == 10000 and expected_sha != CANONICAL_10000:
        raise ValueError('Step10000 must use the canonical common-best SHA')
    before = signature(path)
    sidecar = read(path.with_suffix('.sha256.json'))
    if sidecar.get('step') != expected or sidecar.get('sha256') != expected_sha or sidecar.get('bytes') != before[2]:
        raise ValueError('Checkpoint sidecar step/SHA/bytes differ')
    if sha(path) != expected_sha or signature(path) != before:
        raise ValueError('Checkpoint changed during binding')
    return dict(path=str(path), step=expected, sha256=expected_sha, bytes=before[2], signature=before,
                sidecar_path=str(path.with_suffix('.sha256.json')), sidecar_sha256=sha(path.with_suffix('.sha256.json')))


def source_bindings(source, bigvgan):
    values = {}
    for name, expected in PINS.items():
        path = safe(REPO / name)
        if sha(path) != expected:
            raise ValueError('Immutable source differs: ' + name)
        values[str(path)] = expected
    if sha(safe(source / 'UPSTREAM.json')) != '6215561101b7d6746b5d9b43fe67afa853e7a1db7363206033ad75c7b0183515':
        raise ValueError('NVIDIA pinned source manifest differs')
    upstream = read(source / 'UPSTREAM.json')
    for entry in upstream['files']:
        if entry['path'].endswith('.py') and sha(safe(source / entry['path'])) != entry['sha256']:
            raise ValueError('NVIDIA pinned runtime source differs: ' + entry['path'])
    from training import v2_fastpitch_benchmark as common
    if sha(safe(bigvgan / 'bigvgan_generator.pt')) != common.BIG_WEIGHT_SHA:
        raise ValueError('Pinned BigVGAN weights differ')
    paths = [Path(__file__), BASELINE, source / 'UPSTREAM.json', bigvgan / 'config.json', bigvgan / 'revision.json',
             bigvgan / 'bigvgan_generator.pt']
    paths += sorted(source.rglob('*.py'))
    paths += sorted((bigvgan / 'source').rglob('*.py'))
    for path in paths:
        values[str(safe(path))] = sha(path)
    return values


def verify_bindings(plan):
    for raw, expected in plan['source_hashes'].items():
        if sha(safe(raw)) != expected:
            raise ValueError('Frozen source/validation evidence changed: ' + raw)
    cp = plan['checkpoint']
    if signature(cp['path']) != cp['signature'] or sha(cp['path']) != cp['sha256']:
        raise ValueError('Frozen checkpoint changed')
    if sha(safe(cp['sidecar_path'])) != cp['sidecar_sha256']:
        raise ValueError('Checkpoint sidecar changed')


def planned(args):
    validate_step(args.expected_step)
    bindings = source_bindings(args.source, args.bigvgan)
    baseline = read(BASELINE)['plan']
    cases = read(REPO / 'benchmarks/v2/validation.pt-BR.json')
    if tuple(case['id'] for case in cases) != CASE_IDS or cases != baseline['cases']:
        raise ValueError('Only the seven fixed validation cases are permitted')
    # The existing planner extracts the frozen pure normalize() via AST; no Torch.
    from training.v2_fastpitch_pace_control import expected_frontend
    effective = expected_frontend(cases)
    if effective != baseline['effective_texts']:
        raise ValueError('Frontend differs from common CPU10k')
    cp = checkpoint_binding(args.checkpoint, args.expected_step, args.checkpoint_sha256)
    return dict(schema='aegis.fastpitch.gpu-quality.v1', status='prepared_only', created_utc=utc(),
                expected_step=args.expected_step, checkpoint=cp, output=str(args.output),
                source=str(args.source), bigvgan=str(args.bigvgan), root=str(ROOT),
                source_hashes=bindings, cases=cases, effective_texts=effective,
                spoken=read(REPO / 'benchmarks/v2/spoken-validation.pt-BR.json')['cases'],
                seed=SEED, pace=1.0, frontend='refinement unchanged; text-only native duration/pitch',
                observations=7, per_case=1, warmups=0, precision=dict(acoustic_autocast='bf16', parameters='FP32', vocoder='FP32'),
                device='cuda', max_audio_seconds=MAX_AUDIO_SECONDS, max_tokens=MAX_TOKENS,
                output_cap_bytes=OUTPUT_CAP, rss_cap_bytes=RSS_CAP, wall_seconds=WALL_SECONDS,
                phase_budget=dict(pair_output_caps_bytes=2 * OUTPUT_CAP, cuda_cache_cap_bytes=CACHE_CAP,
                                  pair_external_worker_log_caps_bytes=2 * WORKER_LOG_CAP,
                                  additional_pair_upper_bytes=PAIR_EXTRA, future_training_checkpoint_writes_reserved=0,
                                  note='Main must be quiescent. Resume requires the separate v3 six-state budget guard.'),
                asr=baseline['asr'], asr_execution='Never here. Parent scores14 fixed jobs only after both GPU processes exited.',
                attribution='Joint device/precision/backend comparison; does not isolate BF16 from CPU/CUDA.',
                scope='Validation quality only. No final-data parser, HTTP, streaming, TTFA, latency or identity approval.',
                torch_imported='torch' in sys.modules, gpu_used=False, tts_executed=False)


def process_identity(pid):
    try:
        path = Path('/proc') / str(int(pid))
        before = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        command = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        after = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if before[19] != after[19] or after[0] == 'Z':
            return None
        return dict(pid=int(pid), ppid=int(after[1]), start_ticks=after[19], uid=path.stat().st_uid,
                    command=[word for word in command if word])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
        return None


def flag(command, key):
    try:
        return command[command.index(key) + 1]
    except (ValueError, IndexError):
        return next((item.split('=', 1)[1] for item in command if item.startswith(key + '=')), None)


def guard_contract(me, parent, anchor, authorization, root):
    if not me or not parent or me['uid'] != os.getuid() or parent['uid'] != os.getuid() or me['ppid'] != parent['pid']:
        raise RuntimeError('Expected own immediate GPU-worker parent')
    if flag(parent['command'], '-m') != 'training.v2_gpu_worker' or flag(parent['command'], '--root') != str(root):
        raise RuntimeError('Actual parent is not the project GPU-worker')
    if anchor.get('child_pid') != parent['pid'] or str(anchor.get('child_start')) != str(parent['start_ticks']):
        raise RuntimeError('GPU-worker PID/start differs from owner window')
    if authorization.get('status') != 'active' or any(authorization.get(k) is not True for k in
            ('owner_authorized', 'production_stopped', 'exclusive_aegisvoice_v2_gpu')):
        raise RuntimeError('Exclusive owner GPU window is not active')
    return dict(pid=parent['pid'], start_ticks=parent['start_ticks'], uid=parent['uid'])


def worker_guard():
    return guard_contract(process_identity(os.getpid()), process_identity(os.getppid()),
                          read(ROOT / 'operations/gpu-window.json'), read(ROOT / 'gpu-window-authorized.json'), ROOT)


def gpu_pids():
    values = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                     text=True, timeout=10)
    return [int(line.strip()) for line in values.splitlines() if line.strip()]


def gpu_ownership(pids, own_pid, loaded=False):
    if set(pids) != ({own_pid} if loaded else set()):
        raise RuntimeError('GPU ownership differs from ' + ('the exact own PID' if loaded else 'empty pre-load GPU'))
    return list(sorted(set(pids)))


def budget_guard(used, free, extra, available_ram):
    if used + extra > 40_000_000_000 or free < 5 * 1024**3 + extra or available_ram < 8 * 1024**3:
        raise RuntimeError('Operational disk/RAM resource budget; not model infeasibility')
    return dict(working_allocated_bytes=used, free_bytes=free, extra_bytes=extra, available_ram_bytes=available_ram)


def preflight():
    from training.v2_language_monitor import physical_usage
    git = Path(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip())
    roots = sorted(Path('/var/tmp').glob('aegis*')) + [Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'), REPO, Path('/home/phc/.cache'), git]
    measured = physical_usage(roots)
    if measured['permission_errors']:
        raise RuntimeError('Incomplete working inventory')
    ram = int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1]) * 1024
    result = budget_guard(measured['allocated_bytes'], shutil.disk_usage(ROOT).free, PAIR_EXTRA, ram)
    return dict(**result, inventory=measured, git_common=str(git), future_training_checkpoint_writes_reserved=0)


def output_usage(output):
    return sum(path.lstat().st_blocks * 512 for path in Path(output).rglob('*') if path.is_file() and not path.is_symlink())


def resource_guard(output, started):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    used = output_usage(output)
    if rss > RSS_CAP or used > OUTPUT_CAP or time.monotonic() - started > WALL_SECONDS:
        raise QualityAbort('Quality resource/time bound exceeded')
    return dict(rss_peak_bytes=rss, output_allocated_bytes=used)


def interrupt(*_):
    raise QualityAbort('Owned quality worker interrupted; preserve all7 outcomes')


def runtime_metadata(torch, backend):
    residents = {}
    for name in ('acoustic', 'vocoder'):
        model = getattr(backend, name)
        parameters = list(model.parameters())
        buffers = list(model.buffers())
        if not parameters or any(value.device.type != 'cuda' for value in parameters + buffers):
            raise RuntimeError('All parameters/buffers must reside on CUDA')
        if any(value.dtype != torch.float32 for value in parameters):
            raise RuntimeError('Acoustic/vocoder parameters must remain FP32')
        residents[name] = dict(parameter_count=sum(value.numel() for value in parameters), all_parameters_cuda=True,
                               all_buffers_cuda=True, parameters_dtype='torch.float32', training=bool(model.training))
        if model.training:
            raise RuntimeError('Quality models must be in eval mode')
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=dict(index=torch.cuda.current_device(), name=properties.name, total_memory=properties.total_memory,
                            major=properties.major, minor=properties.minor),
                flags=dict(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                           cudnn_deterministic=torch.backends.cudnn.deterministic,
                           deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                           deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                           float32_matmul_precision=torch.get_float32_matmul_precision()),
                bf16_supported=bool(torch.cuda.is_bf16_supported()), models=residents,
                precision=dict(acoustic_autocast='bf16', parameters='FP32', vocoder='FP32'),
                vram=dict(resident_allocated_bytes=torch.cuda.memory_allocated(), resident_reserved_bytes=torch.cuda.memory_reserved(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def finalize_rows(rows, cases, error=None):
    expected = [case['id'] for case in cases]
    if len(expected) != 7 or len(set(expected)) != 7 or any(row['case'] not in expected for row in rows) or len({row['case'] for row in rows}) != len(rows):
        raise ValueError('All7 unique expected cases required')
    observed = {row['case']: row for row in rows}
    return [observed.get(case_id, dict(case=case_id, outcome='not_attempted', error=error or 'not attempted')) for case_id in expected]


def validate_rows(rows, cases):
    if len(rows) != 7 or {row['case'] for row in rows} != {case['id'] for case in cases} or len({row['case'] for row in rows}) != 7:
        raise ValueError('All7 unique conditions required; never filter denominator')
    for row in rows:
        peak = row.get('peak')
        if row.get('outcome') != 'ok' or row.get('finite') is not True or row.get('outside_unit', 0) or row.get('clipped_samples', 0) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or not 0 <= peak <= 1:
            raise ValueError('All7 valid waveform conditions required')
        path = safe(row['wav'])
        with wave.open(str(path), 'rb') as wav:
            frames = wav.getnframes()
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (24000, 1, 2) or not 0 < frames <= MAX_AUDIO_SECONDS * 24000:
                raise ValueError('Invalid complete PCM WAV format/bound')
            if len(wav.readframes(frames)) != frames * 2:
                raise ValueError('Truncated PCM WAV payload')
        if sha(path) != row['wav_sha256']:
            raise ValueError('PCM WAV changed after generation')


def measure(backend, case, plan, output):
    # NumPy/soundfile are reached only inside the guarded CUDA worker.
    np = backend.np
    from training import v2_fastpitch_benchmark as common
    normal = common.frontend('refinement')(case['text'])
    effective = backend.fp.normalize(normal['tts_text'])
    if effective != plan['effective_texts'][case['id']]:
        raise QualityAbort('Frozen frontend changed')
    backend.torch.manual_seed(SEED)
    backend.synchronize()
    backend.reset_peak()
    tick = time.monotonic()
    raw = None
    path = output / (case['id'] + '.wav')
    row = dict(candidate='fastpitch-step%d-cuda-bf16-quality' % plan['expected_step'], case=case['id'],
               source_text=case['text'], effective_text=effective, precision='acoustic BF16; parameters/vocoder FP32',
               device='cuda', pace=1.0, seed=SEED, checkpoint_step=plan['expected_step'])
    try:
        raw = np.asarray(backend.generate(normal['tts_text']), dtype=np.float32).reshape(-1)
        backend.synchronize()
        row.update(native_generation_wall_seconds=time.monotonic() - tick,
                   finite=bool(np.isfinite(raw).all()), samples=int(raw.size),
                   native_float_sha256=hashlib.sha256(raw.astype('<f4', copy=False).tobytes()).hexdigest())
        finite = row['finite']
        row.update(peak=float(np.abs(raw).max()) if finite and raw.size else None,
                   outside_unit=int((np.abs(raw) > 1).sum()) if finite else None,
                   samples_at_generator_bound=int((np.abs(raw) >= 1).sum()) if finite else None)
        if not finite or not raw.size or raw.size > MAX_AUDIO_SECONDS * 24000 or raw.size % 256:
            raise ValueError('Empty/nonfinite/overlong/invalid native waveform; no truncation')
        if row['outside_unit']:
            raise ValueError('Raw overshoot; refuse gain or extra clamp')
        pcm = np.rint(raw * 32767).astype('<i2').tobytes()
        common.write_pcm(path, pcm)
        # Reuse the frozen metric implementation, expose only audio QC fields.
        qc = common.metrics_module().pcm_metrics(pcm, [dict(ms=0.0, bytes=len(pcm))])
        row.update({key: qc[key] for key in ('audio_seconds', 'peak_pcm16', 'clipped_samples', 'onset_audio_offset_ms', 'onset_rule')})
        row.update(mel_frames=int(raw.size // 256), wav=str(path), wav_sha256=sha(path),
                   pcm_sha256=hashlib.sha256(pcm).hexdigest())
        if row['clipped_samples'] or row['onset_audio_offset_ms'] is None:
            raise ValueError('Clipping or no digital speech onset')
        row['outcome'] = 'ok'
    except Exception as error:
        row.update(outcome='failed', error=repr(error), native_generation_wall_seconds=time.monotonic() - tick)
        if raw is not None and raw.size and raw.size <= MAX_AUDIO_SECONDS * 24000:
            # FLOAT diagnostics only for failed conditions; never expanded successes.
            if output_usage(output) + int(raw.nbytes) + 4096 < OUTPUT_CAP:
                import soundfile as sf
                diagnostic = output / (case['id'] + '-failed-float.wav')
                sf.write(diagnostic, raw, 24000, subtype='FLOAT')
                row.update(failed_float=str(diagnostic), failed_float_sha256=sha(diagnostic))
    row.update(backend.peaks())
    return row


def jobs_for(rows, plan):
    by_case = {row['case']: row for row in rows}
    return [dict(candidate=by_case[case['id']].get('candidate', 'fastpitch-step%d-cuda-bf16-quality' % plan['expected_step']),
                 case=case['id'], wav=by_case[case['id']].get('wav'),
                 input_audio_sha256=by_case[case['id']].get('wav_sha256'), expected_verbatim=case['text'],
                 expected_spoken=plan['spoken'][case['id']], expected_effective=plan['effective_texts'][case['id']],
                 expected_frontend=plan['effective_texts'][case['id']], cohort='validation-common7-cuda-bf16-pace1',
                 artifact='final-PCM16', generation_outcome=by_case[case['id']]['outcome'],
                 product_audio_valid=by_case[case['id']]['outcome'] == 'ok', frontend_mode='refinement')
            for case in plan['cases']]


def load_plan(args):
    path = safe(args.plan_file)
    if sha(path) != validate_sha(args.plan_sha256):
        raise ValueError('Sealed plan SHA differs')
    plan = read(path)
    validate_step(plan['expected_step'])
    if plan.get('schema') != 'aegis.fastpitch.gpu-quality.v1' or plan['root'] != str(ROOT):
        raise ValueError('Unrecognized quality plan')
    if plan['seed'] != SEED or plan['pace'] != 1.0 or plan['observations'] != 7 or plan['warmups'] != 0 or plan['per_case'] != 1:
        raise ValueError('Frozen inference/denominator differs')
    if plan['precision'] != dict(acoustic_autocast='bf16', parameters='FP32', vocoder='FP32') or tuple(case['id'] for case in plan['cases']) != CASE_IDS:
        raise ValueError('Frozen precision/validation cases differ')
    baseline = read(BASELINE)['plan']
    if (plan['cases'] != baseline['cases'] or plan['effective_texts'] != baseline['effective_texts']
            or plan['spoken'] != read(REPO / 'benchmarks/v2/spoken-validation.pt-BR.json')['cases']
            or plan['asr'] != baseline['asr']):
        raise ValueError('Fixed validation text/frontend/scorer differs')
    if (plan['device'] != 'cuda' or plan['max_audio_seconds'] != MAX_AUDIO_SECONDS or plan['max_tokens'] != MAX_TOKENS
            or plan['output_cap_bytes'] != OUTPUT_CAP or plan['rss_cap_bytes'] != RSS_CAP
            or plan['wall_seconds'] != WALL_SECONDS or plan['phase_budget']['additional_pair_upper_bytes'] != PAIR_EXTRA):
        raise ValueError('Frozen device/resource bounds differ')
    if plan['checkpoint'] != checkpoint_binding(plan['checkpoint']['path'], plan['expected_step'], plan['checkpoint']['sha256']):
        raise ValueError('Checkpoint binding differs from sealed plan')
    if plan['source_hashes'] != source_bindings(Path(plan['source']), Path(plan['bigvgan'])):
        raise ValueError('Complete frozen source pin set differs')
    output = safe(plan['output'], directory=True)
    if output != path.parent or not output.is_relative_to(PREPARATION):
        raise ValueError('Unexpected private preparation/output path')
    verify_bindings(plan)
    return plan, output


def execute(args):
    if not args.gpu_authorized:
        raise RuntimeError('Explicit --gpu-authorized required before any neural import')
    plan, output = load_plan(args)
    if (output / 'controller.json').exists() or (output / 'result.json').exists():
        raise RuntimeError('Partial/existing quality execution: preserve; use a fresh plan/output')
    owner = worker_guard()
    gpu_ownership(gpu_pids(), os.getpid(), loaded=False)
    budget = preflight()
    verify_bindings(plan)
    worker_guard()
    gpu_ownership(gpu_pids(), os.getpid(), loaded=False)
    write(output / 'controller.json', dict(status='exec_worker', utc=utc(), identity=process_identity(os.getpid()),
                                         owner=owner, preflight=budget, plan_sha256=args.plan_sha256,
                                         lifecycle='os.exec preserves PID/start and immediate GPU-worker parent; no detached child'))
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       CUDA_CACHE_PATH=str(PREPARATION / 'cuda-cache'), CUDA_CACHE_MAXSIZE=str(CACHE_CAP))
    command = [sys.executable, '-m', 'training.v2_fastpitch_gpu_quality', '--worker', '--gpu-authorized',
               '--plan-file', str(args.plan_file), '--plan-sha256', args.plan_sha256]
    os.execve(sys.executable, command, environment)


def worker(args):
    if not args.gpu_authorized:
        raise RuntimeError('Explicit GPU authorization required')
    plan, output = load_plan(args)
    controller = read(output / 'controller.json')
    identity = process_identity(os.getpid())
    if controller['plan_sha256'] != args.plan_sha256 or any(controller['identity'][key] != identity[key] for key in ('pid', 'ppid', 'uid', 'start_ticks')):
        raise RuntimeError('Controller exec identity changed')
    worker_guard()
    gpu_ownership(gpu_pids(), os.getpid(), loaded=False)
    started = time.monotonic()
    for name in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(name, interrupt)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    rows, metadata, error, backend, resources = [], {}, None, None, {}
    try:
        # All authorization, ownership, pins and budget gates above precede Torch.
        from training import v2_fastpitch_benchmark as common
        options = argparse.Namespace(checkpoint=Path(plan['checkpoint']['path']), checkpoint_sha256=plan['checkpoint']['sha256'],
                                     source=Path(plan['source']), bigvgan=Path(plan['bigvgan']), device='cuda', precision='bf16',
                                     threads=2, max_tokens=MAX_TOKENS, max_audio_seconds=MAX_AUDIO_SECONDS)
        tick = time.monotonic()
        backend = common.Backend(options)
        load_wall = time.monotonic() - tick
        if backend.metadata['step'] != plan['expected_step']:
            raise QualityAbort('Loaded checkpoint step differs')
        gpu_ownership(gpu_pids(), os.getpid(), loaded=True)
        metadata = dict(backend=backend.metadata, actual_before=runtime_metadata(backend.torch, backend),
                        native_load_wall_seconds=load_wall, load_vram=backend.peaks())
        if not metadata['actual_before']['bf16_supported']:
            raise QualityAbort('CUDA BF16 unsupported')
        if 'A2000' not in metadata['actual_before']['device']['name']:
            raise QualityAbort('This qualified pilot requires the actual A2000 device')
        verify_bindings(plan)
        write(output / 'manifest.json', dict(plan_sha256=args.plan_sha256, plan=plan, **metadata))
        for case in plan['cases']:
            worker_guard()
            gpu_ownership(gpu_pids(), os.getpid(), loaded=True)
            resources = resource_guard(output, started)
            rows.append(dict(case=case['id'], outcome='failed', error='Native case interrupted before completion'))
            row = measure(backend, case, plan, output)
            rows[-1] = row
            write(output / 'results.json', rows)
            event(output, dict(case=case['id'], outcome=row['outcome']))
            gpu_ownership(gpu_pids(), os.getpid(), loaded=True)
            resource_guard(output, started)
        validate_rows(rows, plan['cases'])
        verify_bindings(plan)
        worker_guard()
        metadata['actual_after'] = runtime_metadata(backend.torch, backend)
        if metadata['actual_after']['flags'] != metadata['actual_before']['flags']:
            raise QualityAbort('Backend numerical flags changed within run')
        gpu_ownership(gpu_pids(), os.getpid(), loaded=True)
    except BaseException as caught:
        error = repr(caught)[:4096]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            verify_bindings(plan)
            metadata['source_checkpoint_postcheck'] = 'pass'
        except BaseException as closing_error:
            metadata['source_checkpoint_postcheck'] = repr(closing_error)[:4096]
            error = error or repr(closing_error)[:4096]
        if backend is not None and 'actual_after' not in metadata:
            try:
                metadata['actual_after'] = runtime_metadata(backend.torch, backend)
                metadata['gpu_pids_before_exit'] = gpu_ownership(gpu_pids(), os.getpid(), loaded=True)
            except BaseException as closing_error:
                metadata['closing_metadata_error'] = repr(closing_error)[:4096]
                error = error or repr(closing_error)[:4096]
        rows = finalize_rows(rows, plan['cases'], error)
        write(output / 'results.json', rows)
        valid = error is None and all(row['outcome'] == 'ok' for row in rows)
        write(output / 'asr-jobs.json', dict(jobs=jobs_for(rows, plan), expected_observations=7, asr_ready=valid,
                                          generation_manifest=str(output / 'manifest.json'), plan_sha256=args.plan_sha256,
                                          scorer_sha256=PINS['benchmarks/v2/asr_score.py'],
                                          policy='Parent only, after both CUDA process exits; no scoring filtered successful subsets'))
        write(output / 'manifest.json', dict(plan_sha256=args.plan_sha256, plan=plan, **metadata))
        result = dict(status='all7_valid_process_exit_pending' if valid else 'failed_preserve_denominator7', utc=utc(),
                      returncode=0 if valid else 1, expected_observations=7, attempted=len([row for row in rows if row['outcome'] != 'not_attempted']),
                      successes=sum(row['outcome'] == 'ok' for row in rows), failures=sum(row['outcome'] != 'ok' for row in rows),
                      error=error, resources=resources, native_worker_wall_seconds=time.monotonic() - started,
                      identity=identity, cuda_process_exit='Parent GPU-worker done receipt and subsequent GPU-empty audit required',
                      asr_executed=False, final_data_opened=False, product_latency_claim=False)
        write(output / 'result.json', result)
    return result['returncode']


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    action = value.add_mutually_exclusive_group()
    action.add_argument('--plan', dest='action', action='store_const', const='plan')
    action.add_argument('--execute', dest='action', action='store_const', const='execute')
    action.add_argument('--worker', dest='action', action='store_const', const='worker', help=argparse.SUPPRESS)
    value.set_defaults(action='plan')
    value.add_argument('--expected-step', type=int)
    value.add_argument('--checkpoint', type=Path)
    value.add_argument('--checkpoint-sha256')
    value.add_argument('--output', type=Path)
    value.add_argument('--source', type=Path, default=ROOT / 'fastpitch/source')
    value.add_argument('--bigvgan', type=Path, default=ROOT / 'bigvgan')
    value.add_argument('--plan-file', type=Path)
    value.add_argument('--plan-sha256')
    value.add_argument('--gpu-authorized', action='store_true')
    return value


def main(argv=None):
    value = parser()
    args = value.parse_args(argv)
    os.umask(0o077)
    if args.action == 'plan':
        if any(getattr(args, key) is None for key in ('expected_step', 'checkpoint', 'checkpoint_sha256', 'output')):
            value.error('Planning requires --expected-step, --checkpoint, --checkpoint-sha256 and --output')
        args.output = args.output.absolute()
        if not args.output.is_relative_to(PREPARATION) or args.output.resolve() != args.output:
            value.error('Private output must be canonical and under ' + str(PREPARATION))
        plan = planned(args)
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        write(args.output / 'plan.json', plan)
        print(json.dumps(dict(status='prepared_only', plan=str(args.output / 'plan.json'), sha256=sha(args.output / 'plan.json'), torch_imported='torch' in sys.modules)))
        return 0
    if not args.plan_file or not args.plan_sha256:
        value.error('Execution requires --plan-file and --plan-sha256')
    if args.action == 'execute':
        execute(args)
        return 1  # os.execve never returns successfully.
    return worker(args)


if __name__ == '__main__':
    raise SystemExit(main())
