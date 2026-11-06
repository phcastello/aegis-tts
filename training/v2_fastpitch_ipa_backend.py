"""Separate FastPitch native161 IPA + BigVGAN complete-file research backend.

Default CLI is metadata-only. Native eSpeak runs in an owned CPU subprocess;
the process that imports Torch never initializes G2P. No service, streaming,
training-data dependency, offload, text truncation, hidden retry or gain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
import unicodedata
import wave

REPO = Path(__file__).resolve().parents[1]
BIG_SOURCE_REV = '7d2b454564a6c7d014227f635b7423881f14bdac'
BIG_MODEL_REV = 'c329ede9e9bbc100ddf5c91e2330a61921262370'
BIG_WEIGHT_SHA = '6f9c5715550c9d0f11159ceb8935638da5aeb19e27d1e63677632df095e376f5'
BIG_CONFIG_SHA = 'd77e2c96583ca2296ac112a56ec7cc6bd5da4bf7681ceff18448bedc4fcf6512'
RAW_TRAINER_SHA = '7e7c94d8106b2e7729df5e238195f554dbcb5a494a3c5ee179f7ef8ee1a90709'
INHERITED_BACKEND_SHA = '0e108b2dacceb924019a6419dfe9cd89e65457700f123a230a8956d55538d17e'
ALIGNMENT_POLICY = 'FP32 cumulative duration boundaries and pitch sums; acoustic outputs retain selected dtype'
RAW_VOCAB = ['<PAD>', '<UNK>'] + list(" !'(),-.:;?abcdefghijklmnopqrstuvwxyzáàâãäéèêëíìîïóòôõöúùûüçñ0123456789%/ºª+")
NEURAL_LOCK = threading.RLock()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def front_module():
    from training import v2_fastpitch_ipa_frontend
    return v2_fastpitch_ipa_frontend


def normalize_for_frontend(text, mode='refinement'):
    front_module().normalize_input(text)  # Reject NUL/surrogates before any rewriting.
    if mode == 'refinement':
        from training.f5_refine_text import normalize_text
        receipt = normalize_text(text)
    elif mode == 'native':
        receipt = dict(tts_text=text, applied_rules=[], warnings=[])
    else:
        raise ValueError('unknown text normalization mode')
    effective = unicodedata.normalize('NFC', receipt['tts_text']).lower()
    effective = effective.translate(str.maketrans({
        '“': "'", '”': "'", '‘': "'", '’': "'", '"': "'", '–': '-',
        '—': '-', '−': '-', '·': '.', '…': '...'}))
    effective = ' '.join(effective.split())
    front_module().normalize_input(effective)
    return dict(source_text=text, effective_text=effective, mode=mode,
                applied_rules=receipt.get('applied_rules', []),
                warnings=receipt.get('warnings', []))


def validate_ids(ids, max_tokens):
    front_module().validate_ids(ids)
    if len(ids) > max_tokens:
        raise ValueError('phoneme sequence exceeds token bound; no truncation')
    return ids


class FrontendError(RuntimeError):
    pass


class FrontendClient:
    """One sequential JSON-lines CPU child, bounded pipes and explicit cleanup.

    Reader threads always drain both pipes. Queue/line/stderr bounds are hard
    protocol failures; only this Popen child is signalled. No restart or retry.
    """
    def __init__(self, *, command=None, metadata=None, timeout=20., startup_timeout=30.,
                 max_tokens=2048, max_line_bytes=1_000_000,
                 max_text_bytes=262144, max_stderr_bytes=65536):
        if not all(math.isfinite(x) and x > 0 for x in (timeout, startup_timeout)):
            raise ValueError('positive finite RPC timeouts required')
        if any(type(x) is not int or x < 1 for x in
               (max_tokens, max_line_bytes, max_text_bytes, max_stderr_bytes)):
            raise ValueError('positive integer pipe/token bounds required')
        self.timeout, self.max_tokens = timeout, max_tokens
        self.max_line_bytes, self.max_text_bytes = max_line_bytes, max_text_bytes
        self.max_stderr_bytes = max_stderr_bytes
        self.expected = metadata if metadata is not None else front_module().frontend_metadata()
        self.lock = threading.RLock(); self.state_lock = threading.Lock()
        self.lines = queue.Queue(maxsize=2)
        self.failure = None; self.closed = False; self.counter = 0
        self.stderr_tail = bytearray(); self.stderr_bytes = 0
        self.last_request = None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONNOUSERSITE='1', PYTHONPATH=str(REPO))
        cmd = command if command is not None else [
            '/usr/bin/python3', '-S', '-B', '-u', '-m',
            'training.v2_fastpitch_ipa_frontend', '--serve']
        self.command = list(cmd)
        self.process = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        bufsize=0)
        os.set_blocking(self.process.stdin.fileno(), False)
        self.readers = [threading.Thread(target=self._stdout, daemon=True),
                        threading.Thread(target=self._stderr, daemon=True)]
        for thread in self.readers:
            thread.start()
        started = time.perf_counter()
        try:
            ready = self._read(startup_timeout)
            self._validate_ready(ready)
            self.ready = ready
            self.startup_seconds = time.perf_counter() - started
        except Exception:
            self.close()
            raise

    def _fail(self, message):
        with self.state_lock:
            if self.failure is None:
                self.failure = message
        try:
            self.lines.put_nowait(('error', message))
        except queue.Full:
            pass
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass

    def _stdout(self):
        pending = bytearray()
        try:
            while True:
                block = self.process.stdout.read(4096)
                if not block:
                    if pending:
                        self._fail('frontend EOF inside JSON line')
                    else:
                        try:
                            self.lines.put_nowait(('eof', None))
                        except queue.Full:
                            self._fail('frontend unsolicited output at EOF')
                    return
                pending.extend(block)
                while b'\n' in pending:
                    end = pending.index(10)
                    if end > self.max_line_bytes:
                        raise FrontendError('frontend stdout line exceeds bound')
                    line = bytes(pending[:end]); del pending[:end + 1]
                    try:
                        self.lines.put_nowait(('line', line))
                    except queue.Full:
                        raise FrontendError('frontend stdout queue exceeds bound')
                if len(pending) > self.max_line_bytes:
                    raise FrontendError('frontend stdout line exceeds bound')
        except (OSError, ValueError, FrontendError) as error:
            if not self.closed:
                self._fail(str(error))

    def _stderr(self):
        try:
            while True:
                block = self.process.stderr.read(4096)
                if not block:
                    return
                with self.state_lock:
                    self.stderr_bytes += len(block)
                    self.stderr_tail.extend(block)
                    del self.stderr_tail[:-self.max_stderr_bytes]
                    exceeded = self.stderr_bytes > self.max_stderr_bytes
                if exceeded:
                    self._fail('frontend stderr exceeds bound')
                    return
        except (OSError, ValueError) as error:
            if not self.closed:
                self._fail(str(error))

    def _read(self, timeout):
        if self.failure is not None:
            raise FrontendError(self.failure)
        try:
            kind, value = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise FrontendError('frontend response timeout') from None
        if self.failure is not None:
            raise FrontendError(self.failure)
        if kind != 'line':
            raise FrontendError('frontend EOF' if kind == 'eof' else value)
        try:
            result = json.loads(value.decode('utf-8'))
        except (ValueError, UnicodeError) as error:
            raise FrontendError('invalid frontend JSON: ' + str(error)) from error
        if not isinstance(result, dict):
            raise FrontendError('frontend record must be an object')
        return result

    def _validate_ready(self, ready):
        md = ready.get('frontend', {})
        if ready.get('status') != 'ready' or ready.get('vocab_sha256') != self.expected['vocab_sha256']:
            raise FrontendError('frontend READY/vocabulary mismatch')
        for key in ('schema', 'version', 'symbols', 'vocab_sha256', 'map_sha256',
                    'config_sha256', 'source_pins', 'fingerprint_sha256'):
            if md.get(key) != self.expected[key]:
                raise FrontendError('frontend READY pin mismatch: ' + key)
        if md.get('torch_imported') is not False or md.get('native_initialized') is not True:
            raise FrontendError('frontend must initialize native G2P in CPU child without Torch')

    def _write(self, payload, deadline):
        view = memoryview(payload)
        while view:
            if self.failure is not None or self.process.poll() is not None:
                raise FrontendError(self.failure or 'frontend exited before request')
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                raise FrontendError('frontend request write timeout')
            try:
                count = os.write(self.process.stdin.fileno(), view)
            except BlockingIOError:
                continue
            if count <= 0:
                raise FrontendError('frontend request pipe closed')
            view = view[count:]

    def phonemize(self, text):
        with self.lock:
            if self.closed:
                raise FrontendError('frontend client is closed; no automatic restart')
            started = time.perf_counter()
            try:
                expected_text = front_module().normalize_input(text)
                if len(text.encode('utf-8')) > self.max_text_bytes:
                    raise FrontendError('frontend text exceeds byte bound')
                self.counter += 1
                request_id = 'fastpitch-ipa-' + str(self.counter)
                payload = (json.dumps(dict(id=request_id, text=text), ensure_ascii=False,
                                      allow_nan=False) + '\n').encode('utf-8')
                deadline = started + self.timeout
                self._write(payload, deadline)
                response = self._read(max(0., deadline - time.perf_counter()))
                if response.get('id') != request_id:
                    raise FrontendError('frontend response ID mismatch')
                if response.get('ok') is not True:
                    raise FrontendError('frontend rejected text: ' + str(response.get('error')))
                front = front_module()
                for key in ('map_sha256', 'config_sha256', 'vocab_sha256'):
                    if response.get(key) != self.expected[key]:
                        raise FrontendError('frontend response pin mismatch: ' + key)
                if response.get('frontend_version') != self.expected['version']:
                    raise FrontendError('frontend version mismatch')
                if response.get('normalized_text') != expected_text:
                    raise FrontendError('frontend normalized text mismatch')
                sentences, ids = front.encode_sentences(response.get('sentences'))
                if response['sentences'] != sentences or response.get('ids') != ids:
                    raise FrontendError('frontend phoneme/ID consistency mismatch')
                validate_ids(response['ids'], self.max_tokens)
                self.last_request = dict(response, rpc_seconds=time.perf_counter() - started)
                return self.last_request
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=1.)
            for thread in self.readers:
                if thread is not threading.current_thread():
                    thread.join(timeout=1.)
            for stream in (self.process.stdout, self.process.stderr):
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def load_bound(path, expected, torch, weights_only=False):
    """Inherited exact-open-object hash/load; reject concurrent replacement."""
    path = Path(path)
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno()); digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError('artifact SHA256 mismatch: ' + str(path))
        stream.seek(0)
        state = torch.load(stream, map_location='cpu', weights_only=weights_only)
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('artifact replaced while loading; use frozen weights')
    return state


def validate_pair(config, h, fp):
    front = front_module(); md = front.frontend_metadata()
    features, original, binding = config['features'], config['raw_source_features'], config['frontend']
    if features != dict(original, vocab=list(front.SYMBOLS)) or original['vocab'] != RAW_VOCAB:
        raise ValueError('IPA features must preserve original78 statistics with exact161 vocabulary')
    if features['vocab'] != fp.VOCAB or features['mel'] != fp.MEL_CONFIG:
        raise ValueError('checkpoint IPA vocabulary/mel representation mismatch')
    if (config['model']['n_symbols'], config['model']['n_mel_channels']) != (161, 100):
        raise ValueError('checkpoint model must use161 symbols and100 mel channels')
    if config.get('alignment_precision') != ALIGNMENT_POLICY:
        raise ValueError('checkpoint alignment policy mismatch')
    if binding.get('schema') != front.SCHEMA:
        raise ValueError('checkpoint IPA frontend schema mismatch')
    for key in ('phone_manifest_sha256', 'symbols_sha256', 'map_sha256',
                'frontend_code_sha256', 'source_features_sha256',
                'raw_grapheme_trainer_sha256', 'phone_loader_code_sha256'):
        value = binding.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('missing/malformed checkpoint provenance hash: ' + key)
    if (binding['symbols_sha256'] != front.VOCAB_SHA256 or binding['map_sha256'] != front.MAP_SHA256
            or binding['frontend_code_sha256'] != sha(front.__file__)
            or binding['raw_grapheme_trainer_sha256'] != RAW_TRAINER_SHA
            or sha(REPO / 'training/fastpitch_experiment.py') != RAW_TRAINER_SHA):
        raise ValueError('checkpoint frontend/source binding changed')
    # Training-feature and phone-manifest paths are provenance, not inference IO.
    if not all(isinstance(binding.get(key), str) and binding[key] for key in
               ('phone_manifest_path', 'source_features_path')):
        raise ValueError('checkpoint training-source provenance paths missing')
    expected = dict(sampling_rate=24000, n_fft=1024, hop_size=256, win_size=1024,
                    num_mels=100, fmin=0, fmax=None)
    if any(h.get(k) != v for k, v in expected.items()) or math.prod(h['upsample_rates']) != 256:
        raise ValueError('BigVGAN acoustic representation mismatch')
    if not math.isfinite(features['pitch_mean']) or not math.isfinite(features['pitch_std']) or features['pitch_std'] <= 0:
        raise ValueError('invalid pitch normalization')
    return md


@contextmanager
def bounded_regulator(module, torch, max_frames):
    original = module.regulate_len
    def bounded(durations, encoded, pace=1., mel_max_len=None):
        if not math.isfinite(pace) or pace <= 0 or not bool(torch.isfinite(durations).all()) or bool((durations < 0).any()):
            raise ValueError('invalid predicted duration or pace')
        reps = (durations.float() / pace + .5).long()
        if bool((reps < 0).any()) or bool((reps > max_frames).any()):
            raise ValueError('predicted duration exceeds bound; no truncation')
        frames = reps.sum(dim=1)
        if bool((frames <= 0).any()):
            raise ValueError('zero predicted duration; acoustic model is not ready')
        if bool((frames > max_frames).any()):
            raise ValueError('predicted duration exceeds bound; no truncation')
        return original(durations, encoded, pace, mel_max_len)
    module.regulate_len = bounded
    try:
        yield
    finally:
        module.regulate_len = original


class Backend:
    def __init__(self, args):
        if args.device == 'cuda' and not args.gpu_authorized:
            raise ValueError('CUDA requires separate actual exclusive worker authorization')
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if not math.isfinite(args.max_audio_seconds) or args.max_audio_seconds <= 0 or args.max_tokens < 3:
            raise ValueError('positive audio/token bounds required')
        import numpy as np
        import torch
        from training import v2_fastpitch_ipa_experiment as fp
        self.torch, self.np, self.device = torch, np, args.device
        self.max_frames = int(args.max_audio_seconds * 24000 / 256)
        if self.max_frames < 1:
            raise ValueError('audio bound is shorter than one acoustic frame')
        self.max_tokens, self.precision = args.max_tokens, args.precision
        self.frontend_mode = getattr(args, 'frontend', 'refinement')
        self.client = None; self.last_request = None
        torch.set_num_threads(args.threads)
        if args.device == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 is unsupported; no silent precision fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        checkpoint = load_bound(args.checkpoint, args.checkpoint_sha256, torch)
        config = checkpoint['config']; h = json.loads((args.bigvgan / 'config.json').read_text())
        self.frontend_metadata = validate_pair(config, h, fp)
        if sha(args.bigvgan / 'config.json') != BIG_CONFIG_SHA:
            raise ValueError('BigVGAN config pin mismatch')
        if json.loads((args.bigvgan / 'revision.json').read_text())['revision'] != BIG_MODEL_REV:
            raise ValueError('BigVGAN model revision mismatch')
        source = args.bigvgan / 'source'
        if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != BIG_SOURCE_REV:
            raise ValueError('BigVGAN source revision mismatch')
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError('BigVGAN source has modifications')
        cls, _, _, _ = fp.imports(args.source)
        import fastpitch.model as model_module
        self.model_module, self.fp = model_module, fp
        self.acoustic = cls(**config['model'])
        self.acoustic.load_state_dict(checkpoint['model'], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in self.acoustic.parameters()):
            raise ValueError('nonfinite acoustic weights')
        for name in ('pitch_mean', 'pitch_std'):
            if not math.isclose(float(getattr(self.acoustic, name)[0]), config['features'][name], rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError('pitch buffer differs from feature normalization')
        self.metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=args.checkpoint_sha256,
            step=checkpoint['step'], epoch=checkpoint['epoch'], checkpoint_config=config,
            checkpoint_provenance=checkpoint.get('provenance'), loader_code_sha256=sha(fp.__file__),
            fastpitch_source_pin=fp.UPSTREAM, fastpitch_source_manifest_sha256=sha(args.source / 'UPSTREAM.json'),
            bigvgan_source_revision=BIG_SOURCE_REV, bigvgan_model_revision=BIG_MODEL_REV,
            bigvgan_weight_sha256=BIG_WEIGHT_SHA, bigvgan_config_sha256=BIG_CONFIG_SHA,
            vocoder_training_normalize_volume=h.get('normalize_volume'), runtime_waveform_normalization='none',
            native_model_duration_policy='Pinned FastPitch infer uses (exp(logdur)-1).clamp(0,75); no added minimum, retry or total-length truncation',
            native_vocoder_bound='Pinned BigVGAN clamps [-1,1] internally; final PCM endpoints are measured and rejected',
            training_data_or_phone_cache_required_for_inference=False, frontend=self.frontend_metadata,
            precision=dict(acoustic_autocast=self.precision, parameters='FP32', floating_buffers='FP32',
                           vocoder='FP32', vocoder_autocast='explicitly disabled including caller outer context', tf32=False),
            scope='local complete-file backend; CPU only is diagnostic, not A2000 evidence')
        del checkpoint
        sys.path.insert(0, str(source))
        from bigvgan import BigVGAN
        from env import AttrDict
        self.vocoder = BigVGAN(AttrDict(h), use_cuda_kernel=False)
        state = load_bound(args.bigvgan / 'bigvgan_generator.pt', BIG_WEIGHT_SHA, torch, weights_only=True)
        self.vocoder.load_state_dict(state['generator'], strict=True); del state
        self.vocoder.remove_weight_norm()
        self.acoustic.to(args.device).eval(); self.vocoder.to(args.device).eval()
        self._assert_resident()
        if not all(bool(torch.isfinite(p).all()) for p in self.vocoder.parameters()):
            raise ValueError('nonfinite vocoder weights')
        for model in (self.acoustic, self.vocoder):
            if any((b.is_floating_point() or b.is_complex()) and not bool(torch.isfinite(b).all()) for b in model.buffers()):
                raise ValueError('nonfinite model buffer')
        self.synchronize(); self.metadata.update(load_vram=self.peaks())

    def _assert_resident(self):
        for model in (self.acoustic, self.vocoder):
            if any(p.device.type != self.device for p in model.parameters()) or any(b.device.type != self.device for b in model.buffers()):
                raise RuntimeError('all acoustic/vocoder parameters and buffers must remain on selected device')
            for value in (*model.parameters(), *model.buffers()):
                if (value.is_floating_point() or value.is_complex()) and value.dtype != self.torch.float32:
                    raise RuntimeError('acoustic/vocoder floating parameters and buffers must remain FP32')

    def synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def reset_peak(self):
        if self.device == 'cuda':
            self.torch.cuda.reset_peak_memory_stats()

    def peaks(self):
        if self.device != 'cuda':
            return dict(peak_allocated_mib=None, peak_reserved_mib=None)
        return dict(peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=self.torch.cuda.max_memory_reserved()/2**20)

    def generate_ids(self, ids, pace=1., seed=40733133):
        validate_ids(ids, self.max_tokens)
        if not math.isfinite(pace) or pace <= 0 or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('positive finite pace and integer seed required')
        t = self.torch
        with NEURAL_LOCK:
            self._assert_resident()
            devices = [t.cuda.current_device()] if self.device == 'cuda' else []
            with t.random.fork_rng(devices=devices), t.inference_mode(), bounded_regulator(self.model_module, t, self.max_frames):
                t.manual_seed(seed)
                seq = t.tensor(ids, dtype=t.long, device=self.device)[None]
                with t.autocast(self.device, dtype=t.bfloat16, enabled=self.precision == 'bf16'):
                    mel, lens, *_ = self.acoustic.infer(seq, pace=pace)
                frames = int(lens[0])
                if not 0 < frames <= self.max_frames or not bool(t.isfinite(mel).all()):
                    raise ValueError('invalid acoustic mel/length; no clamp or retry')
                # Explicitly override a caller's outer autocast too: FP32 input
                # alone does not force Linear/Conv computations to stay FP32.
                with t.autocast(self.device, enabled=False):
                    audio = self.vocoder(mel[:, :, :frames].float()).squeeze().float().cpu().numpy()
            if audio.ndim != 1 or audio.size != frames * 256 or not self.np.isfinite(audio).all():
                raise ValueError('invalid vocoder output or frame/hop length')
            return audio

    def generate(self, text, pace=1., seed=40733133, frontend=None):
        with NEURAL_LOCK:
            started = time.perf_counter()
            normalized = normalize_for_frontend(text, frontend or self.frontend_mode)
            if self.client is None:
                self.client = FrontendClient(metadata=self.frontend_metadata, max_tokens=self.max_tokens)
            response = self.client.phonemize(normalized['effective_text'])
            self.last_request = dict(normalized, phonemizer=response, requested_seed=seed, pace=pace,
                                     preprocessing_seconds=time.perf_counter()-started,
                                     daemon_startup_seconds=self.client.startup_seconds)
            return self.generate_ids(response['ids'], pace, seed), self.last_request

    def close(self):
        if self.client is not None:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def metrics_module():
    spec = importlib.util.spec_from_file_location('aegis_ipa_file_metrics', REPO / 'benchmarks/v2/benchmark.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--text')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path, default=Path('/var/tmp/aegis-v2-work/fastpitch/source'))
    parser.add_argument('--bigvgan', type=Path, default=Path('/var/tmp/aegis-v2-work/bigvgan'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--frontend', choices=('native', 'refinement'), default='refinement')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--max-audio-seconds', type=float, default=60.)
    parser.add_argument('--pace', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=40733133)
    args = parser.parse_args(argv)
    plan = dict(status='METADATA_ONLY_NOT_EXECUTED', code_sha256=sha(__file__),
                frontend=front_module().frontend_metadata(), native_rate=24000,
                final_format='PCM16LE mono24000Hz', normalize=args.frontend,
                scope='whole-file local backend; no HTTP/streaming/TTFA or neural qualification',
                inherited_loader_source_sha256=INHERITED_BACKEND_SHA,
                checkpoint_or_audio_payload_opened=False, torch_imported='torch' in sys.modules)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, allow_nan=False)); return 0
    if any(getattr(args, key) is None for key in ('checkpoint', 'checkpoint_sha256', 'text', 'output')):
        parser.error('--execute requires checkpoint/SHA, text and fresh output.wav')
    if len(args.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.checkpoint_sha256):
        parser.error('exact lowercase checkpoint SHA256 required')
    if args.output.suffix.lower() != '.wav' or args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('fresh WAV and adjacent JSON paths required')
    if any((p / '.git').exists() for p in (args.output.resolve().parent, *args.output.resolve().parents)):
        parser.error('private WAV/receipt must be outside Git')
    if args.device == 'cuda':
        if not args.gpu_authorized:
            parser.error('CUDA requires actual separate supervised exclusive admission and --gpu-authorized')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('another GPU compute process is active')
    if args.threads < 1 or args.max_tokens < 3 or not math.isfinite(args.max_audio_seconds) or args.max_audio_seconds <= 0:
        parser.error('positive threads/token/audio bounds required')
    os.umask(0o077); args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    backend = None; started = time.perf_counter()
    result = dict(plan, status='EXECUTED', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  argv=sys.argv if argv is None else argv, device=args.device, requested_seed=args.seed,
                  gain=1., trim=False, limiter=False, streaming=False, timing='process-local complete file; includes load separately and required CPU frontend/PCM/WAV close; excludes interpreter startup')
    result.pop('checkpoint_or_audio_payload_opened')
    result.pop('torch_imported')
    result['payload_io_scope'] = 'Execution branch loads hash-bound checkpoints and may write the requested diagnostic WAV; inspect outcome/error and completed stages'
    try:
        backend = Backend(args); result['load_seconds'] = time.perf_counter()-started
        backend.reset_peak(); synthesis_started = time.perf_counter()
        audio, request = backend.generate(args.text, args.pace, args.seed)
        backend.synchronize(); result.update(frontend_request=request, backend=backend.metadata)
        result['neural_output'] = dict(samples=len(audio), finite=bool(backend.np.isfinite(audio).all()),
            peak=float(backend.np.abs(audio).max()) if audio.size else None,
            samples_at_or_outside_unit=int((backend.np.abs(audio) >= 1).sum()))
        from training.prepare_dataset import encode_pcm16
        pcm = encode_pcm16(audio)
        with args.output.open('xb') as stream:
            with wave.open(stream, 'wb') as output:
                output.setnchannels(1); output.setsampwidth(2); output.setframerate(24000); output.writeframes(pcm)
        available = time.perf_counter(); elapsed_ms = (available-synthesis_started)*1000
        measured = metrics_module().pcm_metrics(pcm, [dict(ms=elapsed_ms, bytes=len(pcm))])
        result.update(measured, synthesis_to_complete_file_ms=elapsed_ms,
                      process_timer_to_complete_file_ms=(available-started)*1000,
                      rtf=(available-synthesis_started)/(len(audio)/24000), wav=str(args.output),
                      wav_sha256=sha(args.output), pcm_sha256=hashlib.sha256(pcm).hexdigest(), **backend.peaks())
        with wave.open(str(args.output), 'rb') as output:
            if (output.getframerate(), output.getnchannels(), output.getsampwidth(), output.getnframes()) != (24000, 1, 2, len(audio)) or args.output.stat().st_size != 44+len(pcm):
                raise ValueError('incomplete PCM WAV')
        if measured['first_speech_pcm_available_ms'] is None or measured['clipped_samples']:
            raise ValueError('missing onset or measured clipped PCM; diagnostic retained without correction')
        result.update(outcome='ok', product_audio_valid=True,
                      first_speech_scope='file availability plus shared onset playback rule; no gateway/HTTP TTFA')
    except Exception as error:
        result.update(outcome='failed', product_audio_valid=False, error=repr(error))
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as error:
                result.update(outcome='failed', product_audio_valid=False, cleanup_error=repr(error))
        result['torch_imported'] = 'torch' in sys.modules
        result['operation_wall_seconds'] = time.perf_counter()-started
        json_atomic(args.output.with_suffix('.json'), result)
    print(json.dumps({k:result.get(k) for k in ('outcome', 'error', 'wav', 'synthesis_to_complete_file_ms')}, ensure_ascii=False))
    return 0 if result['outcome'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
