#!/usr/bin/env python3
"""Pinned F5TTS_Base/Vocos probe. Default is a dependency-free, offline plan.

No service control, training, ASR, downloads, loudness normalization or trimming.
CPU execution hides CUDA. CUDA execution requires an approved exclusive window.
Full utterances are generated before any PCM is available: no neural streaming.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unicodedata

try:
    from .benchmark import (gpu_snapshot, json_bytes, load_json, pcm_metrics,
                            private_directory, validate_window, write_private)
    from .voxcpm_runtime import file_hash, require_empty_gpu, require_own_gpu
except ImportError:
    from benchmark import (gpu_snapshot, json_bytes, load_json, pcm_metrics,
                           private_directory, validate_window, write_private)
    from voxcpm_runtime import file_hash, require_empty_gpu, require_own_gpu

SOURCE_COMMIT = "45868abaabd74aabf03cd673d6dd00cc9d9c01ae"
CHECKPOINT_SHA256 = "c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f"
MODEL_REVISION = "0af67bc061699cab46218c150948910fde209614"
RATE, HOP = 24000, 256
LEGACY_MEL_KEYS = {"mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"}
ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
            attn_backend="torch", attn_mask_enabled=False, checkpoint_activations=False)


def normalize_text(text, mode):
    text = unicodedata.normalize("NFC", text).strip()
    return text.lower() if mode == "lowercase" else text


def duration_frames(reference_frames, reference_text, generated_text, max_seconds):
    """Same UTF-8 length heuristic as upstream, with an explicit fail-fast cap."""
    if not reference_text.strip() or not generated_text.strip():
        raise ValueError("both reference and generated text are required")
    frames = reference_frames + int(reference_frames / len(reference_text.encode()) * len(generated_text.encode()))
    if frames * HOP / RATE > max_seconds:
        raise ValueError("duration heuristic exceeds max-seconds; do not truncate speech silently")
    return frames


def ema_state(checkpoint):
    """Accept uniform raw or EMA states; documented buffer adaptation only."""
    state, ignored = {}, []
    keys = set(checkpoint) - {"initted", "step"}
    prefixed = [key.startswith("ema_model.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed raw and EMA checkpoint keys")
    for key, value in checkpoint.items():
        if key in {"initted", "step"}:
            ignored.append(key)
            continue
        mapped = key.removeprefix("ema_model.")
        if mapped in LEGACY_MEL_KEYS:
            ignored.append(key)
        else:
            state[mapped] = value
    return state, ignored


def validate_files(args):
    source = Path(args.source_dir).resolve(strict=True)
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout.strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError("F5 source must be clean and match pinned commit")
    receipt = load_json(args.receipt)
    required = {"vocab": Path(args.vocab),
                "vocoder_config": Path(args.vocoder_dir) / "config.yaml",
                "vocoder_weights": Path(args.vocoder_dir) / "pytorch_model.bin"}
    hashes = {}
    for name, path in required.items():
        expected = receipt.get(name + "_sha256", "")
        if len(expected) != 64 or file_hash(path) != expected:
            raise ValueError("missing or mismatched verified SHA256: " + name)
        hashes[name] = expected
    if args.action == "infer":
        if not args.checkpoint or not str(args.checkpoint).endswith(".safetensors"):
            raise ValueError("inference requires the pinned safetensors checkpoint")
        if file_hash(args.checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("F5 checkpoint SHA256 mismatch")
        hashes["checkpoint"] = CHECKPOINT_SHA256
    reference = Path(args.reference_wav).resolve(strict=True)
    transcript_path = Path(args.reference_text_file).resolve(strict=True)
    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if not transcript or len(transcript) > 8192:
        raise ValueError("correct, explicit reference transcription is required; ASR is disabled")
    return source, reference, transcript, {
        "source_commit": commit, "model_revision": MODEL_REVISION, "hashes": hashes,
        "reference_sha256": file_hash(reference),
        "reference_transcript_sha256": file_hash(transcript_path),
        "vocoder_provenance": receipt.get("vocoder_revision"),
    }


class Guard:
    def __init__(self, args):
        self.args, self.loaded = args, False
        self.stop = threading.Event()

    def __call__(self):
        if self.args.device == "cuda":
            validate_window(load_json(self.args.window_file), self.args.candidate)
            if self.loaded:
                require_own_gpu(gpu_snapshot())
            else:
                require_empty_gpu(gpu_snapshot())

    def start_watchdog(self):
        if self.args.device != "cuda":
            return
        def watch():
            while not self.stop.wait(.5):
                try:
                    validate_window(load_json(self.args.window_file), self.args.candidate)
                except Exception:
                    os._exit(124)  # unload candidate on deadline; never operates production
        threading.Thread(target=watch, daemon=True).start()


def save_wave(output, name, values, sf, np):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("empty/nonfinite audio")
    buffer = io.BytesIO()
    sf.write(buffer, values, RATE, format="WAV", subtype="FLOAT")
    data = buffer.getvalue()
    write_private(output / name, data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "sample_rate": RATE,
            "subtype": "FLOAT", "seconds": values.size / RATE,
            "peak": float(np.abs(values).max()), "samples_outside_unit_range": int((np.abs(values) > 1).sum())}


def useful_metrics(values, elapsed, np):
    if np.abs(values).max() > 1:
        return {"pcm_metrics_unavailable": "native float exceeds PCM16 range; float audio retained without gain/clipping"}
    pcm = np.rint(values * 32767).astype("<i2").tobytes()
    return pcm_metrics(pcm, [{"ms": elapsed * 1000, "bytes": len(pcm)}], RATE)


def execute(args):
    # Authorization is checked before source access or importing any model.
    guard = Guard(args)
    if args.device == "cuda" and not args.window_file:
        raise ValueError("CUDA requires an owner-approved window-file")
    guard()
    source, reference, transcript, provenance = validate_files(args)
    output = private_directory(args.output_dir)
    if (output / "results.json").exists():
        raise ValueError("output already contains results; select a new directory")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, str(source / "src"))
    process_started = time.perf_counter()
    result = {"candidate": args.candidate, "action": args.action, "device": args.device,
              "provenance": provenance, "rate": RATE, "streaming": False,
              "streaming_evidence": "CFM.sample solves complete utterance then Vocos.decode; no PCM before full solve",
              "normalization": "NFC text only or explicit lowercase; no gain, denoise, trim, VAD or ASR",
              "observations": [], "failures": []}
    try:
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from f5_tts.model.modules import MelSpec
        from vocos import Vocos
        torch.set_num_threads(args.threads)
    except Exception as exc:
        result.update(success=False, failure_stage="imports")
        result["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        write_private(output / "results.json", json_bytes(result))
        raise
    result["versions"] = {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                           "cuda_build": torch.version.cuda}
    sync = (lambda: torch.cuda.synchronize()) if args.device == "cuda" else (lambda: None)
    guard.start_watchdog()
    def measured(call):
        sync(); start = time.perf_counter(); value = call(); sync()
        return value, time.perf_counter() - start
    try:
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        vocoder_started = time.perf_counter()
        vocoder = Vocos.from_hparams(str(Path(args.vocoder_dir) / "config.yaml"))
        weights = torch.load(Path(args.vocoder_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True)
        vocoder.load_state_dict(weights, strict=True)
        del weights
        vocoder = vocoder.eval().to(args.device)
        mel_extractor = MelSpec(mel_spec_type="vocos").to(args.device)
        sync(); guard.loaded = args.device == "cuda"
        result["vocoder_load_seconds"] = time.perf_counter() - vocoder_started
        guard()
        reference_started = time.perf_counter()
        audio, source_rate = sf.read(reference, dtype="float32", always_2d=True)
        if audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError("reference must be finite mono audio")
        if not 1 <= len(audio) / source_rate <= 90:
            raise ValueError("reference duration must be between one and ninety seconds")
        tensor = torch.from_numpy(audio[:, 0].copy()).unsqueeze(0)
        if source_rate != RATE:
            tensor = torchaudio.functional.resample(tensor, source_rate, RATE)
        if tensor.square().mean().sqrt().item() < 1e-6:
            raise ValueError("reference is silent")
        tensor = tensor.to(args.device)
        sync(); result["reference_conversion_seconds"] = time.perf_counter() - reference_started
        result["reference_source_rate"] = source_rate
        result["reference_seconds"] = tensor.shape[-1] / RATE
        result["resampling"] = "torchaudio.functional.resample default sinc Hann; gain unchanged"
        mel, result["reference_mel_seconds"] = measured(lambda: mel_extractor(tensor))
        cond = mel.permute(0, 2, 1).contiguous()
        result["reference_cache"] = {"scope": "in-process WAV conversion and mel computed once for all calls",
                                     "mel_shape": list(mel.shape), "dtype": str(mel.dtype)}
        if args.action == "roundtrip":
            write_private(output / "A-original.wav", reference.read_bytes())
            result["original"] = {"path": "A-original.wav", "sha256": file_hash(reference), "sample_rate": source_rate}
            result["converted"] = save_wave(output, "B-input-24000.wav", tensor[0].cpu().numpy(), sf, np)
            with torch.inference_mode():
                reconstructed, cost = measured(lambda: vocoder.decode(mel))
            result["reconstructed"] = save_wave(output, "C-vocos-24000.wav", reconstructed[0].cpu().numpy(), sf, np)
            result["vocoder_reconstruction_seconds"] = cost
            result["reconstruction_path"] = "F5 MelSpec(vocos, FFT1024,hop256,100mels,24k,power1,HTK,log clamp1e-5) -> Vocos.decode"
        else:
            from f5_tts.model import CFM, DiT
            from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
            from safetensors.torch import load_file
            mapping, vocab_size = get_tokenizer(str(args.vocab), "custom")
            if mapping.get(" ") != 0:
                raise ValueError("space must map to zero in exact training vocabulary")
            model_started = time.perf_counter()
            model = CFM(transformer=DiT(**ARCH, text_num_embeds=vocab_size, mel_dim=100),
                        mel_spec_kwargs=dict(mel_spec_type="vocos"),
                        odeint_kwargs=dict(method="euler"), vocab_char_map=mapping)
            state, ignored = ema_state(load_file(str(args.checkpoint), device="cpu"))
            model.load_state_dict(state, strict=True)
            del state
            dtype = torch.float16 if args.device == "cuda" else torch.float32
            model = model.eval().to(device=args.device, dtype=dtype)
            cond = cond.to(dtype)
            sync(); result["acoustic_model_load_seconds"] = time.perf_counter() - model_started
            result["load_compatibility"] = {"architecture": "F5TTS_Base", "strict": True,
                                            "ignored_documented_legacy_buffers": ignored, "vocab_size": vocab_size,
                                            "parameters": sum(p.numel() for p in model.parameters()), "dtype": str(dtype)}
            result["process_import_and_load_seconds"] = time.perf_counter() - process_started
            if args.device == "cuda":
                result["cuda_load_memory"] = {
                    "resident_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "resident_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "snapshot": gpu_snapshot()}
            result["warmup_policy"] = "optional separate warmup retained; first synthesis labeled cold_generation; no discarded calls"
            ref_text = normalize_text(transcript, args.text_mode)
            if len(ref_text[-1].encode()) == 1:
                ref_text += " "
            # Upstream uses floor(wave_samples/hop) for duration/cropping even
            # though center=True mel contains one extra conditioning frame.
            ref_frames = tensor.shape[-1] // HOP
            corpus = load_json(args.corpus)
            if args.case:
                selected = [case for case in corpus if case["id"] in args.case]
                if {case["id"] for case in selected} != set(args.case):
                    raise ValueError("unknown corpus case")
                corpus = selected
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            guard()
            schedule = [(nfe, case, repetition, False) for nfe in args.nfe
                        for case in corpus for repetition in range(args.repetitions)]
            if args.warmup:
                schedule.insert(0, (16, corpus[0], 0, True))
            for nfe, case, repetition, warmup in schedule:
                guard()
                identifier = f"{case['id']}-nfe{nfe}-r{repetition:02d}" + ("-warmup" if warmup else "")
                result["current_attempt"] = {"id": identifier, "case": case["id"], "nfe": nfe,
                                             "repetition": repetition, "seed": args.seed, "warmup": warmup}
                started = time.perf_counter()
                text = normalize_text(case["text"], args.text_mode)
                tokens = convert_char_to_pinyin([ref_text + text])
                oov = sorted({token for token in tokens[0] if token not in mapping})
                duration = duration_frames(ref_frames, ref_text, text, args.max_seconds)
                normal_seconds = time.perf_counter() - started
                with torch.inference_mode():
                    sampled, diffusion_seconds = measured(lambda: model.sample(
                        cond=cond, text=tokens, duration=duration, steps=nfe,
                        cfg_strength=2.0, sway_sampling_coef=-1.0, seed=args.seed,
                        use_epss=False))
                    generated, trajectory = sampled
                    del trajectory, sampled
                    generated_mel = generated[:, ref_frames:, :].permute(0, 2, 1).float()
                    wave, vocoder_seconds = measured(lambda: vocoder.decode(generated_mel))
                values = wave[0].cpu().numpy()
                elapsed = time.perf_counter() - started
                meta = save_wave(output, identifier + ".wav", values, sf, np)
                row = {"case": case["id"], "repetition": repetition, "seed": args.seed,
                       "source_text": case["text"], "effective_text": text,
                       "text_mode": args.text_mode, "oov_tokens": oov,
                       "nfe": nfe, "ode_method": "euler", "use_epss": False,
                       "cfg_strength": 2.0, "sway_sampling_coef": -1.0,
                       "transformer_calls": nfe, "cfg_batch_size": 2,
                       "output_blocks": 1, "sentence_chunks": 1,
                       "normalization_tokenization_seconds": normal_seconds,
                       "diffusion_seconds": diffusion_seconds, "vocoder_seconds": vocoder_seconds,
                       "complete_seconds": elapsed, "rtf": elapsed / meta["seconds"],
                       "reference_cache_hit": True, "warmup": warmup,
                       "cold_generation": not result["observations"], **meta,
                       **useful_metrics(values, elapsed, np)}
                result["observations"].append(row)
                if row["warmup"]:
                    result["warmup_seconds"] = elapsed
                if row["cold_generation"]:
                    result["cold_generation_seconds"] = elapsed
                # Incremental receipts preserve every attempt if a later
                # case fails or the operational deadline terminates us.
                write_private(output / (identifier + ".json"), json_bytes(row))
                del generated, generated_mel, wave
            if args.device == "cuda":
                result["cuda_memory"] = {"resident_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                                         "resident_reserved_mib": torch.cuda.memory_reserved() / 2**20,
                                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                         "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                         "snapshot": gpu_snapshot()}
        result["success"] = True
        result.pop("current_attempt", None)
    except Exception as exc:
        result["success"] = False
        result["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_private(output / "results.json", json_bytes(result))
        guard.stop.set()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--action", choices=("roundtrip", "infer"), default="roundtrip")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate", default="f5-ptbr-vocos")
    for name in ("source-dir", "checkpoint", "vocab", "vocoder-dir", "reference-wav",
                 "reference-text-file", "receipt", "output-dir", "window-file"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("corpus.pt-BR.json"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--nfe", type=int, nargs="+", choices=(8, 16, 32), default=[8, 16, 32])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", action="store_true", help="retain a separate first-case NFE16 warmup before measured schedule")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=40733133)
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--text-mode", choices=("verbatim", "lowercase"), default="lowercase")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"execute": False, "gpu_touched": False, "downloads": False,
                          "source_commit": SOURCE_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA256,
                          "architecture": "F5TTS_Base", "vocoder": "Vocos 24k 100mel",
                          "actions": ["roundtrip", "infer"], "neural_streaming": False,
                          "nfe": args.nfe, "reference_cache": "wave conversion + F5 mel once",
                          "cuda_requires": "owner-approved exclusive window; production stopped separately"}, indent=2))
        return 0
    if not 1 <= args.repetitions <= 3 or not 1 <= args.threads <= 4 or not 0 <= args.seed < 2**32:
        parser.error("bounded repetitions 1..3, threads 1..4 and uint32 seed required")
    if not 1 <= args.max_seconds <= 90:
        parser.error("max-seconds must be between one and ninety")
    required = ("source_dir", "vocab", "vocoder_dir", "reference_wav", "reference_text_file", "receipt", "output_dir")
    if any(getattr(args, name) is None for name in required):
        parser.error("execution requires local source/vocab/vocoder/reference/transcript/receipt/output paths")
    result = execute(args)
    print(json.dumps({"success": result["success"], "observations": len(result["observations"]),
                      "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
