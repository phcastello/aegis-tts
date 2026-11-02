#!/usr/bin/env python3
"""Private, resumable fine-tuning of the pinned firstpixel F5TTS_Base.

No service control, downloads, acoustic effects, test-set optimization or wall
clock training cutoff. CPU check is separate from explicitly authorized CUDA.
Uses upstream CFM loss and activation checkpointing; Vocos remains frozen.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

SOURCE_COMMIT = "45868abaabd74aabf03cd673d6dd00cc9d9c01ae"
BASE_SHA = "c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f"
VOCAB_SHA = "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
PRIVATE = Path("/var/tmp/aegis-v2-evaluation-20261102/f5")
ARCH = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
            text_mask_padding=False, conv_layers=4, pe_attn_head=1,
            attn_backend="torch", attn_mask_enabled=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        os.chmod(tmp, 0o600)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def log_event(root, event, **values):
    value = dict(event=event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values)
    with (root / "events.jsonl").open("a") as f:
        os.chmod(f.name, 0o600)
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def load_rows(root):
    result = {}
    ids = set()
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (root / "prepared" / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"empty {split}")
        for row in rows:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier):
                raise ValueError("ID must be a nonempty filename-safe string")
            if identifier in ids:
                raise ValueError(f"ID occurs across split/rows: {identifier}")
            ids.add(identifier)
            if not 0.3 <= float(row["duration"]) <= 30:
                raise ValueError(f"unsupported duration (never silently replaced): {identifier}")
            if not row["text"].strip():
                raise ValueError(f"empty text: {identifier}")
        result[split] = rows
    return result


def dependencies(args, cuda=False):
    if not cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    source = Path(args.source_dir)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"unrecognized upstream source commit: {commit}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("pinned upstream source has modifications")
    if sha256(args.vocab) != VOCAB_SHA or sha256(args.checkpoint) != BASE_SHA:
        raise ValueError("base checkpoint/vocabulary SHA256 mismatch")
    sys.path.insert(0, str(source / "src"))
    import torch
    torch.set_num_threads(args.threads)
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
    from safetensors.torch import load_file
    mapping, size = get_tokenizer(args.vocab, "custom")
    if mapping.get(" ") != 0 or size != 2545:
        raise ValueError("wrong vocabulary IDs")
    model = CFM(transformer=DiT(**ARCH, checkpoint_activations=not args.no_checkpointing,
                               text_num_embeds=size, mel_dim=100),
                mel_spec_kwargs=dict(mel_spec_type="vocos"), vocab_char_map=mapping,
                odeint_kwargs=dict(method="euler"))
    weights = load_file(args.checkpoint, device="cpu")
    model.load_state_dict(weights, strict=True)
    del weights
    gc.collect()
    return torch, model, mapping, convert_char_to_pinyin


def cache_path(root, row):
    return root / "prepared" / "mels" / (row["id"] + ".npy")


def audit_cache(root, rows, mapping, tokenize):
    import numpy as np
    oov = {}
    result = {}
    for split, records in rows.items():
        summaries = []
        for row in records:
            tokens = tokenize([row["text"]])[0]
            unknown = sorted(set(tokens) - set(mapping))
            if unknown:
                oov[row["id"]] = unknown
            row["_tokens"] = tokens
            path = cache_path(root, row)
            values = np.load(path, allow_pickle=False, mmap_mode="r")
            if values.dtype != np.float32 or values.ndim != 2 or 100 not in values.shape:
                raise ValueError(f"unexpected mel shape or dtype: {path}")
            frames = values.shape[1] if values.shape[0] == 100 else values.shape[0]
            if abs(frames - (float(row["duration"]) * 24000 // 256 + 1)) > 2:
                raise ValueError(f"mel/duration mismatch: {path}")
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite cached mel: {path}")
            row["_frames"] = frames
            summaries.append(dict(id=row["id"], frames=frames, sha256=sha256(path)))
        result[split] = dict(count=len(records), manifest_sha256=sha256(root / "prepared" / f"{split}.jsonl"), mels=summaries)
    if oov:
        raise ValueError(f"OOV tokens: {oov}")
    return result


def check(args, root):
    torch, model, mapping, tokenize = dependencies(args)
    result = dict(source_commit=SOURCE_COMMIT, base_sha256=BASE_SHA,
                  vocab_sha256=VOCAB_SHA, architecture=ARCH, strict_load=True,
                  parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, cuda_build=torch.version.cuda,
                  dataset=audit_cache(root, load_rows(root), mapping, tokenize))
    atomic_json(root / "cpu-check.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "dataset"}), flush=True)


def require_gpu_authorization(args, root):
    if not args.cuda_authorized or not (root / "gpu-window-authorized.json").is_file():
        raise ValueError("CUDA requires --cuda-authorized and private GPU window authorization receipt")
    auth = json.loads((root / "gpu-window-authorized.json").read_text())
    if not (auth.get("owner_authorized") and auth.get("exclusive") and auth.get("status") == "active"):
        raise ValueError("receipt must confirm owner_authorized/exclusive and status=active")
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if any(line.strip() and int(line.strip()) != os.getpid() for line in apps):
        raise RuntimeError("GPU has another compute process; refusing overlap")


class Engine:
    def __init__(self, args, root):
        require_gpu_authorization(args, root)
        self.args, self.root = args, root
        self.torch, self.model, mapping, tokenize = dependencies(args, cuda=True)
        self.device = "cuda"
        self.rows = load_rows(root)
        evidence = audit_cache(root, self.rows, mapping, tokenize)
        self.dataset_fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        t = self.torch
        if not t.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if args.precision == "bf16" and not t.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported; choose fp16 with GradScaler")
        self.dtype = {"bf16": t.bfloat16, "fp16": t.float16, "fp32": t.float32}[args.precision]
        random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed_all(args.seed)
        self.model = self.model.to("cuda")  # FP32 parameters and Adam moments.
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.learning_rate,
                                      weight_decay=0.01, foreach=False, fused=False)
        self.scaler = t.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        self.update = 0
        self.best = math.inf
        self.best_epoch = 0
        self.reference_best = math.inf
        self.patience = 0
        self.lr_reduced = False
        self.phase = "pilot"
        self.initial_validation = None
        self.elapsed_training = 0.0
        self.history = []
        self.peak_allocated_mib = 0.0
        self.peak_reserved_mib = 0.0
        self.checkpoints = root / "checkpoints"
        self.checkpoints.mkdir(mode=0o700, exist_ok=True)

    def amp(self):
        if self.device == "cpu":
            return contextlib.nullcontext()
        return self.torch.autocast("cuda", dtype=self.dtype, enabled=self.args.precision != "fp32")

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def batch(self, row):
        import numpy as np
        values = np.load(cache_path(self.root, row), allow_pickle=False)
        if values.shape[0] == 100:
            values = values.T
        mel = self.torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0).to(self.device)
        lens = self.torch.tensor([mel.shape[1]], device=self.device, dtype=self.torch.long)
        return mel, [row["_tokens"]], lens

    def loss(self, row):
        mel, tokens, lens = self.batch(row)
        with self.amp():
            loss, _, _ = self.model(mel, text=tokens, lens=lens)
        if not self.torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss: {row['id']}")
        return loss

    def memory(self):
        t = self.torch
        # Inference hooks reset PyTorch's peak counters for their own timings;
        # retain training high water marks independently across those resets.
        self.peak_allocated_mib = max(self.peak_allocated_mib, t.cuda.max_memory_allocated() / 2**20)
        self.peak_reserved_mib = max(self.peak_reserved_mib, t.cuda.max_memory_reserved() / 2**20)
        return dict(resident_allocated_mib=t.cuda.memory_allocated() / 2**20,
                    resident_reserved_mib=t.cuda.memory_reserved() / 2**20,
                    peak_allocated_mib=self.peak_allocated_mib,
                    peak_reserved_mib=self.peak_reserved_mib)

    def validate(self):
        t = self.torch
        python_state = random.getstate()
        losses = []
        started = time.monotonic()
        self.model.eval()
        try:
            with t.random.fork_rng(devices=[t.cuda.current_device()] if self.device == "cuda" else []), t.inference_mode():
                for i, row in enumerate(self.rows["validation"]):
                    # Identical mask, noise, time and CFG dropout for each row at every evaluation.
                    random.seed(2000003 + i)
                    t.manual_seed(2000003 + i)
                    if self.device == "cuda":
                        t.cuda.manual_seed_all(2000003 + i)
                    losses.append(float(self.loss(row)))
        finally:
            random.setstate(python_state)
            self.model.train()
        self.synchronize()
        return dict(loss=statistics.mean(losses), standard_error=statistics.stdev(losses) / math.sqrt(len(losses)) if len(losses) > 1 else 0.0,
                    count=len(losses), per_item_losses=losses, seconds=time.monotonic() - started)

    def assess_validation(self, epoch, current):
        """Validation-only phase/stop decisions, independent of elapsed time."""
        meaningful = current < self.reference_best * (1.0 - self.args.min_relative_improvement)
        if meaningful:
            self.reference_best, self.patience = current, 0
        else:
            self.patience += 1
        stop_reason = None
        reduced_now = False
        if self.phase == "pilot" and epoch >= 3:
            if self.best < self.initial_validation * (1.0 - self.args.min_relative_improvement):
                self.phase = "full"
                log_event(self.root, "pilot_promoted_automatically", epoch=epoch, improvement=1.0-self.best/self.initial_validation)
            elif epoch >= 6:
                stop_reason = "pilot_no_meaningful_validation_improvement_after_six_epochs"
        if self.phase == "full" and epoch >= self.args.minimum_epochs and self.patience >= self.args.patience:
            if not self.lr_reduced:
                self.lr_reduced, self.patience = True, 0
                reduced_now = True
                log_event(self.root, "learning_rate_reduced", epoch=epoch, factor=0.3)
            else:
                stop_reason = "validation_converged_after_plateau_and_single_lr_reduction"
        recent = self.history[-3:]
        if self.phase == "full" and epoch >= self.args.minimum_epochs and len(recent) == 3 and all(h["validation_loss"] > self.best * 1.10 for h in recent):
            stop_reason = "validation_deteriorated_over_ten_percent_for_three_evaluations"
        return meaningful, stop_reason, reduced_now

    def synthesize(self, label):
        from training.f5_evaluate import quick_synthesis
        # Evaluator also preserves mode/RNG, and decodes Vocos under its own AMP.
        log_event(self.root, "checkpoint_synthesis_started", label=label, update=self.update)
        result = quick_synthesis(self.model, self.root, label)
        log_event(self.root, "checkpoint_synthesis_completed", label=label, update=self.update, result=result)

    def learning_rate(self):
        # Warmup then slow inverse-square-root decay; no predetermined last epoch.
        factor = min((self.update + 1) / max(1, self.args.warmup_updates),
                     math.sqrt(max(1, self.args.warmup_updates) / max(1, self.update + 1)))
        return self.args.learning_rate * factor * (0.3 if self.lr_reduced else 1.0)

    def train_group(self, group, *, epoch, label=None):
        t = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.learning_rate()
        for params in self.optimizer.param_groups:
            params["lr"] = lr
        self.synchronize()
        started = time.monotonic()
        losses = []
        for row in group:
            loss = self.loss(row)
            losses.append(float(loss.detach()))
            self.scaler.scale(loss / len(group)).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = t.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.update += 1
        self.synchronize()
        elapsed = time.monotonic() - started
        self.elapsed_training += elapsed
        value = dict(update=self.update, epoch=epoch, phase=self.phase, loss=statistics.mean(losses),
                     gradient_norm=float(grad_norm), lr=lr, microbatches=len(group),
                     seconds=elapsed, audio_seconds=sum(float(row["duration"]) for row in group),
                     audio_seconds_per_second=sum(float(row["duration"]) for row in group) / elapsed,
                     label=label, **self.memory())
        log_event(self.root, "update", **value)
        return value

    def weights(self, filename):
        from safetensors.torch import save_file
        path = self.checkpoints / filename
        tmp = path.with_suffix(path.suffix + ".tmp")
        state = {key: value.detach().cpu().contiguous() for key, value in self.model.state_dict().items()}
        save_file(state, str(tmp), metadata={"source_commit": SOURCE_COMMIT, "base_sha256": BASE_SHA,
                                           "update": str(self.update), "dataset_fingerprint": self.dataset_fingerprint})
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return dict(path=str(path), sha256=sha256(path), update=self.update)

    def save(self, epoch, next_group):
        t = self.torch
        if shutil.disk_usage(self.checkpoints).free < 5_000_000_000:
            raise OSError("need 5GB free for atomic resumable checkpoint")
        def cpu_tree(value):
            if t.is_tensor(value):
                return value.detach().cpu()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_tree(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_tree(v) for v in value)
            return value
        state = dict(model=cpu_tree(self.model.state_dict()), optimizer=cpu_tree(self.optimizer.state_dict()),
                     scaler=self.scaler.state_dict(), python_rng=random.getstate(), torch_rng=t.get_rng_state(),
                     cuda_rng=t.cuda.get_rng_state_all() if self.device == "cuda" else [], epoch=epoch, next_group=next_group,
                     args=vars(self.args), source_commit=SOURCE_COMMIT, base_sha256=BASE_SHA,
                     dataset_fingerprint=self.dataset_fingerprint,
                     progress={key: getattr(self, key) for key in ("update", "best", "best_epoch", "reference_best", "patience", "lr_reduced", "phase", "initial_validation", "elapsed_training", "history", "peak_allocated_mib", "peak_reserved_mib")})
        tmp = self.checkpoints / "last.pt.tmp"
        t.save(state, tmp)
        os.chmod(tmp, 0o600)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, self.checkpoints / "last.pt")
        del state
        gc.collect()
        atomic_json(self.root / "progress.json", dict(epoch=epoch, next_group=next_group,
                    update=self.update, best_validation=self.best, best_epoch=self.best_epoch,
                    phase=self.phase, lr_reduced=self.lr_reduced, patience=self.patience,
                    training_seconds=self.elapsed_training, memory=self.memory()))

    def resume(self):
        state = self.torch.load(self.checkpoints / "last.pt", weights_only=True, map_location="cpu")
        if state["source_commit"] != SOURCE_COMMIT or state["dataset_fingerprint"] != self.dataset_fingerprint:
            raise ValueError("resume dataset/source differs")
        for key in ("accumulation", "seed", "learning_rate", "warmup_updates", "precision", "no_checkpointing"):
            if state["args"][key] != getattr(self.args, key):
                raise ValueError(f"resume configuration differs: {key}")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"])
        self.torch.set_rng_state(state["torch_rng"])
        if self.device == "cuda":
            self.torch.cuda.set_rng_state_all(state["cuda_rng"])
        for key, value in state["progress"].items():
            setattr(self, key, value)
        position = state["epoch"], state["next_group"]
        del state
        gc.collect()
        log_event(self.root, "resumed", epoch=position[0], next_group=position[1], update=self.update)
        return position

    def benchmark(self):
        rows = sorted(self.rows["train"], key=lambda row: row["_frames"])
        selected = [("shortest", rows[:self.args.accumulation]),
                    ("median", rows[len(rows)//2:len(rows)//2+self.args.accumulation]),
                    ("median", rows[len(rows)//2:len(rows)//2+self.args.accumulation]),
                    ("longest", rows[-self.args.accumulation:]),
                    ("longest", rows[-self.args.accumulation:])]
        self.torch.cuda.reset_peak_memory_stats()
        initial = self.validate()
        observations = [self.train_group(group, epoch=0, label=label) for label, group in selected]
        final = self.validate()
        middle = statistics.median([obs["seconds"] for obs in observations[1:]])
        updates_per_epoch = math.ceil(len(rows) / self.args.accumulation)
        result = dict(config=vars(self.args), initial_validation=initial, final_validation=final,
                      observations=observations, update_seconds_median=middle,
                      updates_per_epoch=updates_per_epoch,
                      projected_epoch_seconds=updates_per_epoch * middle + final["seconds"],
                      estimate_is_not_a_time_limit=True, memory=self.memory(), success=True)
        atomic_json(self.root / f"benchmark-{self.args.precision}-a{self.args.accumulation}.json", result)
        return result

    def train(self):
        if self.args.resume:
            epoch, skip_group = self.resume()
        else:
            if (self.checkpoints / "last.pt").exists():
                raise ValueError("last.pt exists: use --resume, never overwrite an unexamined run")
            initial = self.validate()
            self.initial_validation = initial["loss"]
            self.best = self.reference_best = initial["loss"]
            atomic_json(self.root / "best.json", dict(epoch=0, validation=initial,
                        weights=self.weights("best.safetensors"), selection="pretrained_initial_validation"))
            log_event(self.root, "initial_validation", validation=initial)
            epoch, skip_group = 1, 0
        stop_reason = None
        while stop_reason is None:
            rng = random.Random(self.args.seed + epoch)
            records = self.rows["train"].copy()
            rng.shuffle(records)
            groups = [records[i:i+self.args.accumulation] for i in range(0, len(records), self.args.accumulation)]
            epoch_losses = []
            for group_index, group in enumerate(groups):
                if group_index < skip_group:
                    continue
                obs = self.train_group(group, epoch=epoch)
                epoch_losses.append(obs["loss"])
                if self.update % self.args.save_every == 0:
                    self.save(epoch, group_index + 1)
            validation = self.validate()
            current = validation["loss"]
            if current < self.best:
                self.best, self.best_epoch = current, epoch
                weights = self.weights("best.safetensors")
                atomic_json(self.root / "best.json", dict(epoch=epoch, validation=validation, weights=weights))
            train_loss = statistics.mean(epoch_losses) if epoch_losses else None
            self.history.append(dict(epoch=epoch, update=self.update, train_loss=train_loss,
                                     validation_loss=current, standard_error=validation["standard_error"],
                                     phase=self.phase))
            if epoch == 3:
                atomic_json(self.root / "pilot-checkpoint.json", self.weights("pilot.safetensors"))
            if epoch == 12:
                atomic_json(self.root / "intermediate-checkpoint.json", self.weights("intermediate.safetensors"))
            meaningful, stop_reason, reduced_now = self.assess_validation(epoch, current)
            log_event(self.root, "epoch", epoch=epoch, update=self.update, train_loss=train_loss,
                      validation=validation, best=self.best, meaningful_improvement=meaningful,
                      patience=self.patience, phase=self.phase, stop_reason=stop_reason)
            self.save(epoch + 1, 0)
            atomic_json(self.root / "loss-history.json", self.history)
            if epoch in (3, 6, 12) or (self.lr_reduced and self.best_epoch == epoch):
                self.synthesize(f"epoch-{epoch:03d}-u{self.update:06d}")
            epoch, skip_group = epoch + 1, 0
        atomic_json(self.root / "last-weights.json", self.weights("last.safetensors"))
        result = dict(stop_reason=stop_reason, completed_epochs=epoch-1, updates=self.update,
                      best_epoch=self.best_epoch, initial_validation=self.initial_validation,
                      best_validation=self.best, training_seconds=self.elapsed_training,
                      memory=self.memory(), dataset_fingerprint=self.dataset_fingerprint,
                      checkpoint_root=str(self.checkpoints), config=vars(self.args))
        atomic_json(self.root / "training-result.json", result)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("check", "benchmark", "train"), default="check")
    p.add_argument("--run-root", default="/var/tmp/aegisvoice-v2-training-20261102")
    p.add_argument("--source-dir", default=str(PRIVATE / "source"))
    p.add_argument("--checkpoint", default=str(PRIVATE / "model_last.safetensors"))
    p.add_argument("--vocab", default=str(PRIVATE / "vocab.hf.txt"))
    p.add_argument("--cuda-authorized", action="store_true")
    p.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-updates", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--minimum-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min-relative-improvement", type=float, default=0.005)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.accumulation < 1 or args.patience < 1 or args.save_every < 1 or args.warmup_updates < 1:
        raise ValueError("positive accumulation/patience/save interval/warmup required")
    os.umask(0o077)
    root = Path(args.run_root).resolve(strict=True)
    if args.action == "check":
        check(args, root)
        return 0
    started = time.monotonic()
    engine = None
    try:
        engine = Engine(args, root)
        result = engine.benchmark() if args.action == "benchmark" else engine.train()
        log_event(root, args.action + "_completed", wall_seconds=time.monotonic()-started,
                  summary={k: v for k, v in result.items() if k not in ("observations", "config", "initial_validation", "final_validation")})
    except BaseException as exc:
        log_event(root, args.action + "_failed", error_type=type(exc).__name__, error=str(exc),
                  update=engine.update if engine else 0,
                  memory=engine.memory() if engine else None)
        raise
    finally:
        if engine is not None:
            del engine
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
