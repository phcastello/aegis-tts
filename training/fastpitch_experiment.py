"""Bounded-resource FastPitch experiments with unmodified NVIDIA model/loss.

Uses a separate PyTorch AdamW training loop (no Apex), explicit PT graphemes,
BigVGAN-compatible Slaney mels, explicit pitch labels, and complete checkpoints.
Private manifests/features/checkpoints belong outside the Git tree.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import types
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F

UPSTREAM = "729963dd47e7c8bd462ad10bfac7a7b0b604e6dd"
DEFAULT_SOURCE = "/var/tmp/aegis-v2-work/fastpitch/source"
VOCAB = ["<PAD>", "<UNK>"] + list(" !'(),-.:;?abcdefghijklmnopqrstuvwxyzáàâãäéèêëíìîïóòôõöúùûüçñ0123456789%/ºª+")
MEL_CONFIG = dict(sr=24000, n_fft=1024, hop=256, win=1024, n_mels=100, fmin=0., fmax=None)
_FEATURE_VERIFIED = {}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_history(path,value,max_bytes=10*1024**2):
    """Durable compact metrics independent of the supervisor's bounded stdout."""
    path=Path(path)
    value=dict(value,utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
    encoded=(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n").encode()
    if path.exists() and path.stat().st_size+len(encoded)>max_bytes:
        previous=path.read_bytes()
        tail=previous[-max_bytes//2:].split(b"\n",1)[-1]
        marker=(json.dumps(dict(event="history_compacted",previous_sha256=hashlib.sha256(previous).hexdigest(),
                                previous_bytes=len(previous),retained_tail_bytes=len(tail)))+"\n").encode()
        temp=path.with_suffix(path.suffix+".tmp")
        with open(temp,"wb") as f:
            f.write(marker+tail);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    with open(path,"ab") as f:
        f.write(encoded);f.flush();os.fsync(f.fileno())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = text.translate(str.maketrans({"“": "'", "”": "'", "‘": "'", "’": "'", "\"": "'", "–": "-", "—": "-", "−": "-", "·": ".", "…": "..."}))
    return " ".join(text.split())


def tokens(text, vocab=VOCAB):
    text = " " + normalize(text) + " "
    lut = {c: i for i, c in enumerate(vocab)}
    missing = sorted(set(text) - set(lut))
    if missing:
        raise ValueError(f"Unsupported graphemes: {missing!r}; never silently discard text")
    return torch.tensor([lut[c] for c in text], dtype=torch.long)


def mel_spectrogram(waveform):
    """Same formula/defaults as pinned BigVGAN, without volume normalization."""
    from librosa.filters import mel
    basis = torch.from_numpy(mel(sr=24000, n_fft=1024, n_mels=100, fmin=0, fmax=None)).float()
    window = torch.hann_window(1024)
    x = F.pad(waveform.reshape(1, 1, -1), (384, 384), mode="reflect").squeeze(1)
    s = torch.stft(x, 1024, hop_length=256, win_length=1024, window=window,
                   center=False, normalized=False, onesided=True, return_complex=True)
    magnitude = torch.sqrt(torch.view_as_real(s).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(basis @ magnitude, min=1e-5)).squeeze(0)


def prepare_one(job):
    import librosa
    import soundfile as sf
    row, root, pitch_method = job
    torch.set_num_threads(1)
    wav = Path(row["audio_filepath"])
    ident = row.get("id", wav.stem)
    dest = Path(root) / "features" / (ident + ".pt")
    dest.parent.mkdir(exist_ok=True)
    start = time.monotonic()
    if not dest.exists():
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 24000:
            x = librosa.resample(x, orig_sr=sr, target_sr=24000)
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1.00001:
            raise ValueError(f"Non-finite or overrange audio {wav}")
        mel = mel_spectrogram(torch.from_numpy(x))
        if pitch_method == "world":
            try:
                import pyworld
            except ImportError:
                # Only the preexisting audited 2.3MB package is exposed.
                sys.path.append("/var/tmp/aegis-v2-work/fastpitch/pyworld-overlay")
                import pyworld
            f0, times = pyworld.dio(x.astype(np.float64), 24000, frame_period=256/24)
            f0 = pyworld.stonemask(x.astype(np.float64), f0, times, 24000)
        else:
            f0, _, _ = librosa.pyin(x, sr=24000, fmin=50, fmax=900,
                                    frame_length=1024, hop_length=256)
            f0 = np.nan_to_num(f0)
            times = np.arange(len(f0)) * 256 / 24000
        centers = (np.arange(mel.shape[1]) * 256 + 128) / 24000
        f0 = np.interp(centers, times, f0).astype("float32")
        data = dict(mel=mel, pitch=torch.from_numpy(f0)[None], text=tokens(row["text"]),
                    id=ident, audio_filepath=str(wav), audio_sha256=digest(wav),
                    normalized_text=normalize(row["text"]), mel_config=MEL_CONFIG,
                    pitch_method=pitch_method)
        atomic_torch(dest, data)
    data = torch.load(dest, map_location="cpu", weights_only=True)
    actual_audio_sha=digest(wav)
    expected_audio_sha=row.get("sha256") or row.get("source_sha256") or row.get("source_audio_sha256")
    if (data["normalized_text"] != normalize(row["text"]) or data["audio_sha256"] != actual_audio_sha
            or (expected_audio_sha and actual_audio_sha != expected_audio_sha)
            or data.get("pitch_method", "pyin") != pitch_method):
        raise ValueError(f"Stale feature cache {dest}")
    voiced = data["pitch"][data["pitch"] > 0].double()
    stats = [float(voiced.sum()), float(voiced.square().sum()), voiced.numel()]
    return dict(feature=str(dest), id=ident, frames=data["mel"].shape[1], feature_sha256=digest(dest),
                text_length=len(data["text"]), group_id=row.get("group_id")), stats, time.monotonic()-start


def prepare(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifests = [Path(p) for p in args.manifests]
    selections={p:[r for r in rows(p) if r.get("id") not in args.exclude_id] for p in manifests}
    seen={}
    for path,selection in selections.items():
        for row in selection:
            ident=row["id"]
            if ident in seen:raise ValueError(f"Repeated ID {ident} in {seen[ident]} and {path}; quarantine conflicting IDs or explicitly deduplicate exact pairs before feature work")
            seen[ident]=str(path)
    stat_sums = [0., 0., 0]
    for manifest in manifests:
        output = []
        selected=selections[manifest]
        if args.limit:selected=selected[:args.limit]
        jobs=[(row,str(root),args.pitch_method) for row in selected]
        pool=ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) if args.workers>1 else None
        results=pool.map(prepare_one,jobs) if pool else map(prepare_one,jobs)
        manifest_hash=digest(manifest)
        try:
            for item,stats,seconds in results:
                if manifest.stem=="train":stat_sums=[a+b for a,b in zip(stat_sums,stats)]
                item["source_manifest_sha256"]=manifest_hash
                output.append(item)
                print(json.dumps(dict(event="prepared",id=item["id"],frames=item["frames"],seconds=seconds)),flush=True)
        finally:
            if pool:pool.shutdown(cancel_futures=True)
        target = root / (manifest.stem + ".jsonl")
        tmp = target.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in output))
        os.replace(tmp, target)
    if stat_sums[2]:
        mean = stat_sums[0] / stat_sums[2]
        std = math.sqrt(max(1., stat_sums[1]/stat_sums[2] - mean*mean))
        atomic_json(root / "features.json", dict(vocab=VOCAB, mel=MEL_CONFIG, pitch_mean=mean,
                    pitch_std=std, pitch_voiced_frames=stat_sums[2], upstream=UPSTREAM,
                    source_manifests={str(p): digest(p) for p in manifests}, pitch_method=args.pitch_method,
                    excluded_ids=args.exclude_id))


def imports(source):
    source = Path(source)
    meta = json.loads((source / "UPSTREAM.json").read_text())
    if meta["commit"] != UPSTREAM:
        raise ValueError("FastPitch source pin mismatch")
    for item in meta["files"]:
        if digest(source / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed upstream source: {item['path']}")
    sys.path.insert(0, str(source))
    import fastpitch.model as model_module
    from fastpitch.model import FastPitch
    from fastpitch.arg_parser import parse_fastpitch_args
    from fastpitch.loss_function import FastPitchLoss
    from fastpitch.attn_loss_function import AttentionBinarizationLoss
    from fastpitch.alignment import mas_width1

    # Quantizing cumulative frame boundaries to BF16 drops short tokens after
    # frame256. Keep all alignment indices/sums FP32; cast only acoustic values.
    if not hasattr(model_module, "_aegis_original_regulate_len"):
        model_module._aegis_original_regulate_len = model_module.regulate_len
        model_module._aegis_original_average_pitch = model_module.average_pitch

    def stable_regulate_len(durations, enc_out, pace=1., mel_max_len=None):
        dtype=enc_out.dtype
        value,lens=model_module._aegis_original_regulate_len(
            durations.float(),enc_out.float(),pace,mel_max_len)
        return value.to(dtype),lens

    def stable_average_pitch(pitch,durs):
        return model_module._aegis_original_average_pitch(pitch.float(),durs.float())

    model_module.regulate_len=stable_regulate_len
    model_module.average_pitch=stable_average_pitch

    class PortableFastPitch(FastPitch):
        def binarize_attention(self, attn, in_lens, out_lens):
            # Upstream uses attn.get_device(), which is -1 on CPU.
            # Same MAS routine, explicit device, also allows CPU pilot.
            with torch.no_grad():
                log_attn = torch.log(attn.detach()).float().cpu().numpy()
                hard = np.zeros_like(log_attn)
                for i in range(len(in_lens)):
                    m, t = int(out_lens[i]), int(in_lens[i])
                    hard[i, 0, :m, :t] = mas_width1(log_attn[i, 0, :m, :t])
                return torch.from_numpy(hard).to(attn.device, dtype=attn.dtype)

    parser = parse_fastpitch_args(argparse.ArgumentParser(add_help=False))
    defaults = vars(parser.parse_args([]))
    config = {k: defaults[k] for k in inspect.signature(FastPitch).parameters if k in defaults}
    config.update(n_speakers=1, n_mel_channels=100, n_symbols=len(VOCAB))
    return PortableFastPitch, FastPitchLoss, AttentionBinarizationLoss, config


def sample_batch(items, device, pitch_mean, pitch_std):
    # Local collator avoids the upstream English text frontend and its imports.
    from scipy.stats import betabinom
    b = len(items)
    tl = torch.tensor([len(x["text"]) for x in items], dtype=torch.long)
    ml = torch.tensor([x["mel"].shape[1] for x in items], dtype=torch.long)
    text = torch.zeros(b, int(tl.max()), dtype=torch.long)
    mel = torch.zeros(b, 100, int(ml.max()))
    pitch = torch.zeros(b, 1, int(ml.max()))
    energy = torch.zeros(b, int(ml.max()))
    prior = torch.zeros(b, int(ml.max()), int(tl.max()))
    for i, item in enumerate(items):
        t, m = int(tl[i]), int(ml[i])
        text[i, :t] = item["text"]
        mel[i, :, :m] = item["mel"]
        p = item["pitch"].clone()
        p[p != 0] = (p[p != 0] - pitch_mean) / pitch_std
        pitch[i, :, :m] = p
        energy[i, :m] = torch.norm(item["mel"], dim=0)
        a = np.arange(1, m+1)[:, None]
        matrix = betabinom.pmf(np.arange(t)[None, :], t, a, m+1-a)
        prior[i, :m, :t] = torch.from_numpy(matrix.astype("float32"))
    text, tl, mel, ml, pitch, energy, prior = [x.to(device) for x in (text, tl, mel, ml, pitch, energy, prior)]
    return [text, tl, mel, ml, pitch, energy, None, prior, [x["id"] for x in items]], [mel, tl, ml]


def load_feature(row):
    path=Path(row["feature"])
    stat=path.stat()
    signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    if _FEATURE_VERIFIED.get(str(path)) != signature:
        if row.get("feature_sha256") and digest(path) != row["feature_sha256"]:
            raise ValueError(f"Changed feature object {path}")
        _FEATURE_VERIFIED[str(path)]=signature
    return torch.load(row["feature"], map_location="cpu", weights_only=True)


def audit_features(features,manifests,output):
    """Validate every selected object before updates; do not trust exit0 alone."""
    started=time.monotonic();source={}
    for name,expected_sha in features["source_manifests"].items():
        if digest(name)!=expected_sha:raise ValueError(f"Source manifest changed: {name}")
        for row in rows(name):
            if row["id"] in source:raise ValueError(f"Duplicate source ID {row['id']}; use an immutable clean selection")
            source[row["id"]]=row
    count=0;audio_bytes=0
    for feature_row in manifests:
        item=load_feature(feature_row)
        expected=source[feature_row["id"]]
        audio=Path(expected["audio_filepath"])
        actual_audio=digest(audio);audio_bytes+=audio.stat().st_size
        source_sha=expected.get("sha256") or expected.get("source_sha256") or expected.get("source_audio_sha256")
        if (item["id"]!=feature_row["id"] or item["normalized_text"]!=normalize(expected["text"])
                or not torch.equal(item["text"],tokens(expected["text"]))
                or item["audio_sha256"]!=actual_audio or (source_sha and source_sha!=actual_audio)
                or Path(item["audio_filepath"]).resolve()!=audio.resolve()
                or item["mel_config"]!=MEL_CONFIG or item.get("pitch_method","pyin")!=features["pitch_method"]
                or item["mel"].shape!=(100,feature_row["frames"])
                or item["pitch"].shape!=(1,feature_row["frames"])
                or len(item["text"])!=feature_row["text_length"]
                or not torch.isfinite(item["mel"]).all() or not torch.isfinite(item["pitch"]).all()):
            raise ValueError(f"Feature/source identity, text, audio, shape or finite-value mismatch: {feature_row['id']}")
        count+=1
    result=dict(event="startup_data_audit",selected_features=count,audio_bytes_hashed=audio_bytes,
                seconds=time.monotonic()-started,source_manifests=features["source_manifests"],
                policy="Every selected feature hash/id/text/token sequence/audio hash/shape/pitch/mel checked before updates")
    atomic_json(Path(output)/"startup-data-audit.json",result)
    print(json.dumps(result),flush=True)


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def train(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open(root / "training.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()) + "\n"); lock.flush()
    cls, loss_cls, bin_cls, model_config = imports(args.source)
    features = json.loads((Path(args.features)/"features.json").read_text())
    if features["vocab"] != VOCAB or features["mel"] != MEL_CONFIG:
        raise ValueError("Prepared vocabulary/mel definition differs from runner")
    train_rows = rows(Path(args.features)/"train.jsonl")
    val_rows = rows(Path(args.features)/"validation.jsonl")
    for name,items in (("train",train_rows),("validation",val_rows)):
        ids=[r["id"] for r in items]
        if len(ids)!=len(set(ids)):raise ValueError(f"Repeated IDs in {name} feature manifest; refuse cached-feature races")
    # Split integrity is checked before any update.
    train_ids = {r["id"] for r in train_rows}
    if train_ids & {r["id"] for r in val_rows}:
        raise ValueError("Train/validation overlap")
    tg = {r["group_id"] for r in train_rows if r.get("group_id")}
    if tg & {r["group_id"] for r in val_rows if r.get("group_id")}:
        raise ValueError("Train/validation group overlap")
    audit_features(features,train_rows+val_rows,root)
    if args.tiny:
        model_config.update(symbols_embedding_dim=64, in_fft_output_size=64, out_fft_output_size=64,
                            in_fft_conv1d_filter_size=128, out_fft_conv1d_filter_size=128,
                            in_fft_n_layers=2, out_fft_n_layers=2,
                            dur_predictor_filter_size=64, pitch_predictor_filter_size=64)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    model = cls(**model_config).to(device)
    model.pitch_mean.fill_(features["pitch_mean"])
    model.pitch_std.fill_(features["pitch_std"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9,.98), eps=1e-8, weight_decay=1e-6)
    lr_state=dict(multiplier=1.,reductions=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step+1)/args.warmup)*lr_state["multiplier"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    precision = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    criterion = loss_cls(dur_predictor_loss_scale=.1, pitch_predictor_loss_scale=.1)
    binary = bin_cls()
    checkpoint_path = root / "last.pt"
    step, epoch, cursor, best, bad = 0, 0, 0, float("inf"), 0
    config = dict(model=model_config, features=features, features_manifest_sha256={
            n:digest(Path(args.features)/(n+".jsonl")) for n in ("train","validation")},
            seed=args.seed, batch_size=args.batch_size, accumulation=args.accumulation,
            precision=args.precision, lr=args.lr, warmup=args.warmup)
    config["alignment_precision"]="FP32 cumulative duration boundaries and pitch sums; acoustic outputs retain selected dtype"
    config["objective"]=dict(mel=1., duration=.1, pitch=.1, attention_ctc=1., energy=.1, gradient_clip=1000.)
    config["alignment_schedule"]=dict(binary_start=args.binary_start,binary_warmup=args.binary_warmup)
    config["selection"]=dict(validate_every=args.validate_every,validation_limit=args.validation_limit,
                             patience=args.patience,min_delta=args.min_delta,
                             plateau_lr_factor=args.plateau_lr_factor,max_lr_reductions=args.max_lr_reductions)
    if args.init_checkpoint:
        if args.resume:
            # The initializer remains provenance; --resume restores full run state.
            init = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]["initialization"]
            if digest(args.init_checkpoint) != init["sha256"]:
                raise ValueError("Initialization object changed since first run")
        else:
            initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            if initial["config"]["model"] != model_config:
                raise ValueError("Initialization architecture differs")
            previous_features = initial["config"]["features"]
            if previous_features["vocab"] != features["vocab"] or previous_features["mel"] != features["mel"]:
                raise ValueError("Initialization vocabulary/mel definition differs")
            model.load_state_dict(initial["model"], strict=True)
            model.pitch_mean.fill_(features["pitch_mean"])
            model.pitch_std.fill_(features["pitch_std"])
            init = dict(path=str(Path(args.init_checkpoint).resolve()), sha256=digest(args.init_checkpoint),
                        source_step=initial["step"], source_config=initial["config"],
                        state_policy="strict model; fresh optimizer/scheduler/scaler/RNG/data position")
        config["initialization"] = init
    if args.resume:
        # Our private, locally written checkpoint contains RNG tuples/arrays.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt["config"] != config:
            previous=ckpt["config"].copy(); current=config.copy()
            resource_keys=("batch_size","accumulation","precision")
            changes={k:dict(before=previous.pop(k),after=current.pop(k)) for k in resource_keys}
            if not args.resume_allow_resource_change or previous != current:
                raise ValueError("Resume configuration/data mismatch; resource-only changes require explicit flag")
            atomic_json(root / f"resume-resource-change-{ckpt['step']}.json",dict(changes=changes,
                        checkpoint_sha256=digest(checkpoint_path), utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()))
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_state.update(ckpt.get("lr_state",{}))
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt["scaler"] and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        step, epoch, cursor, best, bad = [ckpt[k] for k in ("step","epoch","cursor","best","bad")]
        restore_rng(ckpt["rng"], device)
        print(json.dumps(dict(event="resumed", step=step, epoch=epoch, cursor=cursor)), flush=True)
    provenance=dict(code_sha256=digest(__file__), source_manifest_sha256=digest(Path(args.source)/"UPSTREAM.json"),
                    torch=torch.__version__, cuda=torch.version.cuda, upstream=UPSTREAM,
                    argv=sys.argv, utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
    if args.resume and ckpt.get("provenance",{}).get("code_sha256") != provenance["code_sha256"]:
        atomic_json(root/f"resume-code-change-{step}.json",dict(before=ckpt.get("provenance"),after=provenance))
    atomic_json(root / "config.json", dict(**config, provenance=provenance, args=vars(args)))
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    last_save = time.monotonic()

    def save(is_best=False):
        nonlocal last_save
        if __import__("shutil").disk_usage(root).free < sum(p.numel()*p.element_size() for p in model.parameters())*5 + 512*1024**2:
            raise RuntimeError("Insufficient atomic checkpoint margin")
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), rng=rng_state(device), step=step, epoch=epoch, cursor=cursor,
                     best=best, bad=bad, config=config, provenance=provenance,lr_state=lr_state.copy())
        atomic_torch(checkpoint_path, state)
        # Link the exact selected object without another expanded checkpoint copy.
        if is_best:
            tmp = root / "best.pt.tmp"
            if tmp.exists(): tmp.unlink()
            os.link(checkpoint_path, tmp)
            os.replace(tmp, root / "best.pt")
        atomic_json(root/"checkpoint.json", dict(step=step, epoch=epoch, cursor=cursor,
                    checkpoint=str(checkpoint_path), sha256=digest(checkpoint_path), best=best))
        last_save = time.monotonic()

    def evaluate():
        model.eval()
        values = []
        diagnostics=[]
        mel_examples=[]
        max_rows = len(val_rows) if args.validation_limit == 0 else args.validation_limit
        with torch.no_grad():
            for row in val_rows[:max_rows]:
                x,y = sample_batch([load_feature(row)], device, features["pitch_mean"], features["pitch_std"])
                with torch.autocast(device.type, dtype=precision, enabled=args.precision != "fp32"):
                    out = model(x)
                # CTC is FP32 even with autocast model; cuDNN CTC excludes half.
                out = tuple(v.float() if torch.is_tensor(v) else v for v in out)
                loss, meta = criterion(out,y)
                if not torch.isfinite(loss):raise FloatingPointError("Non-finite validation; last checkpoint preserved")
                values.append({k:float(v) for k,v in meta.items()})
                pred_frames=int(torch.round(out[2]).sum())
                hard,soft=out[9],out[8]
                indices=soft[0,0].argmax(-1)
                diagnostics.append(dict(id=row["id"], predicted_frames=pred_frames, target_frames=int(y[2][0]),
                        duration_ratio=pred_frames/int(y[2][0]),
                        attention_max_probability=float(soft.amax(-1).mean()),
                        attention_mas_nll=float(binary(hard,soft)),
                        monotonic_argmax_fraction=float((indices[1:]>=indices[:-1]).float().mean()),
                        predicted_mel_std=float(out[0].std()),target_mel_std=float(y[0].std())))
                if len(mel_examples)<4:
                    example=dict(id=row["id"],mel_prediction=out[0].transpose(1,2).cpu(),
                                 target_mel=y[0].cpu(),
                                 note="mel_prediction uses teacher-forced MAS duration/pitch; not text-only synthesis")
                    if 4<=pred_frames<=max(512,int(y[2][0])*4):
                        with torch.autocast(device.type,dtype=precision,enabled=args.precision!="fp32"):
                            native,native_lens,*_=model.infer(x[0])
                        example["native_mel"]=native[:,:,:int(native_lens[0])].float().cpu()
                        example["native_frames"]=int(native_lens[0])
                        example["native_note"]="Text-only acoustic inference, no ground-truth duration or pitch"
                    else:
                        example["native_skipped"]="Zero/very short or excessive predicted duration; diagnostic failure preserved"
                    mel_examples.append(example)
        model.train()
        if not values:raise ValueError("No validation samples")
        result={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        result.update(zero_predicted_duration_cases=sum(d["predicted_frames"]==0 for d in diagnostics),
                      duration_ratio_mean=float(np.mean([d["duration_ratio"] for d in diagnostics])),
                      attention_mas_nll=float(np.mean([d["attention_mas_nll"] for d in diagnostics])))
        atomic_json(root/"validation-latest.json",dict(step=step,aggregate=result,diagnostics=diagnostics))
        atomic_torch(root/"validation-mels.pt",dict(step=step,examples=mel_examples))
        append_history(root/"validation-history.jsonl",dict(event="initial_validation" if step==0 else "validation",
                         step=step,**result,learning_rate=optimizer.param_groups[0]["lr"],lr_state=lr_state.copy(),
                         code_sha256=provenance["code_sha256"]))
        return result

    start_all=time.monotonic()
    amp_skips=0
    if not args.resume:
        initial=evaluate()
        best=initial["loss"]
        print(json.dumps(dict(event="initial_validation",step=0,**initial)),flush=True)
        save(True)
    while step < args.max_steps and not stop[0]:
        start=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        mean_loss=0.
        step_frames=0
        for micro in range(args.accumulation):
            items=[]
            for _ in range(args.batch_size):
                order = np.random.default_rng(args.seed+epoch).permutation(len(train_rows))
                if cursor >= len(order):
                    epoch+=1;cursor=0
                    order = np.random.default_rng(args.seed+epoch).permutation(len(train_rows))
                items.append(load_feature(train_rows[int(order[cursor])]))
                cursor+=1
            x,y=sample_batch(items,device,features["pitch_mean"],features["pitch_std"])
            step_frames+=int(y[2].sum())
            with torch.autocast(device.type, dtype=precision, enabled=args.precision != "fp32"):
                out=model(x)
            out=tuple(v.float() if torch.is_tensor(v) else v for v in out)
            loss,meta=criterion(out,y)
            # Binarization pressure follows the official warmup principle.
            kl_weight=min(1.,max(0.,(step-args.binary_start)/max(1,args.binary_warmup)))
            loss=loss+kl_weight*binary(out[9],out[8])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at next step {step+1}; last checkpoint preserved")
            scaler.scale(loss/args.accumulation).backward()
            mean_loss+=float(loss.detach())/args.accumulation
        scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1000.)
        if not torch.isfinite(grad) and not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients; last checkpoint preserved")
        before=model.proj.weight.detach().flatten()[:256].clone()
        old_scale=scaler.get_scale()
        scaler.step(optimizer);scaler.update()
        advanced=scaler.get_scale() >= old_scale
        if not advanced:
            amp_skips+=1
            print(json.dumps(dict(event="amp_skipped",next_step=step+1,scale=scaler.get_scale(),
                                  epoch=epoch,cursor=cursor,consecutive=amp_skips)),flush=True)
            if amp_skips>=16:
                raise FloatingPointError("Persistent AMP overflow; last checkpoint preserved")
            continue
        amp_skips=0
        scheduler.step();step+=1
        if device.type=="cuda":torch.cuda.synchronize()
        change=float((model.proj.weight.detach().flatten()[:256]-before).abs().max())
        receipt=dict(event="step",step=step,epoch=epoch,cursor=cursor,loss=mean_loss,
                     seconds=time.monotonic()-start,learning_rate=optimizer.param_groups[0]["lr"],
                     gradient_norm=float(grad),weight_delta=change,
                     samples=args.batch_size*args.accumulation,audio_frames=step_frames,
                     audio_seconds=step_frames*256/24000,
                     vram_peak_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                     elapsed=time.monotonic()-start_all)
        print(json.dumps(receipt),flush=True)
        atomic_json(root/"progress.json",receipt)
        if step==1 or step%10==0:
            append_history(root/"steps-history.jsonl",receipt)
        if step % args.validate_every==0:
            val=evaluate();val["event"]="validation";val["step"]=step
            print(json.dumps(val),flush=True)
            score=val["loss"]
            improved=score < best-args.min_delta
            if improved:best=score;bad=0
            else:bad+=1
            save(improved)
            # No early stop until alignment warmup has finished.
            if bad>=args.patience and step>=args.binary_start+args.binary_warmup:
                if lr_state["reductions"]<args.max_lr_reductions:
                    lr_state["multiplier"]*=args.plateau_lr_factor
                    lr_state["reductions"]+=1
                    bad=0
                    for group,base in zip(optimizer.param_groups,scheduler.base_lrs):
                        group["lr"]=base*min(1.,(scheduler.last_epoch+1)/args.warmup)*lr_state["multiplier"]
                    scheduler._last_lr=[g["lr"] for g in optimizer.param_groups]
                    print(json.dumps(dict(event="learning_rate_reduced",step=step,**lr_state,
                                          learning_rate=optimizer.param_groups[0]["lr"])),flush=True)
                    save()
                    continue
                print(json.dumps(dict(event="plateau",step=step,patience=bad)),flush=True)
                break
        elif time.monotonic()-last_save>=args.checkpoint_seconds:
            save()
    save()
    print(json.dumps(dict(event="stopped",step=step,signal=stop[0],checkpoint=str(checkpoint_path))),flush=True)


def synthesize(args):
    cls,_,_,_=imports(args.source)
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=cls(**ckpt["config"]["model"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"],strict=True)
    text=Path(args.text_file).read_text().strip() if args.text_file else args.text
    seq=tokens(text,ckpt["config"]["features"]["vocab"])[None].to(args.device)
    start=time.monotonic()
    with torch.inference_mode():
        mel,lens,*_=model.infer(seq)
    if not int(lens[0]):raise RuntimeError("Zero predicted duration; model not ready for synthesis")
    mel=mel[:,:,:int(lens[0])]
    if args.device.startswith("cuda"):torch.cuda.synchronize()
    acoustic_seconds=time.monotonic()-start
    if args.bigvgan_dir:
        import soundfile as sf
        d=Path(args.bigvgan_dir)
        sys.path.insert(0,str(d/"source"))
        from bigvgan import BigVGAN
        from env import AttrDict
        vocoder=BigVGAN(AttrDict(json.loads((d/"config.json").read_text())),use_cuda_kernel=False)
        state=torch.load(d/"bigvgan_generator.pt",map_location="cpu",weights_only=True)
        vocoder.load_state_dict(state["generator"],strict=True)
        vocoder=vocoder.to(args.device).eval();vocoder.remove_weight_norm()
        begin=time.monotonic()
        with torch.inference_mode():audio=vocoder(mel).squeeze().cpu().numpy()
        if not np.isfinite(audio).all():raise ValueError("Nonfinite synthesis")
        sf.write(args.output,audio,24000,subtype="FLOAT")
        atomic_json(str(args.output)+".json",dict(acoustic_seconds=acoustic_seconds,vocoder_seconds=time.monotonic()-begin,
                    duration=len(audio)/24000,peak=float(np.abs(audio).max()),frames=int(lens[0]),step=ckpt["step"],
                    note="Full-utterance synthesis; no hypothetical streaming",sha256=digest(args.output)))
    else:
        atomic_torch(args.output,dict(mel=mel.cpu(),text=text,frames=int(lens[0]),seconds=acoustic_seconds))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--manifests",nargs="+",required=True);prep.add_argument("--output",required=True);prep.add_argument("--limit",type=int,default=0)
    prep.add_argument("--pitch-method",choices=["pyin","world"],default="world")
    prep.add_argument("--workers",type=int,default=4)
    prep.add_argument("--exclude-id",nargs="*",default=[],help="Explicit IDs omitted from feature selection; originals unchanged")
    tr=sub.add_parser("train");tr.add_argument("--features",required=True);tr.add_argument("--output",required=True)
    tr.add_argument("--source",default=DEFAULT_SOURCE);tr.add_argument("--device",default="cuda");tr.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp16")
    tr.add_argument("--batch-size",type=int,default=1);tr.add_argument("--accumulation",type=int,default=4);tr.add_argument("--lr",type=float,default=2e-4)
    tr.add_argument("--max-steps",type=int,default=200000);tr.add_argument("--warmup",type=int,default=1000);tr.add_argument("--seed",type=int,default=20261103)
    tr.add_argument("--validate-every",type=int,default=500);tr.add_argument("--validation-limit",type=int,default=0);tr.add_argument("--checkpoint-seconds",type=int,default=1800)
    tr.add_argument("--binary-start",type=int,default=1000);tr.add_argument("--binary-warmup",type=int,default=5000);tr.add_argument("--patience",type=int,default=12);tr.add_argument("--min-delta",type=float,default=.01)
    tr.add_argument("--plateau-lr-factor",type=float,default=.3);tr.add_argument("--max-lr-reductions",type=int,default=1)
    tr.add_argument("--resume",action="store_true");tr.add_argument("--tiny",action="store_true",help="Only CPU engineering smoke; never compare as full FastPitch")
    tr.add_argument("--init-checkpoint",help="Strict model-only initialization for a new stage; optimizer and counters start fresh")
    tr.add_argument("--resume-allow-resource-change",action="store_true",help="Allow only recorded microbatch/accumulation/precision changes while restoring optimizer/RNG/data state")
    syn=sub.add_parser("synthesize");syn.add_argument("--source",default=DEFAULT_SOURCE);syn.add_argument("--checkpoint",required=True)
    syn.add_argument("--text");syn.add_argument("--text-file");syn.add_argument("--output",required=True);syn.add_argument("--device",default="cuda");syn.add_argument("--bigvgan-dir")
    args=p.parse_args();torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
    if args.command=="prepare":prepare(args)
    elif args.command=="train":train(args)
    else:synthesize(args)


if __name__=="__main__":main()
