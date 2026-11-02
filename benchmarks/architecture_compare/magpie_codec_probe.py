#!/usr/bin/env python3
"""CPU-only NanoCodec numeric roundtrip; never loads the Magpie TTS runtime.

The hash-pinned NVIDIA module definitions execute unchanged, except that NeMo
typecheck and TorchScript decorators are removed and NeuralModule is nn.Module.
This deliberately avoids NeMo training/discriminator/ASR dependencies. It is not
a validation of the complete NeMo runtime. No network, CUDA or training path.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

SPEECH_REVISION = "e5d79d5a49cbec49d79440b0070aac0c32a8dbb8"
CODEC_REVISION = "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7"
HASHES = {
    "nemo_collections_tts_modules_audio_codec_modules.py": "2da7857e760a18bc71b6d7c6f3a8906a7f37e290513275cf667c51b2a96ea03c",
    "nemo_collections_common_parts_utils.py": "3c5905994143e715087724c648a215428971a0c813a98c96a65ffdf9d7a32bec",
    "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo": "28c2518de3e3d5a2c7d9bca40a7ebc0644eb76c60b890970365325bdd8e9f099",
}
CONFIG_HASH = "9527829cc3d19cb4f34bbfb4eb63b4c8567ddc4d85d6cc0c11e305b3d768c681"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def verified_files(root):
    for name, expected in HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(f"unverified official artifact: {name}")


class MetadataOnly(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.decorator_list = [d for d in node.decorator_list
                               if ast.unparse(d).split("(")[0] not in {"typecheck", "torch.jit.script"}]
        return node


def numeric_definitions(path, names, namespace):
    """Select exact definitions plus their in-file dependencies, in source order."""
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    selected = set(names)
    while True:
        dependencies = {n.id for name in selected for n in ast.walk(definitions[name])
                        if isinstance(n, ast.Name) and n.id in definitions}
        if dependencies <= selected:
            break
        selected.update(dependencies)
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [MetadataOnly().visit(node) for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return sorted(selected)


def execute(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' explicitly; this probe only permits CPU")
    root, original, output = args.artifact_dir.resolve(), args.audio.resolve(), args.output.resolve()
    verified_files(root)  # Verification precedes any imports or deserialization.
    for parent in [output, *output.parents]:
        if (parent / ".git").exists():
            raise ValueError("private outputs must be outside a Git checkout")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.chmod(output, 0o700)

    import logging
    import math
    from abc import ABC, abstractmethod
    from contextlib import contextmanager
    import einops
    import numpy as np
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import yaml

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(40733133)
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, np=np,
                     math=math, einops=einops, rearrange=einops.rearrange,
                     ABC=ABC, abstractmethod=abstractmethod, contextmanager=contextmanager,
                     NeuralModule=torch.nn.Module, logging=logging)
    common = numeric_definitions(root / "nemo_collections_common_parts_utils.py",
                                 ["mask_sequence_tensor", "ClampActivation", "Snake", "HalfSnake"], namespace)
    classes = numeric_definitions(root / "nemo_collections_tts_modules_audio_codec_modules.py",
                                  ["HiFiGANEncoder", "CausalHiFiGANDecoder", "GroupFiniteScalarQuantizer"], namespace)
    start = time.perf_counter()
    with tarfile.open(root / "nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo", "r:") as archive:
        config_bytes = archive.extractfile("./model_config.yaml").read()
        if hashlib.sha256(config_bytes).hexdigest() != CONFIG_HASH:
            raise ValueError("unexpected NanoCodec configuration")
        config = yaml.safe_load(config_bytes)
        # The whole archive was authenticated above. weights_only=True does not
        # fall back to pickle execution, even for a verified NVIDIA checkpoint.
        state = torch.load(io.BytesIO(archive.extractfile("./model_weights.ckpt").read()),
                           map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    components, loaded = {}, {}
    for name in ["audio_encoder", "vector_quantizer", "audio_decoder"]:
        cfg = dict(config[name])
        class_name = cfg.pop("_target_").split(".")[-1]
        model = namespace[class_name](**cfg).eval()
        local = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
        result = model.load_state_dict(local, strict=True)
        loaded[name] = {"keys": len(local), "parameters": sum(v.numel() for v in model.parameters()),
                        "missing": result.missing_keys, "unexpected": result.unexpected_keys}
        components[name] = model
    load_seconds = time.perf_counter() - start
    del state

    shutil.copyfile(original, output / "A-original.wav")
    audio, rate = sf.read(original, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("select an authorized mono reference; no implicit channel mixing")
    audio = audio[:, 0]
    target_rate = config["sample_rate"]
    start = time.perf_counter()
    divisor = math.gcd(rate, target_rate)
    converted = resample_poly(audio, target_rate // divisor, rate // divisor,
                              window=("kaiser", 10.0)).astype(np.float32)
    conversion_seconds = time.perf_counter() - start
    # Float WAV retains the actual numerical input, including any resampler
    # overshoot. There is no gain normalization, trim, limiter or denoiser.
    sf.write(output / "B-input-22050-float.wav", converted, target_rate, subtype="FLOAT")
    tensor = torch.from_numpy(converted).unsqueeze(0)
    length = torch.tensor([tensor.shape[-1]])
    frame = config["samples_per_frame"]
    padded_length = ((length + frame - 1) // frame) * frame
    tensor = torch.nn.functional.pad(tensor, (0, int(padded_length[0]) - tensor.shape[-1]))
    timings = {}
    with torch.inference_mode():
        start = time.perf_counter()
        encoded, encoded_len = components["audio_encoder"](audio=tensor, audio_len=padded_length)
        timings["encode_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        tokens = components["vector_quantizer"].encode(inputs=encoded, input_len=encoded_len)
        timings["quantize_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        dequantized = components["vector_quantizer"].decode(indices=tokens, input_len=encoded_len)
        timings["dequantize_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        reconstructed, reconstructed_len = components["audio_decoder"](inputs=dequantized, input_len=encoded_len)
        timings["decode_seconds"] = time.perf_counter() - start
    samples = reconstructed[0, :int(reconstructed_len[0])].numpy()
    if not np.isfinite(samples).all():
        raise ValueError("nonfinite codec output")
    sf.write(output / "C-roundtrip-22050-float.wav", samples, target_rate, subtype="FLOAT")
    # PCM16 copies support the existing offline A/B packer, with explicit
    # quantization only. Preserve the native floats above for signal analysis.
    sf.write(output / "B-input-22050.wav", converted, target_rate, subtype="PCM_16")
    sf.write(output / "C-roundtrip-22050.wav", samples, target_rate, subtype="PCM_16")
    def stats(x):
        return {"peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x.astype(float) ** 2))),
                "samples_at_or_above_fullscale": int(np.count_nonzero(np.abs(x) >= 1.0))}
    receipt = {"probe": "NanoCodec CPU numeric modules; not full NeMo runtime", "codec_revision": CODEC_REVISION,
               "speech_revision": SPEECH_REVISION, "artifact_hashes": HASHES, "config_sha256": CONFIG_HASH,
               "torch": torch.__version__, "cuda_used": False, "threads": args.threads,
               "common_definitions": common, "codec_definitions": classes,
               "harness_changes": ["NeuralModule replaced with nn.Module", "typecheck decorators omitted",
                                   "TorchScript decorator omitted; eager formulas unchanged"],
               "strict_load": loaded, "source_rate_hz": rate, "target_rate_hz": target_rate,
               "source_seconds": len(audio) / rate, "input_samples": len(converted),
               "padded_input_samples": int(padded_length[0]), "output_samples": len(samples),
               "tokens_shape_codebook_batch_time": list(tokens.shape), "load_seconds": load_seconds,
               "conversion_seconds": conversion_seconds, **timings,
               "roundtrip_rtf_cpu": sum(timings.values()) / (len(converted) / target_rate),
               "statistics": {"A": stats(audio), "B": stats(converted), "C": stats(samples)},
               "audio_sha256": {p.name: digest(p) for p in sorted(output.glob("*.wav"))},
               "original_source_sha256": digest(original), "training_executed": False,
               "auditory_review": "pending owner; metrics do not certify robotic effect retention"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for p in output.iterdir():
        p.chmod(0o600)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "plan", "cuda": False, "network": False, "training": False,
                          "codec_revision": CODEC_REVISION, "speech_revision": SPEECH_REVISION,
                          "required_hashes": HASHES}, indent=2))
        return 0
    if any(v is None for v in (args.artifact_dir, args.audio, args.output)):
        parser.error("--execute requires --artifact-dir, --audio and --output")
    print(json.dumps(execute(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
