#!/usr/bin/env python3
"""Read-only, sanitized provenance for the existing Aegis TTS container.

Run on the host with ``--container aegis-tts`` (the default), or inside the
existing container with ``--inside``. No model is imported or instantiated.
Private configuration is reduced to an explicit allowlist of numerical and
boolean runtime settings; references, transcripts, environment and logs are
never included. Output is JSON suitable for a private operational snapshot.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


PACKAGES = ("vllm", "vllm-omni", "torch", "transformers", "numpy", "scipy", "pyworld")
SOURCE_FILES = (
    "entrypoints/openai/protocol/audio.py",
    "entrypoints/openai/serving_speech.py",
    "model_executor/models/qwen3_tts/prompt_embeds_builder.py",
    "model_executor/models/qwen3_tts/qwen3_tts_code2wav.py",
    "model_executor/stage_input_processors/qwen3_tts.py",
)
NUMERIC_SETTINGS = {
    "async_chunk", "codec_streaming", "codec_chunk_frames", "codec_left_context_frames",
    "initial_codec_chunk_frames", "ref_code_context_frames", "max_num_seqs",
    "max_num_batched_tokens", "max_model_len", "gpu_memory_utilization",
    "enforce_eager", "enable_prefix_caching", "async_scheduling",
    "connector_get_sleep_s", "connector_get_max_wait_first_chunk", "connector_get_max_wait",
    "shm_threshold_bytes", "decode_chunk_frames", "decode_left_context_frames",
    "decode_cudagraph_capture_sizes", "decode_cudagraph_batch_sizes",
    "decode_cudagraph_extra_capture_shapes", "decode_compile_shapes",
    "decode_batch_bucket_frames", "decode_batch_max_size", "decode_enable_tf32",
    "temperature", "top_k", "top_p", "min_tokens", "max_tokens", "seed",
    "repetition_penalty", "do_sample", "non_streaming_mode",
}


def numeric(value: object) -> bool:
    if type(value) in (bool, int, float):
        return True
    return isinstance(value, (list, tuple)) and all(numeric(item) for item in value)


def selected_settings(value: object, path: str = "") -> list[dict]:
    result = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in NUMERIC_SETTINGS and numeric(child):
                result.append({"path": child_path, "value": child})
            else:
                result.extend(selected_settings(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(selected_settings(child, f"{path}[{index}]"))
    return result


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def parsed(path: Path) -> ast.Module | None:
    return ast.parse(path.read_text()) if path.is_file() else None


def protocol_fields(tree: ast.Module | None) -> set[str]:
    if tree is None:
        return set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenAICreateSpeechRequest":
            return {child.target.id for child in node.body
                    if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)}
    return set()


def gateway_q00(tree: ast.Module | None) -> dict:
    if tree is None:
        return {}
    for node in tree.body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "Q00_PARAMS"):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                return {}
            return {key: value for key, value in values.items()
                    if key in NUMERIC_SETTINGS and numeric(value)}
    return {}


def base_prompt_default(tree: ast.Module | None) -> bool | None:
    """Recognize the installed builder's explicit task-dependent default only."""
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Compare):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "non_streaming_mode"
                   for target in node.targets):
            continue
        expression = node.value
        if (isinstance(expression.left, ast.Name) and expression.left.id == "task_type"
                and len(expression.ops) == 1 and isinstance(expression.ops[0], ast.In)
                and len(expression.comparators) == 1):
            try:
                alternatives = ast.literal_eval(expression.comparators[0])
            except (ValueError, TypeError):
                continue
            if isinstance(alternatives, (list, tuple)) and all(isinstance(x, str) for x in alternatives):
                return "Base" in alternatives
    return None


def inside(deploy_config: Path, gateway_root: Path) -> dict:
    versions, roots = {}, {}
    for package in PACKAGES:
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = distribution.version
            if package == "vllm-omni":
                roots["omni"] = Path(distribution.locate_file("vllm_omni"))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = roots.get("omni")
    source_hashes = {name: file_digest(root / name) for name in SOURCE_FILES} if root else {}
    fields = protocol_fields(parsed(root / SOURCE_FILES[0])) if root else set()
    q00 = gateway_q00(parsed(gateway_root / "default_reference_gateway.py"))
    base_default = base_prompt_default(parsed(root / SOURCE_FILES[2])) if root else None
    settings, config_error = [], None
    try:
        import yaml
        settings = selected_settings(yaml.safe_load(deploy_config.read_text()))
    except Exception as error:
        # YAML parse errors may contain excerpts of the private input. Only
        # retain the exception type, never its text or a traceback.
        config_error = type(error).__name__
    has_field = "non_streaming_mode" in fields if fields else None
    return {
        "schema": "aegis-tts-runtime-inspection-v1",
        "packages": versions,
        "installed_omni_source_sha256": source_hashes,
        "gateway_source_sha256": {
            name: file_digest(gateway_root / name)
            for name in ("default_reference_gateway.py", "voice_pipeline.py", "aegis_speech/service.py")
        },
        "gateway_q00_numeric_settings": q00,
        "deploy_selected_numeric_settings": settings,
        "deploy_read_error": config_error,
        "api_schema": {
            "initial_codec_chunk_frames_supported": "initial_codec_chunk_frames" in fields if fields else None,
            "non_streaming_mode_field_supported": has_field,
            "top_level_temperature_field_supported": "temperature" in fields if fields else None,
        },
        "base_conditioning": {
            "gateway_requested_non_streaming_mode": q00.get("non_streaming_mode"),
            "installed_source_base_default_non_streaming_mode": base_default,
            "gateway_flag_absent_from_api_schema": has_field is False,
            "note": "Source inspection; no request or model execution. Preserve the running conditioning when optimizing latency.",
        },
    }


def docker_json(arguments: list[str]) -> object:
    completed = subprocess.run(["docker", *arguments], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(f"docker inspection failed (exit {completed.returncode})")
    return json.loads(completed.stdout)


def on_host(container: str, deploy_config: Path, gateway_root: Path) -> dict:
    inspected = docker_json(["inspect", container])[0]
    image = docker_json(["image", "inspect", inspected["Image"]])[0]
    completed = subprocess.run([
        "docker", "exec", "-i", container, "python3", "-", "--inside",
        "--deploy-config", str(deploy_config), "--gateway-root", str(gateway_root),
    ], input=Path(__file__).read_bytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(f"container runtime inspection failed (exit {completed.returncode})")
    result = json.loads(completed.stdout)
    labels = inspected.get("Config", {}).get("Labels", {})
    image_labels = image.get("Config", {}).get("Labels", {})
    result["container"] = {
        "id": inspected["Id"], "image_id": inspected["Image"],
        "created": inspected.get("Created"),
        "running": inspected.get("State", {}).get("Running"),
        "compose_project": labels.get("com.docker.compose.project"),
        "compose_service": labels.get("com.docker.compose.service"),
    }
    result["image"] = {
        "id": image["Id"], "repo_digests": image.get("RepoDigests", []),
        "source_revision": image_labels.get("org.opencontainers.image.revision"),
        "version": image_labels.get("org.opencontainers.image.version"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="aegis-tts")
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--deploy-config", type=Path, default=Path("/protected/config/q00-vllm-deploy-exact.yaml"))
    parser.add_argument("--gateway-root", type=Path, default=Path("/gateway"))
    args = parser.parse_args()
    result = inside(args.deploy_config, args.gateway_root) if args.inside else on_host(
        args.container, args.deploy_config, args.gateway_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": type(error).__name__, "operation": "read-only-runtime-inspection"}), file=sys.stderr)
        raise SystemExit(1)
