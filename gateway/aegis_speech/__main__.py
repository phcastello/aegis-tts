"""Small offline operational commands; these never synthesize or expose the reference."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

from .lexicon import PronunciationLexicon
from .normalization import SpeechTextNormalizer
from .segmentation import SpeechSegmenter

ROOT = Path(__file__).resolve().parents[2]
PROTECTED = ROOT / "protected/AegisVoicev1.0"


def validate_profile() -> int:
    manifest = PROTECTED / "provenance/SHA256SUMS"
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.strip().removeprefix("./")] = digest
    required = ["reference/Aninha_AegisVoice4.wav", "reference/Aninha_AegisVoice4.txt", "config/profile.yaml", "config/q00-vllm-deploy-exact.yaml", "config/qwen-selected-p07-m02-v1.0.yaml", "pronunciation/pt-BR.yaml"]
    for name in required:
        target = PROTECTED / name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entries.get(name):
            raise SystemExit(f"invalid protected asset: {name}")
    profile = yaml.safe_load((PROTECTED / "config/profile.yaml").read_text())
    if profile.get("profile") != "AegisVoicev1.0" or profile.get("sample_rate") != 24000 or profile.get("channels") != 1:
        raise SystemExit("invalid protected profile")
    print("AegisVoicev1.0 profile valid")
    return 0


def lexicon() -> PronunciationLexicon:
    return PronunciationLexicon(PROTECTED / "pronunciation/pt-BR.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m aegis_speech")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate-profile", "validate-lexicon", "run-pronunciation-corpus", "run-golden-regression"):
        sub.add_parser(command)
    for command in ("normalize-text", "segment-text"):
        child = sub.add_parser(command); child.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "validate-profile": return validate_profile()
    if args.command == "validate-lexicon":
        value = lexicon(); print(f"lexicon version={value.version} sha256={value.sha256}"); return 0
    if args.command in {"normalize-text", "segment-text"}:
        result = SpeechTextNormalizer(lexicon().apply).normalize(args.text)
        if args.command == "normalize-text": print(result.normalized_text)
        else:
            for item in SpeechSegmenter().segment(result): print(f"{item.segment_index}\t{item.split_reason}\t{item.normalized_text}")
        return 0
    if args.command == "run-pronunciation-corpus":
        normalizer = SpeechTextNormalizer(lexicon().apply)
        failures = [case["id"] for case in yaml.safe_load((ROOT / "config/pronunciation/corpus.yaml").read_text())["cases"] if normalizer.normalize(case["input"]).normalized_text != case["expected_normalized"]]
        if failures: raise SystemExit("corpus failures: " + ", ".join(failures))
        print("pronunciation corpus passed"); return 0
    return validate_profile()


if __name__ == "__main__":
    raise SystemExit(main())
