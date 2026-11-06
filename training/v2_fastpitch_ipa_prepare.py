"""Prepare immutable IPA token sidecars while reusing qualified acoustic caches.

Default is metadata/plan only. --prepare needs an exact reviewed plan hash and a
new private output directory. No WAV or acoustic .pt is opened or rewritten:
their qualified expected hashes are carried forward for the trainer's complete
SHA/tensor audit before it replaces only the in-memory grapheme token tensor.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

from training import v2_fastpitch_ipa_frontend as front

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/var/tmp/aegis-v2-work")
INPUT = ROOT / "portuguese/fastpitch-input-v1"
FEATURES = ROOT / "fastpitch/portuguese-features-clean-v1"
BASELINE = ROOT / "fastpitch/common-cpu-language-control-step10000/manifest.json"
DEFAULT_OUTPUT = ROOT / "fastpitch/ipa-public-sidecars-v1"
SCHEMA = "aegis.fastpitch.ipa.sidecars.v1"
ROW_SCHEMA = "aegis.fastpitch.ipa.phone-row.v1"
INDEX_SCHEMA = "aegis.fastpitch.ipa.row-index.v1"
PLAN_SCHEMA = "aegis.fastpitch.ipa.sidecar-plan.v1"
COUNTS = {"train": 3037, "validation": 172, "probes": 11}
PUBLIC_IDS = tuple("sample-" + str(i) for i in range(1050, 1054))
COMMON_IDS = ("v_command", "v_reply", "v_pressure", "v_acronyms", "v_question", "v_sibilants", "v_long")
SOURCE_EXPECTED = {
    str(INPUT / "train.jsonl"): "2cc96bcd9b180ddaa6f4ef97b73a732d8ede837a477369b665c99bd5628c5f75",
    str(INPUT / "validation.jsonl"): "567075e3a1d8e99458481fa18876773b05479f3068447b94aea278294234d79c",
    str(FEATURES / "train.jsonl"): "d27a53223ac41bd3cabb429702899c805eb33b89b0fa99627f2717f317cb6011",
    str(FEATURES / "validation.jsonl"): "855fe2ae9600320812ef727d733395537388a124d38869201806441219fa3402",
    str(FEATURES / "features.json"): "0b73f4be74ed89af62193aa8db627d3256f8c4035bfd735024546c7efa6d288f",
    str(REPO / "training/fastpitch_experiment.py"): "7e7c94d8106b2e7729df5e238195f554dbcb5a494a3c5ee179f7ef8ee1a90709",
    str(REPO / "training/f5_refine_text.py"): "9528bcb5b9a41f68778a4f9ae0956af9edebc6c0365b604ea60dcbbcc6734c12",
    str(REPO / "gateway/aegis_speech/normalization.py"): "01fbd3eb902637479ae6022bdac65b6a54b9a2ffd85f8628323034237e3c9ee0",
    str(REPO / "benchmarks/v2/validation.pt-BR.json"): "16c54bbc362229f0edde145506d3b79db451c65f61ca98d69806ebf8d975a247",
}
OVERRIDE = dict(fresh_model=True, pretrained_grapheme_weights_reused=False,
                immutable_acoustic_cache=True, reuse_only="mel and pitch after original cache SHA/tensor audit",
                original_grapheme_ids_preserved=True, token_tensor_override="copy dictionary in RAM; replace text only",
                audio_read_by_preparer=False, acoustic_payload_read_by_preparer=False,
                expected_payload_hashes="Qualified manifests; trainer MUST verify actual audio/feature SHA and old glyph tensor before override")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(value):
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ValueError("SHA256 must be64 lowercase hex digits")
    return value


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError("positive integer required: " + name)
    return value


def _atomic(path, payload):
    path = Path(path)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def write_json(path, value):
    _atomic(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def _relative(base, name):
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("sidecar/index path must stay relative to the manifest")
    path = Path(base) / name
    if path.is_symlink() or path.resolve().is_relative_to(Path(base).resolve()) is False:
        raise ValueError("sidecar/index escaped the manifest directory")
    return path


def _source_pins():
    pins = dict(SOURCE_EXPECTED)
    for path, expected in pins.items():
        if front.sha256(path) != expected:
            raise ValueError("Frozen source metadata changed: " + path)
    pins[str(BASELINE)] = front.sha256(BASELINE)
    pins[str(Path(__file__))] = front.sha256(__file__)
    pins.update(front.frontend_metadata()["source_pins"])
    return pins


def _check_source_pins(pins):
    # Pins are metadata/code/native-G2P resources, never a WAV/acoustic/weight payload.
    if not isinstance(pins, dict) or not pins:
        raise ValueError("nonempty source pins required")
    allowed = set(SOURCE_EXPECTED) | {str(BASELINE), str(Path(__file__))} | set(front.frontend_metadata()["source_pins"])
    if set(pins) != allowed:
        raise ValueError("source metadata/resource pin inventory changed")
    for path, expected in pins.items():
        _sha(expected)
        if front.sha256(path) != expected:
            raise ValueError("Source pin mismatch: " + path)


def legacy_normalizer():
    path = REPO / "training/fastpitch_experiment.py"
    if front.sha256(path) != SOURCE_EXPECTED[str(path)]:
        raise ValueError("Original grapheme trainer changed")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalize")
    scope = {"unicodedata": unicodedata}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path) + "::normalize-only", "exec"), scope)
    return scope["normalize"]


def glyph_ids(text, vocab, normalize):
    normalized = normalize(text)
    lut = {symbol: index for index, symbol in enumerate(vocab)}
    missing = sorted(set(" " + normalized + " ") - set(lut))
    if missing:
        raise ValueError("Old grapheme source is unsupported: " + repr(missing))
    return normalized, [lut[c] for c in " " + normalized + " "]


def validate_partitions(parts):
    seen_ids = set()
    shared = {"audio": {}, "group": {}, "text": {}}
    for split in ("train", "validation"):
        rows = parts[split]
        if not rows:
            raise ValueError("empty partition")
        for row in rows:
            ident = row["id"]
            if not isinstance(ident, str) or ID_RE.fullmatch(ident) is None or ident in seen_ids:
                raise ValueError("duplicate/unsafe ID across public train/VAL")
            seen_ids.add(ident)
            text = " ".join(unicodedata.normalize("NFC", row["text"]).casefold().split())
            for label, value in (("audio", row["sha256"]), ("group", row["group_id"]), ("text", text)):
                previous = shared[label].setdefault(value, split)
                if previous != split:
                    raise ValueError("train/VAL leakage: " + label)


def _bindings():
    return dict(inputs={s: str(INPUT / (s + ".jsonl")) for s in ("train", "validation")},
                features={s: str(FEATURES / (s + ".jsonl")) for s in ("train", "validation")},
                feature_config=str(FEATURES / "features.json"), raw_feature_root=str(FEATURES))


def _public_rows():
    parts = {s: read_jsonl(INPUT / (s + ".jsonl")) for s in ("train", "validation")}
    validate_partitions(parts)
    config = read_json(FEATURES / "features.json")
    normalize = legacy_normalizer()
    result = {}
    for split in ("train", "validation"):
        features = read_jsonl(FEATURES / (split + ".jsonl"))
        if len(parts[split]) != COUNTS[split] or len(features) != COUNTS[split]:
            raise ValueError("Qualified public count changed")
        if [r["id"] for r in parts[split]] != [r["id"] for r in features]:
            raise ValueError("Input/acoustic manifest order or IDs changed")
        rows = []
        for source, feature in zip(parts[split], features):
            normalized, original_ids = glyph_ids(source["text"], config["vocab"], normalize)
            if (feature["text_length"] != len(original_ids) or feature["group_id"] != source["group_id"]
                    or feature["source_manifest_sha256"] != SOURCE_EXPECTED[str(INPUT / (split + ".jsonl"))]):
                raise ValueError("Original glyph/acoustic metadata binding differs")
            _positive(feature["frames"], "frames")
            for digest in (feature["feature_sha256"], source["sha256"], source["source_audio_sha256"]):
                _sha(digest)
            rows.append(dict(id=source["id"], split=split, source_text=source["text"],
                             effective_text=source["text"], glyph_normalized_text=normalized,
                             original_grapheme_ids=original_ids, original_grapheme_text_length=len(original_ids),
                             audio_filepath=source["audio_filepath"], audio_sha256=source["sha256"],
                             source_audio_sha256=source["source_audio_sha256"],
                             acoustic_feature_path=feature["feature"], acoustic_feature_sha256=feature["feature_sha256"],
                             frames=feature["frames"], group_id=source["group_id"],
                             input_manifest_sha256=SOURCE_EXPECTED[str(INPUT / (split + ".jsonl"))],
                             feature_manifest_sha256=SOURCE_EXPECTED[str(FEATURES / (split + ".jsonl"))]))
        result[split] = rows
    return result, config


def _probe_rows(parts, config):
    common = read_json(REPO / "benchmarks/v2/validation.pt-BR.json")
    baseline = read_json(BASELINE)["plan"]
    if tuple(row["id"] for row in common) != COMMON_IDS or common != baseline["cases"]:
        raise ValueError("Frozen common7 changed")
    public = {row["id"]: row for row in parts["validation"]}
    if not all(ident in public for ident in PUBLIC_IDS):
        raise ValueError("Fixed public4 not in validation")
    from training.f5_refine_text import normalize_text
    normalize = legacy_normalizer()
    cases = [dict(id=i, text=public[i]["source_text"]) for i in PUBLIC_IDS] + common
    rows = []
    for case in cases:
        effective = normalize(normalize_text(case["text"])["tts_text"])
        if case["id"] in COMMON_IDS and effective != baseline["effective_texts"][case["id"]]:
            raise ValueError("Frozen F5 refinement common7 changed")
        glyph_text, old_ids = glyph_ids(effective, config["vocab"], normalize)
        rows.append(dict(id=case["id"], split="probes", source_text=case["text"], effective_text=effective,
                         glyph_normalized_text=glyph_text, original_grapheme_ids=old_ids,
                         original_grapheme_text_length=len(old_ids), audio_filepath=None, audio_sha256=None,
                         source_audio_sha256=None, acoustic_feature_path=None, acoustic_feature_sha256=None,
                         frames=None, group_id="public4-refinement" if case["id"] in PUBLIC_IDS else "validation-common7",
                         input_manifest_sha256=SOURCE_EXPECTED[str(INPUT / "validation.jsonl")] if case["id"] in PUBLIC_IDS else SOURCE_EXPECTED[str(REPO / "benchmarks/v2/validation.pt-BR.json")],
                         feature_manifest_sha256=None))
    return rows


def build_plan(output=DEFAULT_OUTPUT):
    pins = _source_pins()
    parts, config = _public_rows()
    probes = _probe_rows(parts, config)
    metadata = front.frontend_metadata()
    stable = {k: metadata[k] for k in ("schema", "version", "symbols", "vocab_sha256", "map_sha256", "config_sha256", "fingerprint_sha256", "token_policy")}
    return dict(schema=PLAN_SCHEMA, status="PROPOSED_METADATA_ONLY", output=str(Path(output).resolve()),
                expected_counts=dict(COUNTS), frontend=stable, sources=_bindings(), source_pins=pins,
                acoustic_override=dict(OVERRIDE), probes=probes,
                output_reserve_bytes=64 * 1024**2,
                phonemization_executed=False, audio_read=False, acoustic_payload_read=False,
                torch_imported="torch" in sys.modules, gpu_used=False,
                preservation="No original manifests, cache tensors, text, WAVs or glyphIDs are overwritten; output must be new and outside Git.")


ROW_KEYS = {"schema", "id", "split", "source_text", "effective_text", "normalized_text", "source_text_sha256",
            "phoneme_sentences", "phoneme_ids", "phone_length", "original_grapheme_ids",
            "original_grapheme_text_length", "glyph_normalized_text", "audio_filepath", "audio_sha256",
            "source_audio_sha256", "acoustic_feature_path", "acoustic_feature_sha256", "frames", "group_id",
            "input_manifest_sha256", "feature_manifest_sha256", "map_sha256", "config_sha256"}


def validate_phone_row(row, split):
    if not isinstance(row, dict) or set(row) != ROW_KEYS or row["schema"] != ROW_SCHEMA or row["split"] != split:
        raise ValueError("Strict phone-row schema/split mismatch")
    if not isinstance(row["id"], str) or ID_RE.fullmatch(row["id"]) is None:
        raise ValueError("Unsafe phone row ID")
    if row["map_sha256"] != front.MAP_SHA256 or row["config_sha256"] != front.CONFIG_SHA256:
        raise ValueError("Native map/config changed")
    if not isinstance(row["source_text"], str) or not row["source_text"].strip():
        raise ValueError("Nonempty original source text required")
    if hashlib.sha256(row["source_text"].encode()).hexdigest() != row["source_text_sha256"]:
        raise ValueError("Source text hash differs")
    if row["normalized_text"] != front.normalize_input(row["effective_text"]):
        raise ValueError("NFC input normalization differs")
    sentences, ids = front.encode_sentences(row["phoneme_sentences"])
    if sentences != row["phoneme_sentences"] or ids != row["phoneme_ids"] or row["phone_length"] != len(ids):
        raise ValueError("NFD phones/IDs/length mismatch")
    glyphs = row["original_grapheme_ids"]
    if not isinstance(glyphs, list) or not glyphs or any(type(i) is not int or i < 0 for i in glyphs):
        raise ValueError("Original grapheme IDs required")
    if row["original_grapheme_text_length"] != len(glyphs):
        raise ValueError("Original grapheme length differs")
    _sha(row["input_manifest_sha256"])
    if split in ("train", "validation"):
        if row["source_text"] != row["effective_text"]:
            raise ValueError("Training/VAL text cannot be refined silently")
        _positive(row["frames"], "frames")
        for key in ("audio_sha256", "source_audio_sha256", "acoustic_feature_sha256", "feature_manifest_sha256"):
            _sha(row[key])
        if any(not isinstance(row[k], str) or not Path(row[k]).is_absolute() for k in ("audio_filepath", "acoustic_feature_path")):
            raise ValueError("Absolute acoustic/audio source paths required")
    elif split == "probes":
        if any(row[k] is not None for k in ("audio_filepath", "audio_sha256", "source_audio_sha256", "acoustic_feature_path", "acoustic_feature_sha256", "frames", "feature_manifest_sha256")):
            raise ValueError("Text-only probes must not bind a real/generated audio payload")
    else:
        raise ValueError("Only train/validation/probes sidecars exist")
    return row


def _lengths(rows):
    values = sorted(row["phone_length"] for row in rows)
    def quantile(p):
        return values[round((len(values)-1)*p)]
    observed = Counter(phone for row in rows for sentence in row["phoneme_sentences"] for phone in sentence)
    return dict(rows=len(rows), min=values[0], p50=quantile(.5), p95=quantile(.95), max=values[-1],
                total_ids=sum(values), distinct_observed_symbols=len(observed),
                symbol_counts=dict(sorted(observed.items())), no_unknown_symbols=True,
                accent_and_nasal_marks={mark: observed[mark] for mark in ("ˈ", "ˌ", "\u0303")},
                note="Coverage only; no pronunciation, language-quality or identity approval")


def prepare(plan):
    if "torch" in sys.modules:
        raise RuntimeError("Preparation G2P must be separate from Torch")
    if plan != build_plan(Path(plan["output"])):
        raise ValueError("Exact plan/source metadata changed before preparation")
    output = Path(plan["output"])
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new directory")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise ValueError("Token sidecars must remain outside Git")
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "rows").mkdir(mode=0o700)
    parts, config = _public_rows()
    parts["probes"] = _probe_rows(parts, config)
    partitions, coverage = {}, {}
    allocated = (output.stat().st_blocks + (output / "rows").stat().st_blocks) * 512
    for split in ("train", "validation", "probes"):
        indexes, produced = [], []
        for index, base in enumerate(parts[split]):
            phones = front.phonemize_text(base["effective_text"])
            row = dict(base, schema=ROW_SCHEMA, normalized_text=phones["normalized_text"],
                       source_text_sha256=hashlib.sha256(base["source_text"].encode()).hexdigest(),
                       phoneme_sentences=phones["sentences"], phoneme_ids=phones["ids"], phone_length=len(phones["ids"]),
                       map_sha256=front.MAP_SHA256, config_sha256=front.CONFIG_SHA256)
            validate_phone_row(row, split)
            name = "rows/" + split + "--" + row["id"] + ".json"
            write_json(output / name, row)
            allocated += (output / name).stat().st_blocks * 512
            if allocated > plan["output_reserve_bytes"]:
                raise RuntimeError("IPA sidecar output exceeded the admitted64MiB reserve")
            indexes.append(dict(schema=INDEX_SCHEMA, id=row["id"], split=split, path=name, sha256=front.sha256(output/name)))
            produced.append(row)
            if (index+1) % 100 == 0:
                print(json.dumps(dict(event="phone_sidecars", split=split, completed=index+1, total=len(parts[split]))), file=sys.stderr, flush=True)
        index_path = output / (split + ".jsonl")
        _atomic(index_path, b"".join(front.canonical(row)+b"\n" for row in indexes))
        partitions[split] = dict(path=index_path.name, sha256=front.sha256(index_path), count=len(indexes))
        coverage[split] = _lengths(produced)
    manifest = dict(schema=SCHEMA, status="complete", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    frontend=plan["frontend"], sources=plan["sources"], source_pins=plan["source_pins"],
                    acoustic_override=dict(OVERRIDE), splits=partitions, coverage=coverage,
                    expected_counts=dict(COUNTS), plan_canonical_sha256=hashlib.sha256(front.canonical(plan)).hexdigest())
    write_json(output / "manifest.json", manifest)
    loaded = load_manifest(output / "manifest.json", front.sha256(output / "manifest.json"))
    for split in COUNTS:
        load_phone_rows(loaded, split)
    return dict(status="prepared", manifest=str(output / "manifest.json"), manifest_sha256=front.sha256(output / "manifest.json"),
                counts=dict(COUNTS), audio_read=False, acoustic_payload_read=False, torch_imported=False,
                coverage=coverage, gpu_used=False)


def load_manifest(path, expected_sha=None):
    """Load/pin sidecar metadata without Torch/G2P/audio/acoustic payload IO."""
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular sidecar manifest required")
    actual = front.sha256(path)
    if expected_sha is not None and actual != _sha(expected_sha):
        raise ValueError("Phone manifest SHA differs")
    value = read_json(path)
    keys = {"schema", "status", "created_utc", "frontend", "sources", "source_pins", "acoustic_override", "splits", "coverage", "expected_counts", "plan_canonical_sha256"}
    if set(value) != keys or value["schema"] != SCHEMA or value["status"] != "complete":
        raise ValueError("Strict complete sidecar manifest required")
    if value["expected_counts"] != COUNTS or set(value["splits"]) != set(COUNTS):
        raise ValueError("Frozen3037/172/11 partitions required")
    current = front.frontend_metadata()
    frontend = {k: current[k] for k in ("schema", "version", "symbols", "vocab_sha256", "map_sha256", "config_sha256", "fingerprint_sha256", "token_policy")}
    if value["frontend"] != frontend:
        raise ValueError("Native161 frontend differs")
    if value["sources"] != _bindings() or value["acoustic_override"] != OVERRIDE:
        raise ValueError("Acoustic source/explicit token override differs")
    _sha(value["plan_canonical_sha256"])
    _check_source_pins(value["source_pins"])
    for split, count in COUNTS.items():
        binding = value["splits"][split]
        if set(binding) != {"path", "sha256", "count"} or binding["count"] != count:
            raise ValueError("Index binding/count differs")
        index = _relative(path.parent, binding["path"])
        if front.sha256(index) != _sha(binding["sha256"]):
            raise ValueError("Phone index SHA differs")
    return dict(value, _manifest_path=str(path), _manifest_sha256=actual)


def load_phone_rows(manifest, split):
    if split not in COUNTS:
        raise ValueError("Unknown sidecar split")
    if not isinstance(manifest, dict) or "_manifest_path" not in manifest:
        raise ValueError("Use load_manifest first")
    # Rebind the complete metadata manifest to reject mutation between loader calls.
    path = Path(manifest["_manifest_path"])
    if front.sha256(path) != manifest["_manifest_sha256"]:
        raise ValueError("Phone manifest changed after loading")
    if {k: v for k, v in manifest.items() if not k.startswith("_")} != read_json(path):
        raise ValueError("Loaded phone manifest was mutated in memory")
    binding = manifest["splits"][split]
    index = _relative(path.parent, binding["path"])
    if front.sha256(index) != _sha(binding["sha256"]):
        raise ValueError("Phone index changed")
    entries = read_jsonl(index)
    if len(entries) != COUNTS[split]:
        raise ValueError("Phone row count differs")
    result, ids = [], set()
    for entry in entries:
        if (set(entry) != {"schema", "id", "split", "path", "sha256"}
                or entry["schema"] != INDEX_SCHEMA or entry["split"] != split or entry["id"] in ids):
            raise ValueError("Duplicate/invalid row index")
        ids.add(entry["id"])
        row_path = _relative(path.parent, entry["path"])
        if front.sha256(row_path) != _sha(entry["sha256"]):
            raise ValueError("Phone row SHA differs")
        row = validate_phone_row(read_json(row_path), split)
        if row["id"] != entry["id"]:
            raise ValueError("Phone row ID differs")
        result.append(row)
    if split == "probes":
        if tuple(row["id"] for row in result) != PUBLIC_IDS + COMMON_IDS:
            raise ValueError("Frozen public4+common7 probe order changed")
        parts, config = _public_rows()
        expected = _probe_rows(parts, config)
    else:
        parts, unused = _public_rows()
        expected = parts[split]
        if [r["id"] for r in result] != [r["id"] for r in expected]:
            raise ValueError("Frozen public row order/IDs differ")
    for row, source in zip(result, expected):
        if any(row[key] != value for key, value in source.items()):
            raise ValueError("Frozen source text/glyph IDs/acoustic SHA/partition binding differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.prepare:
        if args.plan is None or args.plan_sha256 is None or args.write_plan is not None:
            parser.error("--prepare requires --plan and --plan-sha256; no --write-plan")
        if front.sha256(args.plan) != _sha(args.plan_sha256):
            raise ValueError("Exact reviewed plan file SHA differs")
        result = prepare(read_json(args.plan))
    else:
        if args.plan is not None or args.plan_sha256 is not None:
            parser.error("--plan/--plan-sha256 apply only to explicit preparation")
        result = build_plan(args.output)
        if args.write_plan is not None:
            if args.write_plan.exists():
                raise ValueError("Do not overwrite a plan")
            if any((p / ".git").exists() for p in (args.write_plan, *args.write_plan.parents)):
                raise ValueError("Plan must be outside Git")
            write_json(args.write_plan, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
