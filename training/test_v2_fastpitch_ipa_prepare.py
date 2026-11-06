"""Metadata and synthetic sidecar-loader tests; no WAV/features/GPU/Torch IO."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training import v2_fastpitch_ipa_frontend as front
from training import v2_fastpitch_ipa_prepare as prep


def synthetic_row(ident="one", split="train"):
    text = "mãe."
    config = prep.read_json(prep.FEATURES / "features.json")
    normalized, glyphs = prep.glyph_ids(text, config["vocab"], prep.legacy_normalizer())
    sentences, ids = front.encode_sentences([["m", "ɐ", "\u0303", "."]])
    base = dict(id=ident, split=split, source_text=text, effective_text=text,
                glyph_normalized_text=normalized, original_grapheme_ids=glyphs,
                original_grapheme_text_length=len(glyphs), audio_filepath="/synthetic/"+ident+".wav",
                audio_sha256="a"*64, source_audio_sha256="b"*64,
                acoustic_feature_path="/synthetic/"+ident+".pt", acoustic_feature_sha256="c"*64,
                frames=100, group_id=split+"-"+ident,
                input_manifest_sha256="d"*64, feature_manifest_sha256="e"*64)
    row = dict(base, schema=prep.ROW_SCHEMA, normalized_text=text,
               source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
               phoneme_sentences=sentences, phoneme_ids=ids, phone_length=len(ids),
               map_sha256=front.MAP_SHA256, config_sha256=front.CONFIG_SHA256)
    return base, row


@contextmanager
def fixture():
    """Tiny synthetic partitions; patches are test-local and never reach disk recipes."""
    with tempfile.TemporaryDirectory(prefix="aegis-ipa-sidecar-stdlib-") as temporary:
        root = Path(temporary)
        (root / "rows").mkdir()
        bases, rows, indexes = {}, {}, {}
        for split, ids in (("train", ["one", "two"]), ("validation", ["three"]), ("probes", [])):
            bases[split], rows[split], indexes[split] = [], [], []
            for ident in ids:
                base, row = synthetic_row(ident, split)
                bases[split].append(base)
                rows[split].append(row)
                name = "rows/"+split+"--"+ident+".json"
                prep.write_json(root/name, row)
                indexes[split].append(dict(schema=prep.INDEX_SCHEMA, id=ident, split=split,
                                           path=name, sha256=front.sha256(root/name)))
        metadata = front.frontend_metadata()
        frontend = {k: metadata[k] for k in ("schema", "version", "symbols", "vocab_sha256", "map_sha256", "config_sha256", "fingerprint_sha256", "token_policy")}
        manifest = dict(schema=prep.SCHEMA, status="complete", created_utc="synthetic",
                        frontend=frontend, sources=prep._bindings(), source_pins=prep._source_pins(),
                        acoustic_override=dict(prep.OVERRIDE), splits={}, coverage={},
                        expected_counts=dict(train=2, validation=1, probes=11), plan_canonical_sha256="f"*64)

        def save():
            for split, entries in indexes.items():
                index = root/(split+".jsonl")
                index.write_bytes(b"".join(front.canonical(r)+b"\n" for r in entries))
                manifest["splits"][split] = dict(path=index.name, sha256=front.sha256(index),
                                               count=manifest["expected_counts"][split])
            prep.write_json(root/"manifest.json", manifest)
            return prep.load_manifest(root/"manifest.json", front.sha256(root/"manifest.json"))

        with mock.patch.object(prep, "COUNTS", dict(train=2, validation=1, probes=11)), \
                mock.patch.object(prep, "_public_rows", return_value=(bases, {})):
            yield root, bases, rows, indexes, manifest, save


class RowTests(unittest.TestCase):
    def test_valid_row_preserves_original_glyphs_and_uses_separate_phone_ids(self):
        base, row = synthetic_row()
        self.assertEqual(prep.validate_phone_row(row, "train"), row)
        self.assertEqual(row["original_grapheme_ids"], base["original_grapheme_ids"])
        self.assertNotEqual(row["original_grapheme_ids"], row["phoneme_ids"])

    def test_oov_tamper_and_nfc_phone_recomposition_are_rejected(self):
        for mutate in (lambda r: r["phoneme_ids"].__setitem__(1, 161),
                       lambda r: r.__setitem__("phoneme_sentences", [["m", "ã", "."]]),
                       lambda r: r.__setitem__("phone_length", 1),
                       lambda r: r.__setitem__("split", "validation"),
                       lambda r: r.__setitem__("source_text", "outra fala.")):
            base, row = synthetic_row()
            mutate(row)
            with self.assertRaises(ValueError):
                prep.validate_phone_row(row, "train")

    def test_train_cannot_silently_switch_to_refined_text(self):
        base, row = synthetic_row()
        row["effective_text"] = row["normalized_text"] = "uma mãe."
        with self.assertRaisesRegex(ValueError, "cannot be refined"):
            prep.validate_phone_row(row, "train")

    def test_unknown_extra_row_key_is_rejected(self):
        base, row = synthetic_row()
        row["unknown"] = True
        with self.assertRaisesRegex(ValueError, "schema"):
            prep.validate_phone_row(row, "train")

    def test_train_val_audio_text_group_or_id_leak_is_rejected(self):
        train = dict(id="one", text="primeira fala", sha256="a"*64, group_id="g1")
        val = dict(id="two", text="segunda fala", sha256="b"*64, group_id="g2")
        prep.validate_partitions(dict(train=[train], validation=[val]))
        for key in ("id", "text", "sha256", "group_id"):
            damaged = dict(val)
            damaged[key] = train[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                prep.validate_partitions(dict(train=[train], validation=[damaged]))


class LoaderTests(unittest.TestCase):
    def test_actual_loader_accepts_pinned_synthetic_sidecars(self):
        with fixture() as (root, bases, rows, indexes, manifest, save):
            loaded = save()
            self.assertEqual(prep.load_phone_rows(loaded, "train"), rows["train"])
            self.assertEqual(prep.load_phone_rows(loaded, "validation"), rows["validation"])

    def test_manifest_expected_sha_and_unknown_fields_fail(self):
        with fixture() as (root, bases, rows, indexes, manifest, save):
            loaded = save()
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                prep.load_manifest(root/"manifest.json", "0"*64)
            manifest["extra"] = True
            with self.assertRaisesRegex(ValueError, "Strict complete"):
                save()

    def test_sidecar_byte_tamper_fails_before_row_use(self):
        with fixture() as (root, bases, rows, indexes, manifest, save):
            loaded = save()
            path = root/indexes["train"][0]["path"]
            path.write_text(path.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "Phone row SHA"):
                prep.load_phone_rows(loaded, "train")

    def test_self_consistent_forged_glyph_or_acoustic_binding_still_fails(self):
        for key, value in (("original_grapheme_ids", [42]*6), ("acoustic_feature_sha256", "0"*64),
                           ("audio_sha256", "0"*64), ("group_id", "validation-group")):
            with self.subTest(key=key), fixture() as (root, bases, rows, indexes, manifest, save):
                damaged = copy.deepcopy(rows["train"][0])
                if key == "original_grapheme_ids":
                    value = [42]*len(damaged[key])
                damaged[key] = value
                entry = indexes["train"][0]
                prep.write_json(root/entry["path"], damaged)
                entry["sha256"] = front.sha256(root/entry["path"])
                loaded = save()
                with self.assertRaisesRegex(ValueError, "Frozen source"):
                    prep.load_phone_rows(loaded, "train")

    def test_duplicate_index_and_in_memory_manifest_mutation_fail(self):
        with fixture() as (root, bases, rows, indexes, manifest, save):
            indexes["train"][1] = dict(indexes["train"][0])
            loaded = save()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prep.load_phone_rows(loaded, "train")
        with fixture() as (root, bases, rows, indexes, manifest, save):
            loaded = save()
            loaded["acoustic_override"]["fresh_model"] = False
            with self.assertRaisesRegex(ValueError, "mutated in memory"):
                prep.load_phone_rows(loaded, "train")

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="aegis-ipa-path-stdlib-") as directory:
            for path in ("../outside.json", "/absolute.json"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    prep._relative(directory, path)

    def test_source_pin_tampering_is_rejected_without_payload_io(self):
        pins = prep._source_pins()
        pins[str(prep.INPUT/"train.jsonl")] = "0"*64
        with self.assertRaisesRegex(ValueError, "Source pin mismatch"):
            prep._check_source_pins(pins)


class PlanningTests(unittest.TestCase):
    def test_actual_default_plan_never_calls_phonemizer_or_imports_torch(self):
        with tempfile.TemporaryDirectory(prefix="aegis-ipa-plan-stdlib-") as directory:
            unused_output = Path(directory) / "not-created"
            with mock.patch.object(front, "phonemize_text", side_effect=AssertionError("G2P is not planning")):
                plan = prep.build_plan(unused_output)
            self.assertEqual(plan["expected_counts"], dict(train=3037, validation=172, probes=11))
            self.assertEqual(tuple(r["id"] for r in plan["probes"]), prep.PUBLIC_IDS+prep.COMMON_IDS)
            self.assertFalse(plan["phonemization_executed"])
            self.assertFalse(plan["audio_read"] or plan["acoustic_payload_read"] or plan["torch_imported"] or plan["gpu_used"])
            self.assertFalse(unused_output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
