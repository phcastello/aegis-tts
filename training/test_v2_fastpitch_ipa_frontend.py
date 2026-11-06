"""Stdlib frontend invariants and small real CPU text-only G2P integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unicodedata
import unittest
from unittest import mock

from training import v2_fastpitch_ipa_frontend as front

REPO = Path(__file__).resolve().parents[1]


class TokenTests(unittest.TestCase):
    def test_native_map_and_boundary_identity(self):
        self.assertEqual(len(front.SYMBOLS), 161)
        self.assertEqual(front.VOCAB["\u0303"], 141)
        self.assertEqual(tuple(front.VOCAB[s] for s in ("_", "^", "$", " ")), (0, 1, 2, 3))
        self.assertEqual(front.VOCAB_SHA256, "b95096575451d8cc9303f039e23239c5a9f83766f458624c25dfda6ced04e0b3")

    def test_nfd_nasal_and_stress_are_independent_ids(self):
        sentences, ids = front.encode_sentences([["m", "ˈ", "ɐ", "\u0303", "j"]])
        self.assertEqual(sentences, [["m", "ˈ", "ɐ", "\u0303", "j"]])
        self.assertEqual(ids, [1, 25, 120, 50, 141, 22, 2])
        self.assertNotIn(0, ids)

    def test_decomposed_phone_marks_are_never_recomposed_to_nfc(self):
        sentences, ids = front.encode_sentences([["ã"]])
        self.assertEqual(sentences, [["a", "\u0303"]])
        self.assertEqual(ids, [1, 14, 141, 2])
        self.assertNotEqual("".join(sentences[0]), unicodedata.normalize("NFC", "".join(sentences[0])))

    def test_two_sentences_have_one_added_space_and_one_bos_eos(self):
        sentences, ids = front.encode_sentences([["a", "."], ["b", "."]])
        self.assertEqual(ids, [1, 14, 10, 3, 15, 10, 2])
        self.assertEqual(len(sentences), 2)

    def test_unknown_or_reserved_phone_is_rejected(self):
        for sentence in (["☃"], ["^"], ["_"], ["$"], [" "], [], ["ab"]):
            with self.subTest(sentence=sentence), self.assertRaises(ValueError):
                front.encode_sentences([sentence])

    def test_id_oov_bool_padding_or_boundary_is_rejected(self):
        for ids in ([1, 161, 2], [1, True, 2], [1, 0, 2], [1, 1, 2], [1, 14, 1], [1, 2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                front.validate_ids(ids)

    def test_input_is_nfc_and_does_not_change_case(self):
        self.assertEqual(front.normalize_input("  MA\u0303E\t e pão. "), "MÃE e pão.")
        for text in ("", "  ", "a\0b", "a\ud800"):
            with self.subTest(text=repr(text)), self.assertRaises(ValueError):
                front.normalize_input(text)

    def test_torch_guard_precedes_native_initialization(self):
        with mock.patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            with self.assertRaisesRegex(RuntimeError, "prohibited after Torch"):
                front.phonemize_text("Uma casa.")


class ProcessTests(unittest.TestCase):
    def test_imports_are_stdlib_only_and_do_not_read_runtime_files(self):
        script = """
import builtins,json,pathlib,sys
original = builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','piper','onnxruntime'}:
        raise AssertionError('forbidden import '+name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
pathlib.Path.open=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('runtime file read'))
from training import v2_fastpitch_ipa_frontend,v2_fastpitch_ipa_prepare
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert not any(k.startswith('_aegis_fastpitch_ipa_piper') for k in sys.modules)
print(json.dumps({'import_safe':True}))
"""
        result = subprocess.run([sys.executable, "-S", "-B", "-c", script], cwd=REPO,
                                capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(result.stdout)["import_safe"])

    def test_real_cpu_daemon_nasal_normalization_error_recovery_and_protocol(self):
        requests = [dict(id="first", text="Mãe, o pão está quente."),
                    dict(id="decomposed", text="Ma\u0303e, o pa\u0303o esta\u0301 quente."),
                    dict(id="bad", text=""), dict(id="after-error", text="Uma casa. Duas salas.")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(["/usr/bin/python3", "-S", "-B", "-m", "training.v2_fastpitch_ipa_frontend", "--serve"],
                                cwd=REPO, env=env, input="".join(json.dumps(r, ensure_ascii=False)+"\n" for r in requests),
                                capture_output=True, text=True, timeout=20, check=True)
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(replies), 5)
        ready, a, b, bad, after = replies
        self.assertEqual(ready["status"], "ready")
        self.assertFalse(ready["frontend"]["torch_imported"])
        self.assertEqual(ready["vocab_sha256"], front.VOCAB_SHA256)
        self.assertEqual([r["id"] for r in replies[1:]], [r["id"] for r in requests])
        self.assertTrue(a["ok"] and b["ok"] and after["ok"])
        self.assertEqual(a["normalized_text"], b["normalized_text"])
        self.assertEqual(a["ids"], b["ids"])
        self.assertIn(141, a["ids"])
        self.assertFalse(bad["ok"])
        self.assertEqual(len(after["sentences"]), 2)
        for row in (a, b, after):
            front.validate_ids(row["ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
