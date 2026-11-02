#!/usr/bin/env python3
"""Probe hash-pinned v2607 PT-BR G2P without Torch, NeMo installation or weights.

Executes the official pure Python G2P/tokenizer definitions, omitting only their
NeMo package imports. Does not perform text normalization or TTS inference.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path
import random
import typing
from contextlib import contextmanager

from magpie_codec_probe import digest, numeric_definitions, SPEECH_REVISION

HASHES = {
    "nemo_collections_common_tokenizers_text_to_speech_ipa_lexicon.py": "75137c41e16d85d1f3a99c6af6ded0fa444bd6c2166163837d464e129d847b2c",
    "nemo_collections_common_tokenizers_text_to_speech_tokenizer_utils.py": "3be2e521296eeb0afb1ceb5c335e450e56dc4ea92553646321d611d805e77182",
    "nemo_collections_tts_g2p_utils.py": "67c319fea55b89075646f9111ac5dee6a5854d8caed4b61dc82376d445e1c5f1",
    "nemo_collections_tts_g2p_models_base.py": "f0c7fc07f7bb06988b8220ca562fccd45d2c2471574d2a79fc010666f290fbfb",
    "nemo_collections_tts_g2p_models_i18n_ipa.py": "7dda75c166dfd22dff8525e23aa73bfa60be6211c26dd8ac0a4a2d9db441bd39",
    "nemo_collections_common_tokenizers_text_to_speech_tts_tokenizers.py": "a118ed54db7af25dc4fa1c94b72c1dbbb01d58ae53448804ba8276e0c4b81674",
    "pt_br_prondict-v1.0.dict": "6492abc404db16bbad83dd9d7f9a60eb617699f0a3ff54b3be6184e14507b745",
}


def probe(root, corpus):
    for name, expected in HASHES.items():
        if digest(root / name) != expected:
            raise ValueError(f"unverified upstream frontend: {name}")
    namespace = {**vars(typing), "logging": logging, "contextmanager": contextmanager,
                 "DEFAULT_LOCALE_SPECIFIC_PUNCT": True}
    # These modules only import Python's standard library and each other. NeMo
    # package initialization is bypassed, but dictionaries, vocabulary sorting,
    # Unicode processing and G2P/token ID logic remain the upstream definitions.
    for name in list(HASHES)[:5]:
        path = root / name
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [n for n in tree.body if not
                     (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("nemo"))]
        tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    numeric_definitions(root / "nemo_collections_common_tokenizers_text_to_speech_tts_tokenizers.py",
                        ["IPATokenizer"], namespace)
    g2p = namespace["IpaG2p"](phoneme_dict=root / "pt_br_prondict-v1.0.dict", locale="pt-BR",
                              phoneme_probability=0.8, ignore_ambiguous_words=False,
                              use_chars=True, use_stresses=True, grapheme_case="upper", grapheme_prefix="#")
    tokenizer = namespace["IPATokenizer"](g2p=g2p, locale="pt-BR", punct=True, apostrophe=True,
                                           pad_with_space=True, locale_specific_punct=False)
    # Explicit diagnostic setting: use dictionary pronunciations deterministically
    # instead of the checkpoint config's 80% phoneme/grapheme mixture. Do not
    # assume the complete NeMo runtime applies this override automatically.
    g2p.phoneme_probability = 1.0
    random.seed(40733133)
    rows = []
    for row in corpus:
        text = row["text"]
        prepared = tokenizer.text_preprocessing_func(text)
        phones = g2p(prepared)
        ids = tokenizer.encode_from_g2p(phones, prepared)
        unknown = sorted({p for p in phones if p != " " and p not in tokenizer.tokens_set})
        words = namespace["any_locale_word_tokenize"](prepared)
        oov = ["".join(w) for w, leave in words
               if not leave and "".join(w).upper() not in g2p.phoneme_dict and any(c.isalpha() for c in "".join(w))]
        rows.append({"id": row["id"], "text": text, "preprocessed_text": prepared,
                     "g2p": phones, "ids": ids, "skipped_symbols": unknown,
                     "words_absent_from_dictionary": oov})
    return {"speech_revision": SPEECH_REVISION, "source_hashes": HASHES,
            "checkpoint_revision": "5023df68bd3f5b5ce6d666a50979bc501af145cc",
            "tokenizer": "portuguese_Brazilian_phoneme", "locale": "pt-BR",
            "locale_specific_punct": False, "phoneme_probability_inference": 1.0,
            "vocabulary_tokens": len(tokenizer.tokens),
            "vocabulary_sha256": hashlib.sha256(json.dumps(tokenizer.tokens, ensure_ascii=False).encode()).hexdigest(),
            "text_normalization_applied": False, "tts_inference_executed": False, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=Path(__file__).with_name("corpus.pt-BR.json"))
    args = parser.parse_args(argv)
    print(json.dumps(probe(args.artifact_dir, json.loads(args.corpus.read_text())), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
