"""Frozen native161 PT-BR IPA frontend for the separate fresh FastPitch recipe.

Import is stdlib-only and performs no file IO or native initialization. Phoneme
IDs are BOS + NFD sentence codepoints joined by SPACE + EOS, without Piper's
blank interleave. Native G2P must run in a separate CPU process before Torch.
The JSON-lines --serve interface emits one ready record, then one response per
request {"id": string, "text": string}; errors never silently discard symbols.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType
import unicodedata

VERSION = "fastpitch-ipa-native161-noblank-v1"
SCHEMA = "aegis.fastpitch.ipa.frontend.v1"
ROOT = Path("/var/tmp/aegis-v2-work")
OVERLAY = ROOT / "piper/deps"
CONFIG = ROOT / "piper-research/cadu-config.json"
CONFIG_SHA256 = "9433ddafa54e7d5c973d6160a29c046e39d65a2dc8892305591c7b93656c2253"
VOICE = "pt-br"
SYMBOLS = ('_', '^', '$', ' ', '!', "'", '(', ')', ',', '-', '.', ':', ';', '?',
           'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
           'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð',
           'ø', 'ħ', 'ŋ', 'œ', 'ǀ', 'ǁ', 'ǂ', 'ǃ', 'ɐ', 'ɑ', 'ɒ', 'ɓ', 'ɔ', 'ɕ',
           'ɖ', 'ɗ', 'ɘ', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɞ', 'ɟ', 'ɠ', 'ɡ', 'ɢ', 'ɣ', 'ɤ',
           'ɥ', 'ɦ', 'ɧ', 'ɨ', 'ɪ', 'ɫ', 'ɬ', 'ɭ', 'ɮ', 'ɯ', 'ɰ', 'ɱ', 'ɲ', 'ɳ',
           'ɴ', 'ɵ', 'ɶ', 'ɸ', 'ɹ', 'ɺ', 'ɻ', 'ɽ', 'ɾ', 'ʀ', 'ʁ', 'ʂ', 'ʃ', 'ʄ',
           'ʈ', 'ʉ', 'ʊ', 'ʋ', 'ʌ', 'ʍ', 'ʎ', 'ʏ', 'ʐ', 'ʑ', 'ʒ', 'ʔ', 'ʕ', 'ʘ',
           'ʙ', 'ʛ', 'ʜ', 'ʝ', 'ʟ', 'ʡ', 'ʢ', 'ʲ', 'ˈ', 'ˌ', 'ː', 'ˑ', '˞', 'β',
           'θ', 'χ', 'ᵻ', 'ⱱ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           '\u0327', '\u0303', '\u032a', '\u032f', '\u0329', 'ʰ', 'ˤ', 'ε', '↓',
           '#', '"', '↑', '\u033a', '\u033b', 'g', 'ʦ', 'X', '\u031d', '\u030a', 'ɝ', 'ʷ')
SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(SYMBOLS)}
VOCAB = dict(SYMBOL_TO_ID)
PAD, BOS, EOS, SPACE = 0, 1, 2, 3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


SYMBOLS_SHA256 = hashlib.sha256(canonical(SYMBOLS)).hexdigest()
VOCAB_SHA256 = SYMBOLS_SHA256
MAP_SHA256 = hashlib.sha256(canonical({s: [i] for s, i in SYMBOL_TO_ID.items()})).hexdigest()
_G2P = None
_LOCK = threading.RLock()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_input(text):
    if not isinstance(text, str) or not text.strip() or "\0" in text:
        raise ValueError("text must be a nonempty string without NUL")
    if any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        raise ValueError("surrogate codepoints are not text")
    return " ".join(unicodedata.normalize("NFC", text).split())


def encode_sentences(sentences):
    """Encode every NFD codepoint; reject empty or unknown output, never drop."""
    if not isinstance(sentences, (list, tuple)) or not sentences:
        raise ValueError("at least one phoneme sentence is required")
    normalized = []
    for sentence in sentences:
        if not isinstance(sentence, (list, tuple)) or not sentence:
            raise ValueError("phoneme sentence must be a nonempty sequence")
        if any(not isinstance(p, str) or len(p) != 1 for p in sentence):
            raise ValueError("phonemes must be single Unicode codepoints")
        phones = list(unicodedata.normalize("NFD", "".join(sentence)))
        unknown = sorted(set(phones) - set(SYMBOL_TO_ID))
        if unknown:
            raise ValueError("unknown IPA codepoints: " + repr(unknown))
        if any(SYMBOL_TO_ID[p] in (PAD, BOS, EOS) for p in phones):
            raise ValueError("reserved PAD/BOS/EOS cannot occur inside a sentence")
        if not any(p != " " for p in phones):
            raise ValueError("space-only phoneme sentence")
        normalized.append(phones)
    ids = [BOS]
    for index, phones in enumerate(normalized):
        if index:
            ids.append(SPACE)
        ids.extend(SYMBOL_TO_ID[p] for p in phones)
    ids.append(EOS)
    validate_ids(ids)
    return normalized, ids


def validate_ids(ids):
    if not isinstance(ids, list) or len(ids) < 3:
        raise ValueError("phoneme IDs must be a BOS/content/EOS list")
    if any(type(i) is not int or not 0 <= i < len(SYMBOLS) for i in ids):
        raise ValueError("unknown or non-integer phoneme ID")
    if ids[0] != BOS or ids[-1] != EOS or any(i in (PAD, BOS, EOS) for i in ids[1:-1]):
        raise ValueError("exactly one boundary BOS/EOS and no PAD are required")
    return ids


def frontend_metadata():
    """Read code/config/dictionary pins only; never initialize native G2P."""
    if len(SYMBOLS) != 161 or len(SYMBOL_TO_ID) != 161:
        raise ValueError("native inventory must have161 unique symbols")
    if sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("pinned Cadu map/config changed")
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (value["phoneme_id_map"] != {s: [i] for s, i in SYMBOL_TO_ID.items()}
            or value["espeak"]["voice"] != VOICE):
        raise ValueError("native Cadu phoneme IDs/voice changed")
    package = OVERLAY / "piper"
    dist = OVERLAY / "piper_tts-1.8.0.dist-info"
    files = [CONFIG, Path(__file__), package / "phonemize_espeak.py", package / "espeakbridge.so",
             dist / "METADATA", dist / "licenses/COPYING"]
    files.extend(package / "espeak-ng-data" / p for p in
                 ("pt_dict", "phondata", "phonindex", "phontab", "intonations", "lang/roa/pt", "lang/roa/pt-BR"))
    pins = {str(p.resolve()): sha256(p) for p in files}
    # Foreign words can consult other dictionaries: bind the complete19MB data
    # tree as one digest without bloating each daemon READY record with373 paths.
    data_dir = package / "espeak-ng-data"
    resources = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError("Unexpected symlink in native G2P data")
        if path.is_file():
            resources.append(dict(path=str(path.relative_to(data_dir)), bytes=path.stat().st_size, sha256=sha256(path)))
    stable = dict(schema=SCHEMA, version=VERSION, voice=VOICE, symbols=list(SYMBOLS),
                  vocab_sha256=VOCAB_SHA256, map_sha256=MAP_SHA256,
                  config_sha256=CONFIG_SHA256, source_pins=pins,
                  espeak_data_tree_sha256=hashlib.sha256(canonical(resources)).hexdigest(),
                  espeak_data_files=len(resources), espeak_data_bytes=sum(r["bytes"] for r in resources),
                  espeak_version="not exposed by pinned bridge API; exact binary and complete dictionary-tree digests are authoritative",
                  input_normalization="NFC; whitespace runs to one SPACE; case retained",
                  output_normalization="NFD codepoints; never NFC; no symbol dropping",
                  token_policy="BOS1 + unchanged NFD sentences joined by one SPACE3 + EOS2; no blank interleave",
                  implementation="piper-tts1.8.0 piper.phonemize_espeak.EspeakPhonemizer/espeakbridge; isolated package without voice/ONNX",
                  licenses=dict(piper_code="GPL-3.0-or-later", espeak="GPL-3.0-or-later",
                                cadu_model="not loaded; map is metadata, not a pretrained voice initialization"))
    return dict(stable, fingerprint_sha256=hashlib.sha256(canonical(stable)).hexdigest(),
                python=sys.version.split()[0], torch_imported="torch" in sys.modules,
                native_initialized=_G2P is not None)


def _native_frontend():
    global _G2P
    if "torch" in sys.modules:
        raise RuntimeError("native IPA G2P is prohibited after Torch import; use --serve CPU child")
    if _G2P is None:
        frontend_metadata()
        name = "_aegis_fastpitch_ipa_piper"
        package = ModuleType(name)
        package.__path__ = [str(OVERLAY / "piper")]
        package.__package__ = name
        sys.modules[name] = package
        fullname = name + ".phonemize_espeak"
        spec = importlib.util.spec_from_file_location(fullname, OVERLAY / "piper/phonemize_espeak.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        _G2P = module.EspeakPhonemizer(module.ESPEAK_DATA_DIR)
    return _G2P


def phonemize_text(text):
    normalized = normalize_input(text)
    with _LOCK:
        native = _native_frontend()
        sentences, ids = encode_sentences(native.phonemize(VOICE, normalized, vowel_clusters=None))
    return dict(normalized_text=normalized, sentences=sentences, ids=ids,
                map_sha256=MAP_SHA256, config_sha256=CONFIG_SHA256,
                vocab_sha256=VOCAB_SHA256, frontend_version=VERSION)


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def serve():
    # Initialize once before READY so startup errors are explicit process failures.
    with _LOCK:
        _native_frontend()
    _emit(dict(status="ready", frontend=frontend_metadata(), vocab_sha256=VOCAB_SHA256))
    for line in sys.stdin:
        request_id = None
        try:
            if len(line) > 1_000_000:
                raise ValueError("request exceeds1MB")
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"id", "text"}:
                raise ValueError("request requires exactly id and text")
            request_id = request["id"]
            if not isinstance(request_id, str):
                raise ValueError("request id must be a string")
            _emit(dict(id=request_id, ok=True, **phonemize_text(request["text"])))
        except Exception as error:
            _emit(dict(id=request_id, ok=False, error=type(error).__name__ + ": " + str(error)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--text")
    args = parser.parse_args()
    if args.serve:
        if args.text is not None:
            parser.error("--serve and --text are mutually exclusive")
        serve()
    elif args.text is not None:
        _emit(dict(ok=True, **phonemize_text(args.text)))
    else:
        _emit(frontend_metadata())


if __name__ == "__main__":
    main()
