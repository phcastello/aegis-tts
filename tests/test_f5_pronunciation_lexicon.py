"""F5 keeps the old pronunciation rules and uses the corrected name aliases."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))

from aegis_speech.lexicon import PronunciationLexicon
from aegis_speech.normalization import SpeechTextNormalizer

PRONUNCIATION = ROOT / "config/pronunciation"
F5_LEXICON = PRONUNCIATION / "pt-BR-f5-v2.yaml"
LEGACY_LEXICON = ROOT / "protected/AegisVoicev1.0/pronunciation/pt-BR.yaml"


def f5_normalizer() -> SpeechTextNormalizer:
    return SpeechTextNormalizer(PronunciationLexicon(F5_LEXICON).apply)


def test_every_legacy_rule_is_preserved_except_the_intended_name_correction() -> None:
    legacy = PronunciationLexicon(LEGACY_LEXICON)
    current = PronunciationLexicon(F5_LEXICON)
    old_entries = {entry.entry_id: entry for entry in legacy.entries}
    new_entries = {entry.entry_id: entry for entry in current.entries}
    assert current.version == "2"
    assert set(new_entries) == set(old_entries) | {"gmail"}
    for entry_id in old_entries.keys() - {"aegis"}:
        assert new_entries[entry_id] == old_entries[entry_id]
    assert new_entries["aegis"].replacement == "Aéjis"
    assert new_entries["aegis"].kind == old_entries["aegis"].kind
    assert new_entries["aegis"].case_sensitive == old_entries["aegis"].case_sensitive
    assert new_entries["aegis"].enabled == old_entries["aegis"].enabled


@pytest.mark.parametrize("case", yaml.safe_load((PRONUNCIATION / "corpus.yaml").read_text(encoding="utf-8"))["cases"])
def test_f5_retains_original_normalization_corpus(case: dict) -> None:
    expected = "Aéjis está operacional." if case["id"] == "aegis" else case["expected_normalized"]
    assert f5_normalizer().normalize(case["input"]).normalized_text == expected


@pytest.mark.parametrize(("source", "expected"), [
    ("Aegis, abra o Gmail.", "Aéjis, abra o gê mail."),
    ("AEGIS, aegis, Gmail, GMAIL e gmail.", "Aéjis, Aéjis, gê mail, gê mail e gê mail."),
    ("Qwen e QWEN; VPN, DNS e GPU.", "Cuén e Cuén; vê pê ene, dê ene ésse e gê pê u."),
    ("vpn dns gpu; microAegis e Gmailbox.", "vpn dns gpu; microAegis e Gmailbox."),
    ("Aegis_ Gmailx AegisPlus.", "Aegis_ Gmailx AegisPlus."),
    ("O endereço é exemplo@gmail.com.", "O endereço é exemplo arroba gê mail ponto com."),
    ("Aegis https://gmail.com.", "Aéjis você pode ver o link pela interface."),
    ("**Aegis** abriu o Gmail (28 registros).", "Aéjis abriu o gê mail vinte e oito registros."),
    ("Aéjis, gê mail, Cuén, vê pê ene, dê ene ésse e gê pê u.",
     "Aéjis, gê mail, Cuén, vê pê ene, dê ene ésse e gê pê u."),
])
def test_f5_aliases_case_boundaries_and_no_second_replacement(source: str, expected: str) -> None:
    normalizer = f5_normalizer()
    result = normalizer.normalize(source)
    assert result.original_text == source
    assert result.normalized_text == expected
    assert normalizer.normalize(result.normalized_text).normalized_text == expected
