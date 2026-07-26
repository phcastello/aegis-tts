"""Deterministic pt-BR text-to-speech normalization; no caller supplied rules."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Callable


@dataclass(frozen=True)
class AppliedRule:
    rule_id: str
    count: int


@dataclass
class NormalizationResult:
    original_text: str
    normalized_text: str
    applied_rules: list[AppliedRule] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_ONES = ("zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove")
_TEENS = ("dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove")
_TENS = ("", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa")
_HUNDREDS = ("", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", "seiscentos", "setecentos", "oitocentos", "novecentos")
_MONTHS = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro")


def number_to_words(value: int) -> str:
    """Portuguese cardinal used in technical speech (bounded by input limits)."""
    if value < 0:
        return "menos " + number_to_words(-value)
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value - 10]
    if value < 100:
        return _TENS[value // 10] + ("" if value % 10 == 0 else " e " + _ONES[value % 10])
    if value == 100:
        return "cem"
    if value < 1000:
        return _HUNDREDS[value // 100] + ("" if value % 100 == 0 else " e " + number_to_words(value % 100))
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        prefix = "mil" if thousands == 1 else number_to_words(thousands) + " mil"
        return prefix if not rest else prefix + (" e " if rest < 100 else " ") + number_to_words(rest)
    if value < 1_000_000_000:
        millions, rest = divmod(value, 1_000_000)
        prefix = "um milhão" if millions == 1 else number_to_words(millions) + " milhões"
        return prefix if not rest else prefix + " " + number_to_words(rest)
    return str(value)


class SpeechTextNormalizer:
    """The ordered pipeline is intentionally code-owned and auditable."""

    def __init__(self, lexicon: Callable[[str], tuple[str, list[str]]] | None = None, max_characters: int = 12000) -> None:
        self.lexicon = lexicon
        self.max_characters = max_characters

    def normalize(self, text: str, *, locale: str = "pt-BR") -> NormalizationResult:
        if locale != "pt-BR":
            raise ValueError("only pt-BR is supported")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be empty")
        if len(text) > self.max_characters:
            raise ValueError("text exceeds maximum length")
        if "\x00" in text:
            raise ValueError("text contains NUL")
        out = unicodedata.normalize("NFC", text)
        rules: dict[str, int] = {}
        warnings: list[str] = []
        def replace(pattern: str, fn: Callable[[re.Match[str]], str], name: str, flags: int = 0) -> None:
            nonlocal out
            out, count = re.subn(pattern, fn, out, flags=flags)
            if count:
                rules[name] = rules.get(name, 0) + count

        # Controls are neither spoken nor allowed to change visual text order.
        out, removed = re.subn(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", out)
        if removed: rules["remove_invisible"] = removed
        out, spaces = re.subn(r"\s+", " ", out.strip())
        if spaces: rules["whitespace"] = spaces

        protected: list[str] = []
        def protect(value: str) -> str:
            protected.append(value)
            return chr(0xE000 + len(protected) - 1)

        # The interface can present links visually. Say so instead of reading a
        # noisy URL aloud, while preserving terminal punctuation for pacing.
        def replace_url(match: re.Match[str]) -> str:
            value = match.group(0)
            trailing = re.search(r"[.,;:!?]+$", value)
            return "você pode ver o link pela interface" + (trailing.group(0) if trailing else "")
        replace(r"https?://[^\s]+", replace_url, "replace_url", re.I)
        replace(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", lambda m: protect(m.group(0)), "protect_email", re.I)
        replace(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", lambda m: protect(m.group(0)), "protect_ip")
        replace(r"\b\d+(?:\.\d+){2,}\b", lambda m: protect(m.group(0)), "protect_version")

        # Remove presentational Markdown while retaining the text it surrounds.
        replace(r"(?<!\w)(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)", lambda m: m[2], "remove_markdown_emphasis")
        replace(r"[()]", lambda _m: " ", "remove_parentheses")

        def speak_date(day: str, month: str, year: str) -> str:
            try:
                parsed = date(int(year) if len(year) == 4 else 2000 + int(year), int(month), int(day))
            except ValueError:
                # Leave malformed dates untouched for the remaining numeric rules.
                return f"{day}/{month}/{year}"
            return f"{number_to_words(parsed.day)} de {_MONTHS[parsed.month - 1]} de {number_to_words(parsed.year)}"

        replace(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b", lambda m: speak_date(m[1], m[2], m[3]), "date")
        replace(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: speak_date(m[3], m[2], m[1]), "iso_date")
        replace(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", lambda m: f"{number_to_words(int(m[1]))} horas e {number_to_words(int(m[2]))} minutos", "time")
        replace(r"\b(\d+)h(\d{1,2})?\b", lambda m: f"{number_to_words(int(m[1]))} horas" + (f" e {number_to_words(int(m[2]))} minutos" if m[2] else ""), "duration")
        replace(r"([+-]?\d+(?:[.,]\d+)?)\s*%", lambda m: self._decimal(m[1]) + " por cento", "percentage")
        units = {"°C": "graus Celsius", "GB": "gigabytes", "MB": "megabytes", "KB": "kilobytes", "TB": "terabytes", "Hz": "hertz", "kHz": "quilohertz", "MHz": "megahertz", "GHz": "gigahertz", "W": "watts", "V": "volts", "A": "amperes", "km": "quilômetros", "cm": "centímetros", "mm": "milímetros", "m": "metros", "B": "bytes"}
        unit_re = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
        replace(rf"([+-]?\d+(?:[.,]\d+)?)\s*({unit_re})\b", lambda m: self._decimal(m[1]) + " " + units[m[2]], "unit")
        # A token containing both letters and digits is an identifier, not a
        # pronounceable word. Separate every character so the TTS spells it.
        replace(r"(?<!\w)(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*(?!\w)", lambda m: " ".join({"-": "hífen", "_": "sublinhado"}.get(char, char) for char in m.group(0)), "spell_alphanumeric_code")
        replace(r"(?<![\w.])([+-]?\d+[.,]\d+)(?!\w|[.,]\d)", lambda m: self._decimal(m[1]), "decimal")
        replace(r"(?<![\w.])-?\d+(?!\w|\.\d)", lambda m: number_to_words(int(m.group(0))), "integer")

        def restore(m: re.Match[str]) -> str:
            value = protected[int(m[1])]
            if "@" in value:
                local, domain = value.split("@", 1)
                return self._spell_technical(local) + " arroba " + self._spell_technical(domain)
            if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value):
                return " ponto ".join(number_to_words(int(part)) for part in value.split("."))
            if re.fullmatch(r"\d+(?:\.\d+){2,}", value):
                return " ponto ".join(number_to_words(int(part)) for part in value.split("."))
            if value.startswith("http"):
                value = re.sub(r"^https?://", "https dois pontos barra barra ", value)
            return self._spell_technical(value)
        def restore_marker(match: re.Match[str]) -> str:
            return restore_index(ord(match.group(0)) - 0xE000)
        def restore_index(index: int) -> str:
            value = protected[index]
            if "@" in value:
                local, domain = value.split("@", 1)
                return self._spell_technical(local) + " arroba " + self._spell_technical(domain)
            if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value):
                return " ponto ".join(number_to_words(int(part)) for part in value.split("."))
            if re.fullmatch(r"\d+(?:\.\d+){2,}", value):
                return " ponto ".join(number_to_words(int(part)) for part in value.split("."))
            if value.startswith("http"):
                value = re.sub(r"^https?://", "https dois pontos barra barra ", value)
            return self._spell_technical(value)
        replace(r"[\uE000-\uEFFF]", restore_marker, "restore_protected")
        if self.lexicon:
            out, applied = self.lexicon(out)
            for item in applied: rules["lexicon:" + item] = rules.get("lexicon:" + item, 0) + 1
        # Remaining uppercase tokens are acronyms (including IATA airport
        # codes). Apply this after the lexicon so configured pronunciations,
        # such as DNS and VPN, remain authoritative.
        replace(r"(?<!\w)[A-Z]{2,}(?!\w)", lambda m: " ".join(m.group(0)), "spell_uppercase_acronym")
        out = out.replace("/", " barra ").replace("=", " igual a ").replace("+", " mais ").replace("*", " vezes ")
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"\s+", " ", out).strip()
        if not out:
            raise ValueError("normalization produced empty text")
        return NormalizationResult(text, out, [AppliedRule(key, value) for key, value in rules.items()], warnings)

    @staticmethod
    def _decimal(value: str) -> str:
        sign = "menos " if value.startswith("-") else ""
        raw = value.lstrip("+-").replace(".", ",")
        if "," not in raw:
            return sign + number_to_words(int(raw))
        whole, fraction = raw.split(",", 1)
        return sign + number_to_words(int(whole)) + " vírgula " + " ".join(number_to_words(int(c)) for c in fraction)

    @staticmethod
    def _spell_technical(value: str) -> str:
        # Dots in a protected technical token are always pronounced as dots.
        value = re.sub(r"\d+", lambda match: number_to_words(int(match.group(0))), value)
        return value.replace(".", " ponto ").replace("-", " hífen ").replace("_", " sublinhado ")
