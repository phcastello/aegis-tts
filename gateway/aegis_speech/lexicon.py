"""Versioned, non-regex pronunciation lexicon."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class LexiconEntry:
    entry_id: str
    kind: str
    value: str
    replacement: str
    case_sensitive: bool
    enabled: bool
    deny_after: tuple[str, ...] = ()


class PronunciationLexicon:
    def __init__(self, path: Path) -> None:
        self.path = path
        source = path.read_bytes()
        self.sha256 = hashlib.sha256(source).hexdigest()
        data = yaml.safe_load(source) or {}
        self.version = str(data.get("version", ""))
        if not self.version:
            raise LexiconError("lexicon version is required")
        self.entries = self._parse(data.get("entries", []))

    def _parse(self, raw: list[dict]) -> tuple[LexiconEntry, ...]:
        entries: list[LexiconEntry] = []
        seen: set[tuple[str, str, bool]] = set()
        for item in raw:
            match = item.get("match", {})
            entry = LexiconEntry(
                str(item.get("id", "")), str(match.get("type", "")), str(match.get("value", "")),
                str(item.get("replacement", "")), bool(match.get("case_sensitive", False)),
                bool(item.get("enabled", True)), tuple(item.get("contexts", {}).get("deny_after", [])),
            )
            if not entry.entry_id or not entry.value or not entry.replacement:
                raise LexiconError("lexicon entries require id, value and replacement")
            if entry.kind not in {"token", "phrase", "case_insensitive_phrase"}:
                raise LexiconError(f"unsupported lexicon match type: {entry.kind}")
            key = (entry.kind, entry.value.casefold() if not entry.case_sensitive else entry.value, entry.case_sensitive)
            if key in seen:
                raise LexiconError(f"duplicate lexicon entry: {entry.entry_id}")
            seen.add(key)
            if entry.value.casefold() == entry.replacement.casefold():
                raise LexiconError(f"self cycle in lexicon entry: {entry.entry_id}")
            entries.append(entry)
        # Longest literal first means a phrase cannot be consumed by a token.
        return tuple(sorted(entries, key=lambda item: len(item.value), reverse=True))

    def apply(self, text: str) -> tuple[str, list[str]]:
        applied: list[str] = []
        output = text
        replacements = {entry.value.casefold(): entry.replacement.casefold() for entry in self.entries if entry.enabled}
        for start in replacements:
            seen: set[str] = set()
            current = start
            while current in replacements:
                if current in seen:
                    raise LexiconError("lexicon replacement cycle")
                seen.add(current)
                current = replacements[current]
        for entry in self.entries:
            if not entry.enabled:
                continue
            flags = 0 if entry.case_sensitive else re.IGNORECASE
            pattern = re.escape(entry.value)
            if entry.kind == "token":
                pattern = rf"(?<!\w){pattern}(?!\w)"
            def substitute(match: re.Match[str]) -> str:
                tail = output[match.end():].lstrip().split(maxsplit=1)
                if tail and tail[0] in entry.deny_after:
                    return match.group(0)
                applied.append(entry.entry_id)
                return entry.replacement
            output = re.sub(pattern, substitute, output, flags=flags)
        return output, applied
