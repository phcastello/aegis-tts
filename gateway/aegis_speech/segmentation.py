"""Sentence segmentation after normalization, preserving deterministic spans."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .normalization import NormalizationResult


@dataclass(frozen=True)
class SpeechSegmentDraft:
    segment_index: int
    original_span: tuple[int, int]
    normalized_span: tuple[int, int]
    original_text: str
    normalized_text: str
    split_reason: str


class SpeechSegmenter:
    def __init__(self, min_chars: int = 20, preferred_max_chars: int = 180, hard_max_chars: int = 320, min_words: int = 3, hard_max_words: int = 60) -> None:
        self.min_chars, self.preferred_max_chars, self.hard_max_chars = min_chars, preferred_max_chars, hard_max_chars
        self.min_words, self.hard_max_words = min_words, hard_max_words

    def segment(self, normalization: NormalizationResult) -> list[SpeechSegmentDraft]:
        text = normalization.normalized_text.strip()
        protected = self._protect_non_boundaries(text)
        parts = [(self._restore(part).strip(), reason) for part, reason in self._boundaries(protected)]
        parts = [part for part in parts if part[0]]
        merged: list[tuple[str, str]] = []
        for part, reason in parts:
            if merged and (len(merged[-1][0]) < self.min_chars or len(merged[-1][0].split()) < self.min_words) and len(merged[-1][0]) + len(part) + 1 <= self.preferred_max_chars:
                merged[-1] = (merged[-1][0] + " " + part, "merged_short_sentences")
            else:
                merged.append((part, reason))
        output: list[SpeechSegmentDraft] = []
        cursor = 0
        original_cursor = 0
        for item, reason in merged:
            for fragment, fragment_reason in self._split_long(item, reason):
                start = text.find(fragment, cursor)
                if start < 0: start = cursor
                end = start + len(fragment)
                cursor = end
                # Exact original mapping is unavailable after number expansion;
                # retain a monotonic source span for UI/audit consumers.
                source_start = normalization.original_text.find(fragment, original_cursor)
                if source_start < 0: source_start = original_cursor
                source_end = min(len(normalization.original_text), max(source_start, source_start + len(fragment)))
                original_cursor = source_end
                output.append(SpeechSegmentDraft(len(output), (source_start, source_end), (start, end), normalization.original_text[source_start:source_end], fragment, fragment_reason))
        if not output:
            raise ValueError("segmentation produced no segments")
        return output

    @staticmethod
    def _protect_non_boundaries(text: str) -> str:
        # Normalizer normally removes these as technical tokens; this also makes
        # standalone segmenter use safe for callers and tests.
        items: list[str] = []
        def hold(match: re.Match[str]) -> str:
            items.append(match.group(0))
            return f"¤{len(items)-1}¤"
        result = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b\d+(?:\.\d+){1,}\b|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|https?://\S+", hold, text, flags=re.I)
        # Store on a reversible marker object encoded in-band.
        return result + "".join(f"\x00{i}:{value}\x00" for i, value in enumerate(items))

    @staticmethod
    def _restore(text: str) -> str:
        suffix = re.findall(r"\x00(\d+):(.*?)\x00", text)
        clean = re.sub(r"\x00\d+:.*?\x00", "", text)
        for index, value in suffix:
            clean = clean.replace(f"¤{index}¤", value)
        return clean

    def _boundaries(self, protected: str) -> list[tuple[str, str]]:
        # Keep quote/parens punctuation with the sentence. Semicolon/colon are
        # secondary boundaries only when the fragment is already substantial.
        body = protected.split("\x00", 1)[0]
        suffix = protected[len(body):]
        chunks = re.split(r"(?<=[.!?…])(?:[”’'\")\]]*)\s+|(?<=[;:])\s+", body)
        reasons = ["sentence_boundary" if re.search(r"[.!?…][”’'\")\]]*$", c) else "secondary_punctuation" for c in chunks]
        if chunks:
            chunks[-1] += suffix
        return list(zip(chunks, reasons))

    def _split_long(self, text: str, reason: str) -> list[tuple[str, str]]:
        if len(text) <= self.hard_max_chars and len(text.split()) <= self.hard_max_words:
            return [(text, reason)]
        pieces: list[tuple[str, str]] = []
        remaining = text
        while len(remaining) > self.hard_max_chars or len(remaining.split()) > self.hard_max_words:
            limit = min(self.preferred_max_chars, self.hard_max_chars)
            cut = max(remaining.rfind(mark, 0, limit) for mark in (",", ";", ":", " "))
            if cut <= 0: cut = min(limit, len(remaining))
            pieces.append((remaining[:cut + (1 if remaining[cut:cut+1] != " " else 0)].strip(), "secondary_punctuation" if remaining[cut:cut+1] in ",;:" else "hard_length_limit"))
            remaining = remaining[cut + 1:].strip()
        if remaining: pieces.append((remaining, "final_fragment"))
        return pieces
