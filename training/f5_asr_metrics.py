"""Plain ASR word comparison without production text/number/lexicon rewriting.

This module has no audio, model, gateway, NumPy or CUDA dependencies. Benchmark
inputs remain verbatim; approved spoken number cases remain separate inputs.
"""
from __future__ import annotations

import re
import unicodedata

METRIC_POLICY = {
    'normalization': 'NFD, remove combining accent marks, casefold, split punctuation/whitespace; preserve letters and digits',
    'rewriting': 'none: no number expansion, currency conversion, acronym expansion, name/lexicon substitution or gateway frontend',
    'wer': 'word-level Levenshtein edits / max(1, reference token count); insertions may make WER greater than 1',
    'interpretation': 'Numeric formatting, acronym and name variations can create ASR text mismatches. WER is an auxiliary transcription comparison, not a direct pronunciation or voice-quality score.',
    'pass_meaning': 'Normalized token equality only. Does not approve intelligibility, identity, robotic effects or perceptual quality.',
    'human_approval': 'Requires owner listening; never inferred from ASR metrics.',
}


def plain_word_tokens(text: str) -> list[str]:
    """Remove only accents, case and separators; retain numeric digit spelling."""
    if not isinstance(text, str):
        raise TypeError('ASR comparison expects strings')
    normalized = ''.join(
        char for char in unicodedata.normalize('NFD', text.casefold())
        if not unicodedata.category(char).startswith('M')
    )
    return re.findall(r'[^\W_]+', normalized, flags=re.UNICODE)


def plain_word_comparison(expected: str, recognized: str) -> dict:
    """Compare reference/hypothesis tokens without changing either source text."""
    reference, hypothesis = plain_word_tokens(expected), plain_word_tokens(recognized)
    previous = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(hypothesis, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + (token != other)))
        previous = current
    edits = previous[-1]
    reference_digits = [t for t in reference if any(c.isdecimal() for c in t)]
    hypothesis_digits = [t for t in hypothesis if any(c.isdecimal() for c in t)]
    numeric_ambiguity = bool(reference_digits or hypothesis_digits) and reference_digits != hypothesis_digits
    flags = []
    if numeric_ambiguity:
        flags.append('numeric_format_ambiguity: digit-bearing tokens differ; review raw versus spoken wording before attributing an error to pronunciation')
    if not reference:
        flags.append('empty_reference: WER denominator clamped to one')
    return {
        'word_error_rate': edits / max(1, len(reference)),
        'word_edits': edits,
        'reference_tokens': reference,
        'hypothesis_tokens': hypothesis,
        # Existing reports use these names; both pairs describe the same tokens.
        'expected_tokens': reference,
        'recognized_tokens': hypothesis,
        'content_check': 'pass' if edits == 0 else 'review_required',
        'numeric_format_ambiguity': numeric_ambiguity,
        'metric_flags': flags,
        'metric_policy': METRIC_POLICY.copy(),
        'perceptual_approval': 'pending owner listening',
    }
