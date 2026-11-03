"""Lossless text chunks and frame-aligned F5/Vocos inference helpers.

No model imports, weights, service control, VAD, acoustic effects or GPU calls.
The caller supplies already generated [batch,time,mel] tensors to the optional
vocoder adapter. Prefix removal is exactly the known conditioning context on
the STFT sample grid; no generated tail or consonant is detected or trimmed.
"""
from __future__ import annotations

import math
import re

RATE, HOP = 24000, 256
DEFAULT_LOOKBACK_FRAMES = 32
ABBREVIATIONS = frozenset(('sr', 'sra', 'srta', 'dr', 'dra', 'prof', 'profa',
                           'av', 'rod', 'etc', 'ex', 'pág', 'págs', 'art', 'arts',
                           'nº', 'núm', 'eng', 'engª'))


def split_sentences(text: str) -> list[str]:
    """Return exact source spans, including punctuation and whitespace.

    Delimiters are terminal .!? followed by whitespace/end (with closing quotes
    retained). Decimal/thousands separators, common PT-BR abbreviations and
    initials are protected. This is a conservative text heuristic, not a full
    Portuguese parser. Joining the returned strings always reproduces input.
    """
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if not text:
        return []
    chunks, start = [], 0
    for match in re.finditer(r'[.!?]+["\'»”’\)\]]*(?:\s+|$)', text):
        first = match.start()
        if text[first] == '.':
            before = text[:first]
            token = re.search(r'([^\s]+)$', before)
            word = token.group(1) if token else ''
            if word.casefold().rstrip('.') in ABBREVIATIONS:
                continue
            if len(word) == 1 and word.isalpha() and word.isupper():
                continue
        end = match.end()
        chunks.append(text[start:end])
        start = end
    if start < len(text):
        chunks.append(text[start:])
    if ''.join(chunks) != text:
        raise AssertionError('sentence segmentation lost source content')
    return chunks


def linguistic_chunks(text: str, *, max_chars=180, max_words=32) -> list[str]:
    """Prefer whole sentences, then clause punctuation, then a word boundary.

    Each source character occurs in exactly one chunk. No punctuation is added,
    rewritten or discarded. A single token exceeding max_chars fails explicitly
    instead of truncating it. Limits are text limits, not audio-duration claims.
    """
    if not isinstance(max_chars, int) or max_chars < 1 or not isinstance(max_words, int) or max_words < 1:
        raise ValueError('positive integer chunk limits required')
    chunks = []
    for sentence in split_sentences(text):
        rest = sentence
        while len(rest) > max_chars or len(rest.split()) > max_words:
            spans = list(re.finditer(r'\S+\s*', rest))
            candidates = [m.end() for i, m in enumerate(spans, 1) if i <= max_words and m.end() <= max_chars]
            if not candidates:
                raise ValueError('a source token/whitespace span exceeds max_chars; refusing content loss')
            limit = candidates[-1]
            clause = [m.end() for m in re.finditer(r'[,;:—–]\s+', rest[:limit])]
            # Avoid a tiny prefix when a later legal word boundary is available.
            eligible = [end for end in clause if len(rest[:end].split()) >= 3]
            end = eligible[-1] if eligible else limit
            chunks.append(rest[:end])
            rest = rest[end:]
        if rest:
            chunks.append(rest)
    if ''.join(chunks) != text:
        raise AssertionError('linguistic segmentation lost source content')
    return chunks


def centered_mel_frames(wave_samples: int, *, hop_length=HOP) -> int:
    """Frame count for the pinned F5 center=True MelSpec (no STFT padding)."""
    if not isinstance(wave_samples, int) or wave_samples < 1 or hop_length < 1:
        raise ValueError('positive waveform sample count and hop required')
    return wave_samples // hop_length + 1


def duration_for_chunk(text: str, conditioning_frames: int, *, target_wpm: float,
                       max_generated_seconds=15.0, max_total_seconds=90.0,
                       sample_rate=RATE, hop_length=HOP) -> dict:
    """Independent per-chunk neural duration; never time-stretch or crop audio.

    A centered Vocos ISTFT returns (T-1)*hop samples. The +1 mel endpoint here
    ensures the generated waveform, after conditioning removal, has the intended
    quantized duration. A caller must split an overlong chunk rather than cap its
    duration silently. No prior chunk's duration or waveform affects this budget.
    """
    words = len(text.split())
    if not words or not isinstance(conditioning_frames, int) or conditioning_frames < 1 or not math.isfinite(target_wpm) or target_wpm <= 0:
        raise ValueError('nonempty text, conditioning frames and positive finite word rate required')
    if sample_rate < 1 or hop_length < 1 or max_generated_seconds <= 0 or max_total_seconds <= 0:
        raise ValueError('positive audio rate, hop and duration bounds required')
    requested = words * 60.0 / target_wpm
    generated_hops = max(1, round(requested * sample_rate / hop_length))
    generated_seconds = generated_hops * hop_length / sample_rate
    total = conditioning_frames + generated_hops + 1
    if requested > max_generated_seconds or generated_seconds > max_generated_seconds:
        raise ValueError('chunk exceeds maximum generated duration; split text instead of truncating speech')
    if total * hop_length / sample_rate > max_total_seconds:
        raise ValueError('chunk plus conditioning exceeds maximum total duration')
    return {'word_count': words, 'target_wpm': target_wpm, 'requested_generated_seconds': requested,
            'conditioning_frames': conditioning_frames, 'generated_wave_samples': generated_hops * hop_length,
            'generated_wave_seconds': generated_seconds, 'total_frames': total,
            'duration_policy': 'per-chunk word-rate neural budget, centered ISTFT endpoint included; no audio stretching/trimming'}


def generated_crop_plan(total_frames: int, conditioning_frames: int, *,
                        lookback_frames=DEFAULT_LOOKBACK_FRAMES, hop_length=HOP) -> dict:
    """Plan sample-exact context removal on the pinned center=True Vocos grid."""
    if not all(isinstance(x, int) for x in (total_frames, conditioning_frames, lookback_frames, hop_length)):
        raise ValueError('integer frame counts and hop required')
    if conditioning_frames < 1 or total_frames - conditioning_frames < 2 or lookback_frames < 0 or hop_length < 1:
        raise ValueError('positive conditioning, at least two output frames and nonnegative context required')
    start = max(0, conditioning_frames - lookback_frames)
    context = conditioning_frames - start
    return {'mel_start_frame': start, 'conditioning_frames': conditioning_frames,
            'lookback_frames_requested': lookback_frames, 'lookback_frames_used': context,
            'decode_frames': total_frames - start, 'remove_prefix_samples': context * hop_length,
            'decode_wave_samples': (total_frames - start - 1) * hop_length,
            'generated_wave_samples': (total_frames - conditioning_frames - 1) * hop_length,
            'hop_length': hop_length, 'istft_padding': 'center',
            'crop_policy': 'remove exactly known conditioning context on mel-grid; no VAD, tail cut, gain, fade or resampling'}


def decode_generated(vocoder, generated, conditioning_frames: int, *,
                     lookback_frames=DEFAULT_LOOKBACK_FRAMES, hop_length=HOP):
    """Decode with authentic left context; return (generated waveform, receipt).

    generated must use CFM's [batch,time,mel] layout. Use cond.shape[1] for
    conditioning_frames; floor(original_wave_samples/hop) is one frame shorter
    for the pinned center=True mel and would leak a copied prompt frame.
    """
    if len(generated.shape) != 3:
        raise ValueError('CFM output must have [batch,time,mel] layout')
    plan = generated_crop_plan(int(generated.shape[1]), conditioning_frames,
                               lookback_frames=lookback_frames, hop_length=hop_length)
    decoded = vocoder.decode(generated[:, plan['mel_start_frame']:, :].transpose(1, 2).float())
    if int(decoded.shape[-1]) != plan['decode_wave_samples']:
        raise ValueError('decoder output length differs from pinned center=True ISTFT contract')
    output = decoded[..., plan['remove_prefix_samples']:]
    if int(output.shape[-1]) != plan['generated_wave_samples']:
        raise AssertionError('context removal changed generated duration')
    return output, plan
