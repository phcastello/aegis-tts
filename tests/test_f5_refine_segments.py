"""Content preservation and exact frame/sample contracts; no neural runtime."""
import pytest

from training.f5_refine_segments import (
    centered_mel_frames, duration_for_chunk, generated_crop_plan,
    linguistic_chunks, split_sentences,
)


def test_sentence_spans_preserve_punctuation_numbers_quotes_and_abbreviations():
    text = '  A Sra. Joana pagou R$ 1.247,83. “Está tudo correto?” Sim... Aguarde!  '
    chunks = split_sentences(text)
    assert ''.join(chunks) == text
    assert chunks == ['  A Sra. Joana pagou R$ 1.247,83. ', '“Está tudo correto?” ', 'Sim... ', 'Aguarde!  ']
    assert split_sentences('Sem pontuação final') == ['Sem pontuação final']


def test_linguistic_chunks_keep_every_character_and_respect_limits():
    text = ('Antes de enviar, confira o relatório de Júlia e João; preserve a versão original, '
            'depois confirme a última linha e aguarde a resposta. Não apague o arquivo!')
    chunks = linguistic_chunks(text, max_chars=72, max_words=12)
    assert ''.join(chunks) == text
    assert len(chunks) > 2
    assert all(len(c) <= 72 and len(c.split()) <= 12 for c in chunks)
    assert any(c.rstrip().endswith((',', ';')) for c in chunks)
    with pytest.raises(ValueError, match='refusing content loss'):
        linguistic_chunks('palavraimpossivelmentelonga', max_chars=10)


def test_centered_frames_and_context_sample_alignment():
    assert centered_mel_frames(242472) == 948
    assert 242472 // 256 == 947  # Legacy crop leaked one conditioned mel frame.
    aligned = generated_crop_plan(1500, 948, lookback_frames=0)
    contextual = generated_crop_plan(1500, 948, lookback_frames=32)
    assert contextual['mel_start_frame'] == 916
    assert contextual['remove_prefix_samples'] == 8192
    assert contextual['generated_wave_samples'] == aligned['generated_wave_samples'] == 551 * 256
    assert contextual['decode_wave_samples'] - 8192 == aligned['decode_wave_samples']
    short = generated_crop_plan(100, 10, lookback_frames=32)
    assert short['mel_start_frame'] == 0
    assert short['lookback_frames_used'] == 10
    with pytest.raises(ValueError):
        generated_crop_plan(948, 948)


def test_duration_is_independent_per_chunk_bounded_and_uses_center_endpoint():
    first = duration_for_chunk('Confira a porta azul.', 948, target_wpm=120)
    second = duration_for_chunk('Confira a porta azul.', 948, target_wpm=120)
    assert first == second
    assert first['generated_wave_samples'] == 48128
    assert abs(first['generated_wave_seconds'] - 2.0) <= 128/24000 + 1e-12
    assert first['total_frames'] == 948 + round(48000/256) + 1
    crop = generated_crop_plan(first['total_frames'], 948)
    assert crop['generated_wave_samples'] == first['generated_wave_samples']
    with pytest.raises(ValueError, match='split text'):
        duration_for_chunk('Uma frase longa com palavras demais.', 948, target_wpm=60, max_generated_seconds=3)
