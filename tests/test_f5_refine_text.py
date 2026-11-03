"""Speech-only lexical choices and numeric semantics, without model imports."""
import pytest

from training.f5_refine_text import normalize_text, pronunciation_cases


def test_display_is_byte_exact_and_name_target_is_explicit():
    text = '  Aegis, confirme o cafe\u0301!\n'
    result = normalize_text(text)
    assert result['display_text'].encode() == text.encode()
    assert result['tts_text'] == '  A éjis, confirme o café!\n'
    assert result['aegis_target'] == 'A-é-jis'
    assert result['owner_target_confirmed'] is True
    assert result['owner_pronunciation_approved'] is False
    assert 'Égide' not in result['tts_text']


def test_acronyms_have_individual_choices_and_literal_token_boundaries():
    text = 'CPU, API, CPF, SUS, URL e HTTPS; AegisVoice, API2, cpu, suspeito.'
    spoken = normalize_text(text)['tts_text']
    assert spoken == ('cê pê u, a pê i, cê pê efe, sus, u erre ele e agá tê tê pê ésse; '
                      'AegisVoice, API2, cpu, suspeito.')
    assert normalize_text('aegis e AEGIS')['tts_text'] == 'A éjis e A éjis'
    assert normalize_text('HTTP GPU')['tts_text'] == 'HTTP GPU'  # No universal guessing.


def test_money_thousands_precision_percent_and_units_do_not_lose_value():
    raw = 'R$ 1.247,83 em 6 parcelas; -12,50%, 32 kHz e 3,2 cm.'
    result = normalize_text(raw)
    assert result['display_text'] == raw
    assert result['tts_text'] == ('mil duzentos e quarenta e sete reais e oitenta e três centavos em seis parcelas; '
                                  'menos doze vírgula cinco zero por cento, trinta e dois quilohertz e três vírgula dois centímetros.')
    assert normalize_text('R$ 1,01 e R$ 0,05.')['tts_text'] == 'um real e um centavo e zero reais e cinco centavos.'
    assert normalize_text('R$ 1,005')['tts_text'] == 'R$ 1,005'
    assert normalize_text('R$ 1,005')['warnings']


def test_dates_and_clocks_expand_valid_values_and_retain_invalid_values():
    spoken = normalize_text('18/12/2026 às 07h35; 2028-02-29 às 01:01 e 22h00.')['tts_text']
    assert spoken == ('dezoito de dezembro de dois mil e vinte e seis às sete horas e trinta e cinco minutos; '
                      'vinte e nove de fevereiro de dois mil e vinte e oito às uma hora e um minuto e vinte e duas horas.')
    raw = '31/02/2026, 18/12/26, 25:00 e 07h65.'
    result = normalize_text(raw)
    assert result['tts_text'] == raw
    assert len(result['warnings']) == 4


def test_technical_strings_and_identifiers_remain_literal():
    raw = 'https://aegis.example/v1.2.3 API2 e versão 1.2.3; 192.168.0.1.'
    assert normalize_text(raw)['tts_text'] == raw


def test_listening_cases_are_small_paired_and_do_not_approve_outputs():
    cases = pronunciation_cases()
    assert len(cases) == 8
    assert len({c['id'] for c in cases}) == 8
    for group in ('name', 'acronyms', 'numbers', 'dates'):
        pair = [c for c in cases if c['group'] == group]
        assert len(pair) == 2
        assert pair[0]['display_text'] == pair[1]['display_text']
        assert pair[0]['tts_text'] != pair[1]['tts_text']
    assert all(c['owner_approval'] is False for c in cases)
    assert 'sus' in cases[2]['tts_text'] and 'ésse u ésse' in cases[3]['tts_text']


@pytest.mark.parametrize('text', ['', '   ', '\x00', '\ue000', None])
def test_invalid_input_is_rejected(text):
    with pytest.raises(ValueError):
        normalize_text(text)
