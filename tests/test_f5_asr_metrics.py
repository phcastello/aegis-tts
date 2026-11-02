from training.f5_asr_metrics import plain_word_comparison, plain_word_tokens


def test_accents_punctuation_case_and_word_level_edits():
    row = plain_word_comparison('NÃO, Júlia! Três opções.', 'nao julia tres opcoes')
    assert row['word_error_rate'] == 0
    assert row['reference_tokens'] == row['hypothesis_tokens'] == ['nao', 'julia', 'tres', 'opcoes']
    assert row['perceptual_approval'] == 'pending owner listening'
    changed = plain_word_comparison('uma porta azul', 'uma janela')
    assert changed['word_edits'] == 2
    assert changed['word_error_rate'] == 2 / 3


def test_aegis_name_is_never_rewritten_by_gateway_lexicon():
    assert plain_word_tokens('Aegis, confirme.') == ['aegis', 'confirme']
    assert plain_word_comparison('Aegis, confirme.', 'aegis confirme')['word_edits'] == 0
    different = plain_word_comparison('Aegis, confirme.', 'Égide, confirme.')
    assert different['word_edits'] == 1
    assert different['reference_tokens'][0] == 'aegis'
    assert different['hypothesis_tokens'][0] == 'egide'
    assert 'name variations' in different['metric_policy']['interpretation']


def test_digit_vs_spoken_number_flags_ambiguity_without_rewriting():
    row = plain_word_comparison('O saldo é R$ 1.247,83.', 'O saldo é mil duzentos e quarenta e sete reais e oitenta e três centavos.')
    assert row['reference_tokens'] == ['o', 'saldo', 'e', 'r', '1', '247', '83']
    assert row['word_edits'] > 0
    assert row['numeric_format_ambiguity'] is True
    assert any('numeric_format_ambiguity' in flag for flag in row['metric_flags'])
    assert 'not a direct pronunciation' in row['metric_policy']['interpretation']
    equal = plain_word_comparison('São 12 opções.', 'sao 12 opcoes')
    assert equal['word_edits'] == 0
    assert equal['numeric_format_ambiguity'] is False
