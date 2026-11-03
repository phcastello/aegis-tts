"""Experimental F5 speech text only; never imported by the production gateway.

The display string is retained byte for byte. Pronunciations are orthographic
prompts for the pinned grapheme F5 frontend, not phonemes or owner approval.
Unknown acronyms are left alone instead of guessed from a universal spelling
rule. The owner confirmed the Aegis target A-é-jis; orthographic realizations
still require listening to the actual F5 output.
"""
from __future__ import annotations

from datetime import date
import re
import unicodedata

from gateway.aegis_speech.normalization import number_to_words

VERSION = 'f5-refinement-1'
AEGIS_TARGET = 'A-é-jis'
LEXICON = {
    'Aegis': 'A éjis',
    'CPU': 'cê pê u',
    'API': 'a pê i',
    'CPF': 'cê pê efe',
    'SUS': 'sus',
    'URL': 'u erre ele',
    'HTTPS': 'agá tê tê pê ésse',
}
MONTHS = ('janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
          'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro')
UNITS = {'kHz': 'quilohertz', 'Hz': 'hertz', 'GHz': 'gigahertz',
         'MHz': 'megahertz', 'cm': 'centímetros', 'mm': 'milímetros',
         'GB': 'gigabytes', 'MB': 'megabytes', '°C': 'graus Celsius'}


def decimal_words(raw: str) -> str:
    """pt-BR thousands '.', decimal ','; decimal digits preserve precision."""
    raw = raw.replace('.', '')
    negative = raw.startswith('-')
    raw = raw.lstrip('+-')
    integer, separator, fraction = raw.partition(',')
    value = int(integer)
    if value >= 1_000_000_000:
        raise ValueError('numeric value exceeds experimental cardinal range')
    out = ('menos ' if negative else '') + number_to_words(value)
    return out + (' vírgula ' + ' '.join(number_to_words(int(c)) for c in fraction) if separator else '')


def normalize_text(text: str, *, lexicon: dict[str, str] | None = None) -> dict:
    if not isinstance(text, str) or not text.strip() or '\x00' in text:
        raise ValueError('nonempty string without NUL required')
    display = text
    out = unicodedata.normalize('NFC', text)
    rules, warnings, held = [], [], []

    def substitute(pattern, fn, rule, flags=0):
        nonlocal out
        out, count = re.subn(pattern, fn, out, flags=flags)
        if count:
            rules.append({'id': rule, 'count': count})

    def protect(value, reason):
        held.append(value)
        warnings.append(reason + ': ' + value)
        return chr(0xE000 + len(held) - 1)

    if re.search('[\uE000-\uF8FF]', out):
        raise ValueError('private-use characters are not supported')
    # Technical strings require application-specific semantics and are not
    # dismantled into numbers by this deliberately narrow experiment.
    substitute(r'https?://[^\s]+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b\d+(?:\.\d+){2,}\b',
               lambda m: protect(m[0], 'technical token left literal'), 'protect_technical')

    def speak_date(raw, day, month, year):
        if len(year) != 4:
            return protect(raw, 'ambiguous two-digit year left literal')
        try:
            parsed = date(int(year), int(month), int(day))
        except ValueError:
            return protect(raw, 'invalid date left literal')
        return f'{number_to_words(parsed.day)} de {MONTHS[parsed.month-1]} de {number_to_words(parsed.year)}'

    substitute(r'(?<!\w)(\d{4})-(\d{1,2})-(\d{1,2})(?!\w)',
               lambda m: speak_date(m[0], m[3], m[2], m[1]), 'iso_date')
    substitute(r'(?<!\w)(\d{1,2})/(\d{1,2})/(\d{2,4})(?!\w)',
               lambda m: speak_date(m[0], m[1], m[2], m[3]), 'date')

    def speak_clock(m):
        hour, minute = int(m[1]), int(m[2] or m[3] or 0)
        if hour > 23 or minute > 59:
            return protect(m[0], 'invalid clock time left literal')
        hour_words = {1: 'uma', 2: 'duas', 21: 'vinte e uma', 22: 'vinte e duas'}.get(hour, number_to_words(hour))
        return (f'{hour_words} ' + ('hora' if hour == 1 else 'horas') +
                (f' e {number_to_words(minute)} ' + ('minuto' if minute == 1 else 'minutos') if minute else ''))

    substitute(r'(?<!\w)(\d{1,2})(?:h(\d{2})?|:(\d{2}))(?!\w|\d)', speak_clock, 'clock')
    numeric = r'[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?'

    def currency(m):
        raw = m[1].replace('.', '')
        negative = raw.startswith('-')
        raw = raw.lstrip('+-')
        integer, _, fraction = raw.partition(',')
        if len(fraction) > 2:
            return protect(m[0], 'currency precision over two digits left literal')
        reais, cents = int(integer), int((fraction + '00')[:2])
        if reais >= 1_000_000_000:
            raise ValueError('currency exceeds experimental cardinal range')
        value = (number_to_words(reais) + (' real' if reais == 1 else ' reais'))
        if cents:
            value += ' e ' + number_to_words(cents) + (' centavo' if cents == 1 else ' centavos')
        return ('menos ' if negative else '') + value

    substitute(r'R\$\s*(' + numeric + r')(?![\w,]|\.\d)', currency, 'currency_brl')
    substitute(r'(?<![\w.])(' + numeric + r')\s*%', lambda m: decimal_words(m[1]) + ' por cento', 'percent')
    unit_pattern = '|'.join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
    substitute(r'(?<![\w.])(' + numeric + r')\s*(' + unit_pattern + r')(?!\w)',
               lambda m: decimal_words(m[1]) + ' ' + UNITS[m[2]], 'unit')
    # Do not split an alphanumeric identifier into unrelated spoken cardinals.
    substitute(r'(?<![\w.])(' + numeric + r')(?!\w|[.,]\d)', lambda m: decimal_words(m[1]), 'number')
    entries = LEXICON if lexicon is None else lexicon
    for token in sorted(entries, key=len, reverse=True):
        replacement = entries[token]
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError('lexicon replacements must be nonempty strings')
        substitute(r'(?<!\w)' + re.escape(token) + r'(?!\w)', lambda m, value=replacement: value,
                   'lexicon:' + token, re.I if token == 'Aegis' else 0)
    out = re.sub('[\uE000-\uF8FF]', lambda m: held[ord(m[0])-0xE000], out)
    return {'display_text': display, 'tts_text': out, 'version': VERSION,
            'applied_rules': rules, 'warnings': warnings,
            'aegis_target': AEGIS_TARGET, 'owner_target_confirmed': True,
            'owner_pronunciation_approved': False}


def pronunciation_cases() -> list[dict]:
    """Eight bounded clips: two name prompts, two acronym styles, two raw/A-Bs."""
    name = 'Aegis, confirme o registro e aguarde.'
    acronyms = 'A CPU consultou a API. O SUS confirmou o CPF, e a URL usa HTTPS.'
    numbers = 'O saldo é R$ 1.247,83 em 6 parcelas; a taxa caiu para 9,75%.'
    dates = 'A revisão será em 18/12/2026, às 07h35; mantenha 32 kHz.'
    alternate = {**LEXICON, 'CPU': 'ci pi iú', 'API': 'êi pi ai',
                 'SUS': 'ésse u ésse', 'URL': 'iú ar él'}
    cases = [
        ('name-separated', 'name', name, normalize_text(name)['tts_text']),
        ('name-hyphenated', 'name', name, normalize_text(name, lexicon={**LEXICON, 'Aegis': 'A-é-jis'})['tts_text']),
        ('acronyms-pt-word-sus', 'acronyms', acronyms, normalize_text(acronyms)['tts_text']),
        ('acronyms-mixed-spelled-sus', 'acronyms', acronyms, normalize_text(acronyms, lexicon=alternate)['tts_text']),
        ('numbers-raw', 'numbers', numbers, numbers),
        ('numbers-expanded', 'numbers', numbers, normalize_text(numbers)['tts_text']),
        ('dates-raw', 'dates', dates, dates),
        ('dates-expanded', 'dates', dates, normalize_text(dates)['tts_text']),
    ]
    return [{'id': identifier, 'group': group, 'display_text': display,
             'tts_text': spoken, 'owner_approval': False,
             'note': 'owner target A-é-jis; orthographic prompts pending listening; acronym choices evaluated per token'}
            for identifier, group, display, spoken in cases]
