"""Add explicit PCM16 browser copies to an existing private listening package.

Preserves native WAV files byte for byte. Uses the same nearest-PCM16 rounding
policy as architecture_compare/package_listening.listening_copy, with an
explicit refusal whenever nearest quantization would require clipping. Loads no
models and uses no GPU, network, frontend, gain, normalization or resampling.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import wave

RECEIPT = 'browser-playback-receipt.json'
INTRO = ('<section id="browser-playback-compat"><p>Os players usam cópias PCM16 para '
         'compatibilidade com o navegador quando a conversão cabe na faixa sem clipping. '
         'A conversão aplica somente arredondamento: sem ganho, normalização, efeitos, '
         'dither, cortes ou reamostragem. Os WAVs originais, nas taxas e formatos nativos, '
         'permanecem intactos e disponíveis nos links de download. Quando a conversão é '
         'recusada, o player mantém o original; o suporte a WAV FLOAT depende do navegador. '
         'A aprovação da voz exige a escuta do proprietário.</p></section>')


def digest(path):
    with Path(path).open('rb') as inp:
        return hashlib.file_digest(inp, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.new')
    with temporary.open('w', encoding='utf-8') as out:
        os.chmod(temporary, 0o600)
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')
    temporary.replace(path)


def local_path(directory, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise ValueError('manifest paths must be relative POSIX paths')
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in rel.parts[0]:
        raise ValueError('manifest path escapes package')
    path = directory.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('manifest path resolves outside package')
    if any(p.is_symlink() for p in (path, *path.parents) if p != directory.parent):
        raise ValueError('package paths must not use symlinks')
    return path


def convert(source, target):
    """Return a portable receipt; leave target absent if conversion is unsafe."""
    import numpy as np
    import soundfile as sf
    info = sf.info(source)
    samples, rate = sf.read(source, dtype='float64', always_2d=True)
    row = {'native_subtype': info.subtype, 'sample_rate': rate, 'channels': info.channels,
           'frames': info.frames, 'pcm16': False, 'playback_created': False}
    if not samples.size or not np.isfinite(samples).all():
        return dict(row, status='declined_nonfinite_or_empty', conversion='original retained; no PCM16 copy created')
    row.update(native_peak=float(np.max(np.abs(samples))))
    if info.subtype == 'PCM_16':
        return dict(row, status='native_pcm16', pcm16=True, maximum_error_lsb=0.0,
                    maximum_error_amplitude=0.0, conversion='native PCM16 already compatible; no copy needed')
    quantized = np.rint(samples * 32768)
    if samples.min() < -1 or samples.max() >= 1 or quantized.min() < -32768 or quantized.max() > 32767:
        return dict(row, status='declined_overrange_or_rounding_overflow', conversion='original retained; nearest PCM16 quantization would require clipping')
    encoded = quantized.astype('<i2')
    error = float(np.max(np.abs(encoded.astype('float64') / 32768 - samples)))
    if error * 32768 > 0.500001:
        raise ValueError('PCM16 nearest rounding exceeded half an LSB')
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('preserve existing playback file; verify receipt before reusing')
    temporary = target.with_name(target.name + '.new')
    with wave.open(str(temporary), 'wb') as out:
        out.setnchannels(info.channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(encoded.tobytes())
    temporary.chmod(0o600)
    temporary.replace(target)
    return dict(row, status='converted', pcm16=True, playback_created=True,
                playback_subtype='PCM_16', maximum_error_lsb=error * 32768,
                maximum_error_amplitude=error,
                conversion='nearest PCM16; same rate/channels/frames; no gain, normalization, dither, effects, trim or resampling')


def rewrite_html(document, rows):
    by_native = {r['native_path']: r for r in rows}
    by_playback = {r['playback_path']: r for r in rows if r.get('playback_path')}
    document = re.sub(r'<section id="browser-playback-compat">.*?</section>', '', document, flags=re.S)
    document = re.sub(r'<a data-native-download="true"[^>]*>.*?</a>', '', document, flags=re.S)
    def replace(match):
        element = match.group(0)
        src = re.search(r'\bsrc="([^"]*)"', element)
        if src is None:
            return element
        path = html.unescape(src.group(1))
        row = by_native.get(path) or by_playback.get(path)
        if row is None:
            raise ValueError('HTML audio source is not a manifest native/playback WAV: ' + path)
        playback = row.get('playback_path', row['native_path'])
        element = element[:src.start(1)] + html.escape(playback, quote=True) + element[src.end(1):]
        return element + ('<a data-native-download="true" href="' + html.escape(row['native_path'], quote=True) +
                          '" download>Baixar original ' + html.escape(row['native_subtype']) + '</a>')
    document = re.sub(r'<audio\b[^>]*>.*?</audio>', replace, document, flags=re.S)
    if '</h1>' in document:
        return document.replace('</h1>', '</h1>' + INTRO, 1)
    return INTRO + document


def process(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    if any((p / '.git').exists() for p in (directory, *directory.parents)):
        raise ValueError('private listening package must be outside Git')
    manifest_path = directory / 'manifest-sha256.json'
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('expected path-to-SHA256 WAV manifest')
    natives = {k: v for k, v in manifest.items() if not k.startswith('playback/')}
    # Validate every original before creating copies or rewriting the index.
    for relative, sha in natives.items():
        source = local_path(directory, relative)
        if source.suffix.lower() != '.wav' or not source.is_file() or digest(source) != sha:
            raise ValueError('native WAV does not match manifest: ' + relative)
    prior = {}
    if (directory / RECEIPT).is_file():
        prior = {r['native_path']: r for r in json.loads((directory / RECEIPT).read_text())['files']}
    rows = []
    updated = dict(natives)
    for relative, sha in natives.items():
        source = local_path(directory, relative)
        target_relative = 'playback/' + relative
        target = local_path(directory, target_relative)
        old = prior.get(relative)
        if old is not None and old['native_sha256'] == sha:
            row = old.copy()
            if row.get('playback_created'):
                if not target.is_file() or digest(target) != row['playback_sha256']:
                    raise ValueError('saved playback copy differs from prior receipt')
        else:
            row = dict(native_path=relative, native_sha256=sha, **convert(source, target))
            row['playback_path'] = target_relative if row['playback_created'] else relative
            row['playback_sha256'] = digest(target) if row['playback_created'] else sha
        if digest(source) != sha:
            raise RuntimeError('native WAV changed during conversion')
        if row['playback_created']:
            updated[row['playback_path']] = row['playback_sha256']
        rows.append(row)
    receipt = {'schema': 'aegis.f5.browser_playback.v1', 'policy': 'native WAV bytes preserved; explicit nearest PCM16 copies only when finite and representable without gain/clipping; no resampling, normalization, effects, trimming or dither', 'files': rows,
               'native_files': len(natives), 'playback_copies': sum(r['playback_created'] for r in rows),
               'declined': sum(r['status'].startswith('declined') for r in rows)}
    document = rewrite_html((directory / 'index.html').read_text(), rows)
    temporary = directory / 'index.html.new'
    temporary.write_text(document, encoding='utf-8')
    temporary.chmod(0o600)
    temporary.replace(directory / 'index.html')
    write_json(directory / RECEIPT, receipt)
    write_json(manifest_path, updated)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listening-dir', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    result = process(args.listening_dir)
    print(json.dumps({k: result[k] for k in ('native_files', 'playback_copies', 'declined')}))


if __name__ == '__main__':
    main()
