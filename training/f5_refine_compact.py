"""Export one verified playback WAV per clip from a full private listening set.

No audio encoding, trimming, gain, model load, service operation or deletion.
Native FLOAT originals and the full archive remain untouched. Original waveform
measurements retain explicit native provenance while packaged hashes identify
exactly the selected PCM16 or retained FLOAT bytes actually exported.
"""
from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from training import f5_listening_compat as compat
from training.f5_refine_report import digest, private_path, verify_directory, write

DEFAULT_ROOT = Path('/var/tmp/aegisvoice-v2-refinement-20261103')
ARCHIVE_NAME = 'aegisvoice-v2-refinement-compact-20261103.zip'
POLICY = ('One WAV per distinct clip: existing nearest-PCM16 browser copy when '
    'conversion was safe; native PCM16 when already available; original FLOAT '
    'when PCM16 conversion was declined. Bytes copied without reencoding, gain, '
    'normalization, clipping, resampling, effects, dither or trimming. Original '
    'native waveform measurements and ASR receipts remain unchanged as source '
    'evidence; original natives and full archive remain in the private run root.')


def read(path):
    return json.loads(path.read_text())


def verify_source(source):
    artifacts = read(source / 'artifacts-sha256.json')
    for relative, sha in artifacts.items():
        path = compat.local_path(source, relative)
        if not path.is_file() or digest(path) != sha:
            raise ValueError('full listening artifact changed: ' + relative)
    verify_directory(source)
    return artifacts


def selection(source, browser):
    manifest = read(source / 'manifest-sha256.json')
    rows, seen_native = [], set()
    for row in browser['files']:
        native, native_sha = row['native_path'], row['native_sha256']
        if native in seen_native or manifest.get(native) != native_sha:
            raise ValueError('duplicate or mismatched full native receipt')
        seen_native.add(native)
        converted = row['status'] == 'converted'
        selected = row['playback_path'] if converted else native
        selected_sha = row['playback_sha256'] if converted else native_sha
        if converted != bool(row.get('playback_created')):
            raise ValueError('inconsistent conversion status')
        if manifest.get(selected) != selected_sha or digest(compat.local_path(source, selected)) != selected_sha:
            raise ValueError('selected playback does not match full manifest')
        # The destination keeps the clip's original relative name, with its
        # actual packaged format/hash identified explicitly in every receipt.
        rows.append(dict(original_native_path=native, original_native_sha256=native_sha,
            original_native_subtype=row['native_subtype'], source_selected_path=selected,
            source_selected_sha256=selected_sha, packaged_path=native,
            packaged_sha256=selected_sha,
            packaged_subtype='PCM_16' if converted else row['native_subtype'],
            sample_rate=row['sample_rate'], channels=row['channels'], frames=row['frames'],
            source_conversion_status=row['status'], maximum_error_lsb=row.get('maximum_error_lsb'),
            full_native_private_path=str(source / native)))
    if {r['original_native_path'] for r in rows} != {k for k in manifest if not k.startswith('playback/')}:
        raise ValueError('browser receipt does not cover full native manifest')
    return rows


def rewrite_html(document, chosen, source):
    by_source = {path: row for row in chosen for path in (row['original_native_path'], row['source_selected_path'])}
    # Remove only native-download anchors; other local evidence links remain.
    document = re.sub(r'<a\b(?=[^>]*\bdata-native-download="true")[^>]*>.*?</a>', '', document, flags=re.S)
    def audio(match):
        element = match.group(0)
        attr = re.search(r'\bsrc="([^"]*)"', element)
        if attr is None:
            raise ValueError('audio element lacks source')
        relative = html.unescape(attr.group(1))
        if relative not in by_source:
            raise ValueError('audio element is outside playback receipt')
        target = by_source[relative]['packaged_path']
        return element[:attr.start(1)] + html.escape(target, quote=True) + element[attr.end(1):]
    document = re.sub(r'<audio\b[^>]*>.*?</audio>', audio, document, flags=re.S)
    intro = ('<section id="browser-playback-compat"><p><b>Pacote compacto: um WAV por clipe.</b> '
        'Os players usam a cópia PCM16 já validada quando possível. Quando essa conversão exigiria clipping, '
        'o FLOAT original é mantido; seu suporte depende do navegador. Nenhum áudio foi cortado, '
        'normalizado ou reprocessado nesta exportação. As medições acústicas e o ASR pertencem aos originais preservados.</p>'
        '<p>Os nativos completos continuam em <code>' + html.escape(str(source)) +
        '</code>, fora deste ZIP. Os caminhos e hashes nativos são proveniência; '
        '<code>packaged_path/packaged_sha256</code> identificam os bytes efetivamente incluídos. '
        '<a href="compact-receipt.json">Recibo da seleção e quantização</a>.</p></section>')
    if 'id="browser-playback-compat"' not in document:
        raise ValueError('full listening policy section missing')
    return re.sub(r'<section id="browser-playback-compat">.*?</section>', lambda _: intro, document, count=1, flags=re.S)


def metrics_for_compact(metrics, chosen):
    by_native = {r['original_native_path']: r for r in chosen}
    result = []
    for metric in metrics:
        row = dict(metric)
        native = row.pop('native_path')
        selection = by_native[native]
        if row['sha256'] != selection['original_native_sha256']:
            raise ValueError('metric native SHA does not match browser receipt')
        row.update({k: selection[k] for k in ('original_native_path', 'original_native_sha256',
            'packaged_path', 'packaged_sha256', 'packaged_subtype')})
        row['sha256_scope'] = 'preserved original_native_sha256; use packaged_sha256 for the exported WAV'
        row['measurement_scope'] = 'original native waveform; ASR/timings unchanged; packaged playback may use nearest PCM16'
        result.append(row)
    return result


def build(root, archive=None):
    root = private_path(Path(root)).resolve(strict=True)
    source = root / 'listening'
    artifacts = verify_source(source)
    browser = read(source / compat.RECEIPT)
    chosen = selection(source, browser)
    target = root / 'listening-compact'
    if target.is_symlink():
        raise ValueError('compact output must not be a symlink')
    target.mkdir(mode=0o700, exist_ok=True)
    target.chmod(0o700)
    described, manifest = set(), {}
    # No conversion happens here: byte-identical copies of the previously
    # validated selected representation, preserving overshoot when necessary.
    for row in chosen:
        path = compat.local_path(target, row['packaged_path'])
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists() and digest(path) != row['packaged_sha256']:
            raise ValueError('existing compact WAV differs; refusing overwrite')
        if not path.exists():
            shutil.copyfile(compat.local_path(source, row['source_selected_path']), path)
            path.chmod(0o600)
        manifest[row['packaged_path']] = row['packaged_sha256']
        described.add(row['packaged_path'])
    # Preserve technical evidence unchanged; its native hashes/absolute source
    # paths refer to preserved source WAVs, not to compact quantized copies.
    for relative in artifacts:
        if relative.endswith('.wav') or relative in ('index.html', 'manifest-sha256.json',
                'metrics.json', 'metrics.csv', 'report.json', compat.RECEIPT):
            continue
        destination = compat.local_path(target, relative)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(compat.local_path(source, relative), destination)
        destination.chmod(0o600)
        described.add(relative)
    metrics = metrics_for_compact(read(source / 'metrics.json'), chosen)
    write(target / 'metrics.json', metrics)
    with (source / 'metrics.csv').open(newline='', encoding='utf-8-sig') as inp:
        fields = [k for k in csv.DictReader(inp).fieldnames if k != 'native_path']
    fields += ['original_native_path', 'original_native_sha256', 'packaged_path',
        'packaged_sha256', 'packaged_subtype', 'sha256_scope', 'measurement_scope']
    fields += [key for key in ('comparison_timing', 'streaming_scope') if key not in fields]
    with (target / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics)
    (target / 'metrics.csv').chmod(0o600)
    receipt = dict(schema='aegis.f5.refinement.compact.v1', policy=POLICY,
        source_full_listening=str(source), source_artifacts_manifest_sha256=digest(source / 'artifacts-sha256.json'),
        source_browser_receipt_sha256=digest(source / compat.RECEIPT),
        distinct_clips=len(chosen), packaged_wav_files=len(manifest), original_natives_preserved=True,
        original_waveform_metrics_unchanged=True, files=chosen)
    report = read(source / 'report.json')
    report['compact_delivery'] = {k:v for k,v in receipt.items() if k != 'files'}
    write(target / 'report.json', report)
    write(target / 'compact-receipt.json', receipt)
    write(target / 'manifest-sha256.json', manifest)
    (target / 'index.html').write_text(rewrite_html((source / 'index.html').read_text(), chosen, source))
    (target / 'index.html').chmod(0o600)
    described.update(('index.html', 'manifest-sha256.json', 'metrics.json', 'metrics.csv', 'report.json', 'compact-receipt.json'))
    verification = verify_directory(target)
    files = {relative:digest(compat.local_path(target,relative)) for relative in sorted(described)}
    write(target / 'artifacts-sha256.json', files)
    described.add('artifacts-sha256.json')
    archive = private_path(Path(archive) if archive else root / ARCHIVE_NAME)
    if archive.is_relative_to(source) or archive.is_relative_to(target):
        raise ValueError('archive must be outside listening directories')
    archive.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = archive.with_name(archive.name + '.new')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for relative in sorted(described):
            out.write(compat.local_path(target,relative),relative)
    temp.chmod(0o600)
    with zipfile.ZipFile(temp) as check:
        if check.testzip() is not None or set(check.namelist()) != described:
            raise ValueError('compact ZIP CRC or entry set mismatch')
        if sum(name.endswith('.wav') for name in check.namelist()) != len(chosen):
            raise ValueError('compact ZIP must contain one WAV per distinct clip')
        if any(digest_value != hashlib.sha256(check.read(relative)).hexdigest() for relative,digest_value in files.items()):
            raise ValueError('compact ZIP entry SHA mismatch')
    temp.replace(archive)
    # Prove the source package remained unchanged throughout this export.
    verify_source(source)
    verification.update(archive=str(archive),archive_bytes=archive.stat().st_size,
        archive_sha256=digest(archive),archive_crc_verified=True,archive_entry_hashes_verified=True,
        compact_wav_files=len(manifest),original_natives_preserved=True,
        source_files_unchanged=True,pcm16_selected=sum(r['packaged_subtype']=='PCM_16' for r in chosen),
        float_originals_retained=sum(r['packaged_subtype']=='FLOAT' for r in chosen))
    write(root / 'compact-package-verification.json',verification)
    return verification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--archive',type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    print(json.dumps(build(args.root,args.archive),ensure_ascii=False))


if __name__=='__main__':
    main()
