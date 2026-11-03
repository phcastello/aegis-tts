"""CPU-only, compact offline listening evidence for inference refinement.

Never loads models, runs ASR, uses CUDA, changes production, or grants perceptual
approval. Native audio bytes are preserved; browser copies use the existing
strict nearest-PCM16 adapter. Only comparison outputs are copied, never individual
sentence blocks or stability repeats. Operational secrets are not exported.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import statistics
from urllib.parse import unquote, urlsplit
import zipfile

from training import f5_listening_compat as compat
from training.f5_refine_audit import asr_for
from training.f5_report import operation_windows

DEFAULT_ROOT = Path('/var/tmp/aegisvoice-v2-refinement-20261103')
DEFAULT_ARCHIVE = DEFAULT_ROOT / 'aegisvoice-v2-refinement-20261103.zip'
CASES = ('ordinary', 'lexical', 'long')
STAGES = ('controlled', 'references', 'pronunciation', 'alignment', 'duration')
MEASURES = ('total_seconds', 'audio_seconds', 'rtf', 'first_pcm_ms',
            'first_useful_pcm_available_ms', 'onset_audio_offset_ms',
            'estimated_first_audible_playback_ms', 'peak', 'outside_unit',
            'word_error_rate', 'word_edits')
PERCEPTUAL_FIELDS = ('intelligibility_1_5', 'robotic_identity_1_5',
                     'pronunciation_1_5', 'acoustic_artifacts_0_4',
                     'continuity_1_5', 'notes')
OPTIONAL_EVIDENCE = ('selected-config.json', 'technical-audit.json',
    'segmentation-source-review.json', 'training-adaptation-review.json',
    'duration-diagnostic.json', 'alignment.json', 'qwen/frontend-audit.json')


def digest(path):
    with Path(path).open('rb') as inp:
        return hashlib.file_digest(inp, 'sha256').hexdigest()


def load(root, name, default=None):
    path = root / name
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, value):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def private_path(path):
    if path.is_symlink():
        raise ValueError('private output must not be a symlink')
    path = path.expanduser().resolve()
    if any((p / '.git').exists() for p in (path, *path.parents)):
        raise ValueError('private artifacts must be outside Git')
    return path


def choose_rows(rows, qwen_rows):
    """Require the requested small matched cohort; expose only useful contrasts."""
    chosen = [dict(r) for r in [*rows, *qwen_rows] if r.get('stage') in STAGES]
    ids = [r['id'] for r in chosen]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate listening IDs')
    for case in CASES:
        expected = {'qwen-' + case, 'original-' + case + '-16',
                    'original-' + case + '-8', 'selected-' + case}
        actual = {r['id'] for r in chosen if r['stage'] == 'controlled' and r['case'] == case}
        if expected != actual:
            raise ValueError(f'incomplete controlled cohort for {case}: {actual}')
    return chosen


def metric_row(row, asr, frontend=None):
    value = {k: row.get(k) for k in ('id', 'stage', 'label', 'case', 'reference_id', 'nfe', 'seed', 'blocks', *MEASURES)}
    value['display_text'] = row['text']
    value['tts_text'] = row.get('tts_text', row['text'])
    value['request_text'] = value['tts_text']
    if frontend is not None:
        if frontend['audio_sha256'] != row['sha256'] or frontend['request_text'] != value['request_text']:
            raise ValueError('Qwen frontend receipt does not match captured request/audio')
        value['tts_text'] = frontend['tts_text']
        value['frontend_audited'] = True
        value['upstream_segments'] = frontend.get('segments')
        value['asr_comparability'] = frontend.get('wer_scope')
    value['effective_text'] = value['tts_text'] if row['id'].startswith('qwen-') else value['tts_text'].lower()
    value['sha256'] = row['sha256']
    value['native_path'] = row.get('native_path')
    value['boundaries'] = row.get('boundaries')
    value['streaming_scope'] = ('HTTP PCM, production frontend/DSP' if row['id'].startswith('qwen-')
        else 'local complete sentence buffers, sequential FIFO availability' if row.get('blocks')
        else 'local complete utterance buffer after ODE + Vocos')
    value['comparison_timing'] = row.get('comparison_timing')
    if asr is not None:
        for key in ('word_error_rate', 'word_edits', 'hypothesis'):
            value[key] = asr.get(key)
        value['asr_expected'] = asr.get('expected')
    value['perceptual_status'] = 'pending_owner_listening'
    return value


def table_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, places=3):
    return '—' if value is None else f'{value:.{places}f}'


def clip(row):
    label = row.get('label', row['id'])
    details = []
    for key, title in [('reference_id', 'ref'), ('nfe', 'NFE')]:
        if row.get(key) is not None:
            details.append(f'{title}={row[key]}')
    timings = ('RTF ' + number(row.get('rtf')) + '; PCM útil ' +
               number(row.get('first_useful_pcm_available_ms'), 1) + ' ms; áudio ' +
               number(row.get('audio_seconds')) + ' s')
    return (f'<article id="clip-{esc(row["id"])}"><h3>{esc(label)}</h3>'
        f'<p class="meta">{esc(row["id"])} · {esc(" · ".join(details))}</p>'
        f'<audio controls preload="none" src="{esc(row["native_path"])}"></audio>'
        f'<p class="meta">{esc(timings)}</p>'
        f'<details><summary>Texto e diagnóstico auxiliar</summary><p><b>Exibido:</b> {esc(row["display_text"])}</p>'
        f'<p><b>Texto da requisição:</b> {esc(row.get("request_text", row["tts_text"]))}</p>'
        f'<p><b>Representação de fala normalizada:</b> {esc(row["tts_text"])}</p>'
        f'<p><b>Texto efetivo do frontend de grafemas:</b> {esc(row.get("effective_text", row["tts_text"]))}</p>'
        f'<p><b>ASR auxiliar:</b> {esc(row.get("hypothesis", "pendente"))}</p>'
        f'<p><b>Referência textual do ASR:</b> {esc(row.get("asr_expected", "—"))}</p>'
        f'<p>{esc(row.get("asr_comparability", "Comparação auxiliar contra a representação de fala desta condição."))}</p>'
        f'<p>WER textual auxiliar: {number(row.get("word_error_rate"))}; não é aprovação de pronúncia ou identidade.</p>'
        '</details></article>')


def render(metrics, references, report):
    parts = ['<!doctype html><html lang="pt-BR"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>AegisVoice — refinamento de inferência</title>',
        '<style>body{font:16px system-ui,sans-serif;max-width:1120px;margin:auto;padding:24px;line-height:1.5;background:#f4f6f8;color:#17202a}'
        'h1{font-size:1.8rem}h2{margin-top:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}'
        'article{padding:14px;background:white;border:1px solid #ccd2da;border-radius:8px;overflow-wrap:anywhere}'
        'article h3{font-size:1rem;margin-top:0}audio{width:100%}.meta{font-size:.8rem;color:#47515d}summary{cursor:pointer}'
        'details{margin:12px 0}a{color:#17489d}code{overflow-wrap:anywhere}td,th{padding:6px;border:1px solid #cbd2da}table{border-collapse:collapse}</style>',
        '<h1>AegisVoice v2 — refinamento sem alterar pesos</h1>',
        '<p>Compare a mesma frase entre condições. A seleção técnica é provisória: identidade robótica, pronúncia, artefatos e continuidade aguardam sua escuta. '
        'NFE 8 não é declarado superior por soar menos humano. Não há áudio cortado por duração nem efeitos aplicados aos resultados.</p>',
        '<p><a href="metrics.csv" download>Medições CSV</a> · <a href="metrics.json">Medições JSON</a> · '
        '<a href="scorecard.csv" download>Ficha de escuta vazia</a> · <a href="report.json">Relatório técnico</a></p>',
        '<p>Ficha: inteligibilidade, identidade robótica, pronúncia e continuidade, de 1 (ruim) a 5 (adequada). '
        'Artefatos, de 0 (ausentes) a 4 (graves). Valores não foram preenchidos automaticamente. '
        'O texto exibido permanece separado da representação de fala; o Qwen usa seu frontend/DSP de produção.</p>']
    names = {'ordinary': 'Frase curta inédita', 'lexical': 'Nome e siglas', 'long': 'Texto longo e continuidade'}
    order = lambda r: (0 if r['id'].startswith('qwen-') else 1 if r['id'].endswith('-16') else 2 if r['id'].endswith('-8') else 3)
    for case in CASES:
        rows = sorted([r for r in metrics if r['stage'] == 'controlled' and r['case'] == case], key=order)
        parts += [f'<h2>{names[case]}</h2><p>{esc(rows[0]["display_text"])}</p><div class="grid">',
                  *[clip(r) for r in rows], '</div>']
    parts += ['<details><summary>Referências tratadas e matriz NFE 8/16</summary>',
        '<p>Quatro referências de grupos distintos do treino, selecionadas por diversidade acústica antes das sínteses. '
        'Pitch e textura são descritores, não medidas perceptivas de robotização. As transcrições seguem os arquivos originais; '
        'ASR das fontes sinalizou divergências em 0586 e 0042 que merecem conferência auditiva.</p>',
        '<div class="grid">', *[clip(r) for r in references], '</div>']
    refrows = [r for r in metrics if r['stage'] == 'references']
    for case in dict.fromkeys(r['case'] for r in refrows):
        rows = [r for r in refrows if r['case'] == case]
        parts += [f'<h3>{esc(case)}</h3><p>{esc(rows[0]["display_text"])}</p><div class="grid">',
                  *[clip(r) for r in rows], '</div>']
    parts += ['</details>', '<details><summary>Variantes de pronúncia</summary>',
        '<p>Escute alternativas do nome, das siglas e dos números. ASR não distingue de forma confiável a pronúncia desejada da grafia reconhecida.</p>']
    pron = [r for r in metrics if r['stage'] == 'pronunciation']
    for text in dict.fromkeys(r['display_text'] for r in pron):
        parts += [f'<h3>{esc(text)}</h3><div class="grid">',
                  *[clip(r) for r in pron if r['display_text'] == text], '</div>']
    parts += ['</details>', '<details><summary>Início do áudio: alinhamento e contexto do Vocos</summary>',
        '<p>Mesmo mel gerado, três decodificações. As latências deste diagnóstico compartilham geração e não formam benchmark de velocidade. '
        'A remoção corresponde apenas ao prefixo de condicionamento conhecido na grade mel; nenhuma consoante gerada ou cauda é detectada/cortada.</p><div class="grid">',
        *[clip(r) for r in metrics if r['stage'] == 'alignment'], '</div></details>']
    diagnostics = [r for r in metrics if r['stage'] == 'duration']
    if diagnostics:
        parts += ['<details><summary>Diagnóstico de duração do texto longo</summary>',
            '<p>O texto completo é regenerado com outro orçamento neural de frames. '
            'O áudio não é encurtado depois da geração. Este diagnóstico não substitui a síntese por sentenças.</p><div class="grid">',
            *[clip(r) for r in diagnostics], '</div></details>']
    parts += [
        '<h2>Como interpretar as medições</h2><p>Primeiro PCM útil exige atividade em duas janelas consecutivas de 10 ms '
        '(pico &gt;100 e RMS &gt;30, PCM16). Disponibilidade do buffer, offset acústico e início estimado da reprodução FIFO são campos separados. '
        'RTF é tempo de produção dividido pela duração: Qwen inclui HTTP/DSP, F5 mede inferência local; carga única e ASR ficam fora. '
        'Uma observação por condição não demonstra estabilidade estatística nem garante a mesma latência em produção.</p>',
        f'<p>Seleção técnica: <code>{esc(report["selected_configuration"])}</code>. Sem promoção à produção ou novo treinamento.</p>',
        '</html>']
    return '\n'.join(parts)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('src', 'href') and value:
                self.links.append(value)


def verify_directory(directory):
    parser = Links()
    parser.feed((directory / 'index.html').read_text())
    for link in parser.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            raise ValueError('offline package must not have external links')
        if parsed.path and not compat.local_path(directory, unquote(parsed.path)).is_file():
            raise ValueError('broken local package link: ' + link)
    manifest = load(directory, 'manifest-sha256.json', {})
    for relative, expected in manifest.items():
        if digest(compat.local_path(directory, relative)) != expected:
            raise ValueError('package audio hash mismatch: ' + relative)
    return dict(local_links=len(parser.links), wav_hashes=len(manifest), all_local_links_exist=True,
                native_and_playback_hashes_verified=True)


def collect(root):
    complete = load(root, 'gpu-complete.json', {})
    if (complete.get('all_gpu_experiments_complete') is not True
            or complete.get('no_weight_updates') is not True
            or complete.get('additional_training_started') is not False
            or complete.get('owner_approval') is not False):
        raise ValueError('inference-only completion, unchanged weights and pending owner approval receipts required')
    operation = operation_windows(root)
    restored, baseline = operation['restored'], operation['baseline']
    if (not operation['production_restoration_verified'] or not restored
            or restored.get('running') is not True or restored.get('health') != 'healthy'):
        raise ValueError('verified production restoration required before final package')
    if not baseline or any(not baseline.get(k) or restored.get(k) != baseline[k] for k in ('container_id', 'image_id')):
        raise ValueError('restored production container/image differs from baseline')
    preflight = load(root, 'preflight.json', {})
    audit = load(root, 'technical-audit.json', {})
    seal = audit.get('seal', {})
    expected = preflight.get('checkpoint_sha256')
    checkpoint_name = Path(preflight.get('checkpoint', '')).name
    if (not expected or not preflight.get('checkpoint_hashes')
            or complete.get('checkpoint_sha256_after') != expected
            or seal.get('selected_checkpoint_sha256') != expected
            or seal.get('all_checkpoint_hashes') != preflight['checkpoint_hashes']
            or preflight['checkpoint_hashes'].get(checkpoint_name) != expected
            or seal.get('no_weight_updates') is not True
            or seal.get('additional_training_started') is not False
            or seal.get('production_restored') is not True
            or seal.get('container_id') != restored['container_id']
            or seal.get('image_id') != restored['image_id']
            or audit.get('owner_perceptual_approval') is not False):
        raise ValueError('valid inference-only technical audit and unchanged checkpoint seal required')
    required_inputs = ('preflight.json', 'gpu-complete.json', 'results.json', 'asr.json',
        'qwen/results.json', 'operations/gpu-window.json', 'operations/restored.json', 'operations/baseline.json')
    hashes = audit.get('input_receipt_sha256', {})
    if any(hashes.get(name) != digest(root / name) for name in required_inputs):
        raise ValueError('technical audit input receipts changed or are not sealed')
    rows = choose_rows(load(root, 'results.json', []), load(root, 'qwen/results.json', []))
    asrs = load(root, 'asr.json', [])
    frontend_audit = load(root, 'qwen/frontend-audit.json', {})
    if frontend_audit and frontend_audit.get('qwen_results_sha256') != digest(root / 'qwen/results.json'):
        raise ValueError('Qwen frontend audit refers to different original results')
    frontends = {r['id']: r for r in frontend_audit.get('records', [])}
    metrics = [metric_row(r, dict(expected=r['tts_text'], **asr_for(r, asrs)), frontends.get(r['id'])) for r in rows]
    comparison = load(root, 'reference-comparison.json', {})
    reference_document = load(root, 'references.json', {})
    source_references = (reference_document.get('references', [])
                         if isinstance(reference_document, dict) else reference_document)
    controlled = [r for r in metrics if r['stage'] == 'controlled']
    aggregates = []
    for condition in ('qwen', 'original16', 'original8', 'selected'):
        group = [r for r in controlled if (r['id'].startswith('qwen-') if condition == 'qwen'
            else r['id'].startswith('selected-') if condition == 'selected'
            else r['id'].startswith('original-') and r['nfe'] == int(condition.removeprefix('original')))]
        aggregates.append(dict(condition=condition, cases=len(group), **{
            k + '_median': statistics.median([r[k] for r in group if r.get(k) is not None])
            if any(r.get(k) is not None for r in group) else None
            for k in ('rtf', 'first_useful_pcm_available_ms', 'estimated_first_audible_playback_ms')}))
    report = dict(schema='aegis.f5.refinement.report.v1', utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=complete.get('checkpoint_sha256_after'), no_weight_updates=True,
        technical_seal=seal,
        production_restoration_verified=True, production_container_id=operation['restored']['container_id'],
        downtime_seconds=operation['downtime_seconds'], additional_training_started=False,
        selected_configuration=comparison.get('selected', {'reference_id': complete.get('selected_reference'), 'nfe': complete.get('selected_nfe')}),
        reference_scores=comparison.get('scores'), reference_selection_policy=comparison.get('selection_policy'),
        qwen_frontend_audit=frontend_audit or None,
        reference_sources=[{k: ref.get(k) for k in (
            'id', 'group_id', 'duration', 'sha256', 'source_audio_sha256',
            'reference_role', 'selection_metrics')} for ref in source_references],
        controlled_aggregates=aggregates, listening_files=len(rows) + 4,
        perceptual_status='pending_owner_listening', nfe8_superiority_established=False,
        stage2_training_decision='deferred until owner evaluates consistent robotic identity and acoustic degradation; no new training authorized by this report',
        limitations=[
            'Auxiliary ASR/WER can disagree on names, acronyms, numbers and spelling; not perceptual intelligibility or pronunciation scores.',
            'Pitch/flatness/spectral texture distances cannot establish robotic identity or distinguish desired synthesis from acoustic degradation.',
            'Small fixed cohort, one seed and one timed observation per controlled condition; no claim of statistical superiority.',
            'Reference comparison controls generated frame budgets; conditioned reference length and acoustic content differ.',
            'Qwen includes the current HTTP frontend/DSP; F5 is local inference; no F5 HTTP deployment or within-sentence neural streaming demonstrated.',
            'Qwen lexical ASR originally used the raw request as expected text; the production frontend expands spelling and maps Aegis to Égide. Original WER retained and not directly comparable to F5 spoken variants.',
            'Alignment comparisons share generated mel; cumulative total_seconds is diagnostic and excluded from controlled aggregates.',
            'Segment boundaries retain full generated audio and punctuation; waveform jumps are not perceptual continuity approval.',
            'Source transcript file integrity verified; auxiliary source ASR flags0586 and0042 for listening; no source text rewritten.',
        ], preserved_source_artifacts={name: digest(root / name) for name in (
            'preflight.json', 'gpu-complete.json', 'results.json', 'asr.json', 'reference-comparison.json',
            'alignment.json', 'references.json', 'reference-candidates-audit.json', 'reference-source-asr.json',
            'cpu-tail-audit.json', 'qwen/results.json', *OPTIONAL_EVIDENCE) if (root / name).is_file()})
    return rows, metrics, report


def build(root, archive):
    root = private_path(root).resolve(strict=True)
    archive = private_path(archive)
    rows, metrics, report = collect(root)
    directory = root / 'listening'
    if directory.is_symlink():
        raise ValueError('listening directory must not be a symlink')
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    manifest = {}
    copied = {}
    def copy_native(source, expected, relative):
        source = Path(source)
        if digest(source) != expected:
            raise ValueError('source audio changed: ' + str(source))
        # Share physically identical audio without creating redundant files.
        if expected in copied:
            return copied[expected]
        target = compat.local_path(directory, relative)
        target.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
        if target.exists():
            if digest(target) != expected:
                raise ValueError('existing native output differs; refusing overwrite')
        else:
            shutil.copyfile(source, target)
            target.chmod(0o600)
        manifest[relative] = expected
        copied[expected] = relative
        return relative
    for row, metric in zip(rows, metrics):
        metric['native_path'] = copy_native(row['path'], row['sha256'], 'audio/' + row['id'] + '.wav')
    refjson = load(root, 'references.json')
    references = refjson['references'] if isinstance(refjson, dict) else refjson
    if len(references) != 4:
        raise ValueError('expected four selected source references')
    refmetrics = []
    for ref in references:
        relative = copy_native(ref['audio_filepath'], ref['sha256'], 'references/' + ref['id'] + '.wav')
        refmetrics.append(dict(id='source-' + ref['id'], label=ref['id'], display_text=ref.get('original_text', ref['text']),
            tts_text=ref['text'], native_path=relative, audio_seconds=ref['duration'],
            reference_id=ref['id'], source_group=ref['group_id']))
    write(directory / 'manifest-sha256.json', manifest)
    write(directory / 'metrics.json', metrics)
    write(directory / 'report.json', report)
    fields = ['id', 'stage', 'label', 'case', 'reference_id', 'nfe', 'seed', 'blocks', *MEASURES,
              'display_text', 'request_text', 'tts_text', 'effective_text', 'hypothesis', 'asr_expected', 'asr_comparability',
              'sha256', 'native_path', 'streaming_scope', 'perceptual_status']
    table_csv(directory / 'metrics.csv', metrics, fields)
    scorecard = [dict(id=r['id'], stage=r['stage'], case=r['case'], label=r['label'],
        status='pending_owner_listening', **{field: '' for field in PERCEPTUAL_FIELDS}) for r in metrics]
    # Rebuilding evidence must preserve any owner scores already entered locally.
    if (directory / 'scorecard.csv').is_file():
        with (directory / 'scorecard.csv').open(newline='', encoding='utf-8-sig') as inp:
            previous = {row['id']: row for row in csv.DictReader(inp)}
        for row in scorecard:
            old = previous.get(row['id'], {})
            for key in ('status', *PERCEPTUAL_FIELDS):
                if old.get(key):
                    row[key] = old[key]
    table_csv(directory / 'scorecard.csv', scorecard, ['id', 'stage', 'case', 'label', 'status', *PERCEPTUAL_FIELDS])
    (directory / 'index.html').write_text(render(metrics, refmetrics, report))
    (directory / 'index.html').chmod(0o600)
    browser = compat.process(directory)
    # Export only currently described files, even if a prior build directory had
    # additional files. A smaller new selection never silently keeps stale WAVs.
    described = set(load(directory, 'manifest-sha256.json')) | {
        'manifest-sha256.json', 'metrics.json', 'metrics.csv', 'scorecard.csv', 'report.json',
        'index.html', compat.RECEIPT}
    for relative in OPTIONAL_EVIDENCE:
        if (root / relative).is_file():
            target = compat.local_path(directory, relative)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(root / relative, target)
            target.chmod(0o600)
            described.add(relative)
    verification = verify_directory(directory)
    artifacts = {relative: digest(compat.local_path(directory, relative)) for relative in sorted(described)}
    write(directory / 'artifacts-sha256.json', artifacts)
    described.add('artifacts-sha256.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + '.new')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for relative in sorted(described):
            out.write(compat.local_path(directory, relative), relative)
    temporary.chmod(0o600)
    with zipfile.ZipFile(temporary) as check:
        if check.testzip() is not None:
            raise ValueError('archive CRC failed')
        if set(check.namelist()) != described:
            raise ValueError('archive has missing or stale entries')
        for relative, expected in artifacts.items():
            if hashlib.sha256(check.read(relative)).hexdigest() != expected:
                raise ValueError('archive entry SHA256 mismatch: ' + relative)
    temporary.replace(archive)
    verification.update(archive=str(archive), archive_sha256=digest(archive),
        archive_bytes=archive.stat().st_size, archive_crc_verified=True,
        archive_entry_hashes_verified=True, archive_entries=len(described),
        native_files=browser['native_files'], playback_copies=browser['playback_copies'],
        declined_pcm16=browser['declined'], listening_rows=len(metrics), source_references=len(refmetrics),
        block_and_repeat_duplicates_exported=False)
    write(root / 'package-verification.json', verification)
    write(root / 'report.json', report)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(build(args.root, args.archive), ensure_ascii=False))


if __name__ == '__main__':
    main()
