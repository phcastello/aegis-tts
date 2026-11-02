#!/usr/bin/env python3
"""Package the F5/Vocos/NanoCodec investigation for private offline listening.

Only explicit WAV/JSON inputs are read. No model, GPU, network or publication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import itertools
import json
from pathlib import Path
import random
import re
import shutil
import wave
import zipfile

import benchmark

HIDDEN = "NAO_ABRIR_ANTES_DA_ESCUTA"
CRITERIA = {
    "inteligibilidade": "Inteligibilidade",
    "portugues_brasileiro": "Pronúncia brasileira",
    "naturalidade": "Naturalidade",
    "identidade": "Identidade vocal desejada",
    "textura_robotica": "Textura robótica desejada",
    "ataques_sibilantes": "Ataques e sibilantes",
    "ausencia_artefatos": "Ausência de artefatos",
    "expressividade_excessiva": "Expressividade excessiva (1=baixa; 5=alta)",
}
METRIC_KEYS = (
    "case", "repetition", "seed", "nfe", "source_text", "effective_text", "text_mode", "oov_tokens",
    "ode_method", "use_epss", "cfg_strength", "sway_sampling_coef", "transformer_calls", "cfg_batch_size",
    "output_blocks", "sentence_chunks", "normalization_tokenization_seconds", "diffusion_seconds",
    "vocoder_seconds", "complete_seconds", "rtf", "reference_cache_hit", "warmup", "cold_generation",
    "sha256", "sample_rate", "subtype", "seconds", "peak", "samples_outside_unit_range", "first_pcm_ms",
    "first_useful_pcm_available_ms", "onset_audio_offset_ms", "estimated_first_audible_playback_ms",
    "audio_seconds", "peak_pcm16", "clipped_samples", "immediate_playback_underruns_over_10ms",
    "immediate_playback_underrun_ms", "onset_proxy", "playback_assumption",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    benchmark.write_private(path, benchmark.json_bytes(data))


def portable(value):
    """Reject absolute host paths in nested portable receipts."""
    if isinstance(value, str) and (value.startswith(("/", "file:", "\\\\")) or
                                  any(prefix in value for prefix in ("/home/", "/var/tmp/", "/tmp/")) or
                                  re.match(r"^[A-Za-z]:[\\/]", value)):
        raise ValueError("absolute paths cannot enter a portable receipt")
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value


def listening_copy(source, target):
    """Keep PCM16 bit-for-bit; round floats, or retain floats on overload."""
    try:
        with wave.open(str(source), "rb") as stream:
            pcm16 = stream.getnchannels() == 1 and stream.getsampwidth() == 2
        if pcm16:
            shutil.copyfile(source, target)
            target.chmod(0o600)
            return {"conversion": "original PCM16 copied bit-for-bit", "pcm16": True, "maximum_error_lsb": 0}
    except wave.Error:
        pass
    import numpy as np
    import soundfile as sf
    samples, rate = sf.read(source, dtype="float64", always_2d=True)
    if samples.shape[1] != 1 or not np.isfinite(samples).all():
        raise ValueError("refuse nonmono/nonfinite audio")
    samples = samples[:, 0]
    quantized = np.rint(samples * 32768)
    if np.max(np.abs(samples)) >= 1 or quantized.min() < -32768 or quantized.max() > 32767:
        shutil.copyfile(source, target)
        target.chmod(0o600)
        return {"conversion": "PCM16 blocked by overload; original FLOAT retained unchanged",
                "pcm16": False, "peak": float(np.max(np.abs(samples))),
                "samples_at_or_above_fullscale": int(np.count_nonzero(np.abs(samples) >= 1)),
                "browser_note": "FLOAT WAV playback depends on browser; device output may saturate; use originals for numeric analysis"}
    encoded = quantized.astype("<i2")
    error = float(np.max(np.abs(encoded.astype(float) / 32768 - samples)) * 32768)
    if error > 1:
        raise ValueError("PCM16 conversion exceeded one LSB")
    with wave.open(str(target), "wb") as stream:
        stream.setparams((1, 2, rate, len(encoded), "NONE", "not compressed"))
        stream.writeframes(encoded.tobytes())
    target.chmod(0o600)
    return {"conversion": "nearest PCM16; no gain, trim, resample, dither or limiting", "pcm16": True,
            "maximum_error_lsb": error, "sample_rate": rate, "frames": len(samples)}


def listening_html(review):
    cards = []
    for row in review:
        columns = []
        for label in ("A", "B"):
            fields = "".join(f'<label>{html.escape(title)} <select data-pair="{row["pair"]}" data-label="{label}" data-criterion="{criterion}"><option value="">—</option>'
                             + "".join(f"<option>{i}</option>" for i in range(1, 6)) + "</select></label>"
                             for criterion, title in CRITERIA.items())
            columns.append(f'<section><h3>{label}</h3><audio controls preload="none" src="{row[label]}"></audio>{fields}</section>')
        cards.append(f'<article><h2>Par {row["pair"]:03d}</h2><p>{html.escape(row["text"])}</p><div class="columns">'
                     + "".join(columns) + f'</div><label>Preferência <select data-pair="{row["pair"]}" data-label="par" data-criterion="preferencia"><option value="">—</option><option>A</option><option>B</option><option>Empate</option><option>Ambos inadequados</option></select></label>'
                     + f'<label>Notas: omissões, repetições, pronúncia, cortes e timbre <textarea data-pair="{row["pair"]}" data-label="par" data-criterion="notas"></textarea></label></article>')
    data = json.dumps([{k: row[k] for k in ("pair", "case", "text", "A", "B")} for row in review], ensure_ascii=False).replace("<", "\\u003c")
    return '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AegisVoice — escuta privada</title>
<style>body{font:17px system-ui;max-width:1050px;margin:2em auto;padding:1em;line-height:1.5}article{border:1px solid #bbb;border-radius:8px;padding:1em;margin:1em 0}.columns{display:flex;gap:2em;flex-wrap:wrap}section{flex:1;min-width:280px}label{display:block;margin:.5em 0}select{font-size:1em}textarea{display:block;width:95%;min-height:70px}button{font:inherit;margin:.4em;padding:.5em}audio{max-width:100%}.tools{position:sticky;top:0;background:white;border-bottom:1px solid #bbb;padding:.5em}</style>
<h1>Escuta privada AegisVoice</h1><p>Extraia o ZIP antes de abrir. Compare em volume fixo, sem filtros automáticos. Nas escalas, 5 é melhor, exceto expressividade excessiva: 1 é baixa e 5 é alta. Deixe em branco critérios sem evidência. Em reconstruções, identidade mede retenção da faixa original.</p>
<p>Ouça todos os pares antes de abrir a pasta de respostas. Salve suas notas pelos botões abaixo; o navegador pode não preservar dados ao fechar o arquivo. Existe também uma ficha CSV vazia. <a href="../coorte_qwen.html">Coorte identificada</a> (abra depois da rodada cega).</p>
<div class="tools"><button onclick="downloadJSON()">Baixar minhas notas JSON</button><button onclick="downloadCSV()">Baixar minhas notas CSV</button><span id="status">Nenhuma nota gravada em arquivo ainda.</span></div>
''' + "".join(cards) + '''<script>
const pairs=''' + data + ''';
function rows(){return [...document.querySelectorAll('[data-pair]')].map(e=>({par:Number(e.dataset.pair),amostra:e.dataset.label,criterio:e.dataset.criterion,nota:e.value}));}
function save(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.getElementById('status').textContent='Download solicitado. Confira sua pasta de downloads.';}
function downloadJSON(){save('minhas-notas-aegisvoice.json',JSON.stringify({avaliador:'',data:new Date().toISOString(),pares:pairs,notas:rows()},null,2),'application/json');}
function downloadCSV(){const cells=v=>'"'+String(v??'').replaceAll('"','""')+'"';save('minhas-notas-aegisvoice.csv','\\ufeff'+[['par','amostra','criterio','nota'],...rows().map(r=>[r.par,r.amostra,r.criterio,r.nota])].map(r=>r.map(cells).join(';')).join('\\r\\n'),'text/csv;charset=utf-8');}
</script></html>'''


def package(args):
    root, out, archive = args.experiment_root.resolve(), args.output_dir.resolve(), args.zip.resolve()
    if out.exists() or archive.exists():
        raise ValueError("select NEW output directory and ZIP; existing files are preserved")
    for target in (out, archive.parent):
        if any((p / ".git").exists() for p in (target, *target.parents)):
            raise ValueError("private package must be outside any Git checkout")
    corpus = read_json(Path(__file__).with_name("corpus.pt-BR.json"))
    texts = {row["id"]: row["text"] for row in corpus}
    hidden = out / HIDDEN
    for directory in (out, hidden, hidden / "originais", hidden / "amostras", out / "escuta", out / "coorte_qwen"):
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    samples, metrics, aliases = [], {}, {}

    def sample(source, identity, text, receipt=None):
        if not source.is_file() or source.is_symlink():
            raise ValueError("explicit audio input missing or symlinked")
        identifier = f"arquivo-{len(samples) + 1:03d}"
        original = hidden / "originais" / (identifier + ".wav")
        target = hidden / "amostras" / (identifier + ".wav")
        shutil.copyfile(source, original)
        original.chmod(0o600)
        conversion = listening_copy(source, target)
        row = {"sample_id": identifier, "identity": identity, "declared_text": text,
               "original": original.relative_to(out).as_posix(), "original_sha256": digest(original),
               "listening_audio": target.relative_to(out).as_posix(), "listening_sha256": digest(target), **conversion}
        samples.append(row)
        if receipt is not None:
            metrics[identifier] = portable(receipt)
        aliases[str(target)] = identifier
        return identifier, target, conversion["pcm16"]

    treated_text = "Seis sensores silenciosos estão instalados no sistema de segurança."
    groups = []
    for model, folder, filenames in [
        ("F5 Vocos numeric roundtrip", root / "f5/roundtrip-treated-02", ("A-original.wav", "B-input-24000.wav", "C-vocos-24000.wav")),
        ("Magpie NanoCodec numeric roundtrip; no Magpie speech synthesis", root / "magpie/roundtrip-treated-v1", ("A-original.wav", "B-input-22050-float.wav", "C-roundtrip-22050-float.wav")),
    ]:
        roles = {}
        for role, filename in zip(("original", "input", "reconstruction"), filenames):
            roles[role] = sample(folder / filename, {"model": model, "role": role}, treated_text)
        groups.append(roles)

    qwen = {}
    qwen_root = root / "qwen-fixed-cohort"
    if (qwen_root / "manifest.json").is_file():
        observations = read_json(qwen_root / "manifest.json")["observations"]
        by_id = {row["case"]: row for row in observations}
        if len(observations) != 9 or set(by_id) != set(texts):
            raise ValueError("Qwen manifest must contain the complete fixed nine-case cohort")
        for case, text in texts.items():
            receipt = by_id[case]
            source = qwen_root / (case + ".wav")
            if receipt["text"] != text or receipt["sha256"] != digest(source):
                raise ValueError("Qwen text/hash does not match the fixed cohort")
            safe = {k: receipt[k] for k in ("case", "text", "sha256", "sample_rate", "channels", "samples")}
            qwen[case] = sample(source, {"model": "Qwen 0.6B production W04/P07/M02", "case": case}, text, safe)
            shutil.copyfile(source, out / "coorte_qwen" / (case + ".wav"))
            (out / "coorte_qwen" / (case + ".wav")).chmod(0o600)

    f5 = []
    directory = root / "f5/infer-novel-short-01"
    for receipt_path in sorted(directory.glob("*.json")):
        match = re.fullmatch(r"(novel_short)-nfe(8|16|32)-r(\d+)\.json", receipt_path.name)
        if not match:
            continue
        row = read_json(receipt_path)
        case, nfe, repetition = match[1], int(match[2]), int(match[3])
        source = directory / (receipt_path.stem + ".wav")
        if (row.get("source_text") != texts[case] or row.get("nfe") != nfe or
                row.get("repetition") != repetition or row.get("sha256") != digest(source)):
            raise ValueError("F5 receipt/text/hash mismatch; do not package an unfinished output")
        ref = sample(source, {"model": "F5-TTS PT-BR", "case": case, "nfe": nfe, "repetition": repetition}, texts[case],
                     {k: row[k] for k in METRIC_KEYS if k in row})
        f5.append((case, nfe, repetition, ref))
    if not set(args.require_nfe or []) <= {row[1] for row in f5}:
        raise ValueError("required F5 NFE outputs have not all completed")

    pairs = []
    def pair(left, right, text, purpose):
        pairs.append({"a": str(left[1]), "b": str(right[1]), "text": text,
                      "category": "listening_control", "same_generation": purpose == "acoustic roundtrip",
                      "_identities": {"source_a": left[0], "source_b": right[0], "purpose": purpose},
                      "_pcm16": left[2] and right[2]})
    for roles in groups:
        for a, b in (("original", "input"), ("input", "reconstruction"), ("original", "reconstruction")):
            pair(roles[a], roles[b], treated_text, "acoustic roundtrip")
    for (case_a, nfe_a, rep_a, ref_a), (case_b, nfe_b, rep_b, ref_b) in itertools.combinations(f5, 2):
        if case_a == case_b and rep_a == rep_b and nfe_a != nfe_b:
            pair(ref_a, ref_b, texts[case_a], "F5 step comparison")
    for case, nfe, repetition, ref in f5:
        if case in qwen:
            pair(qwen[case], ref, texts[case], "same-text synthesis; production DSP versus native F5")
    random.Random(args.blind_seed + 1).shuffle(pairs)
    # Reuse the existing PCM16 builder verbatim. If FLOAT overload prevents
    # quantization, append those pairs with original WAVs instead of discarding
    # attempts or normalizing them to make them fit the builder.
    safe_pairs = [p for p in pairs if p["_pcm16"]]
    float_pairs = [p for p in pairs if not p["_pcm16"]]
    pairs = safe_pairs + float_pairs
    for number, row in enumerate(pairs, 1):
        row["case"] = f"controle_{number:03d}"
    review = benchmark.build_ab({"pairs": safe_pairs}, out / "escuta", args.blind_seed)
    key_path = out / "escuta/answer-key.private.json"
    key = read_json(key_path)
    for index, row in enumerate(key["pairs"]):
        row.update(safe_pairs[index]["_identities"])
        for source in row["sources"]:
            source["sample_id"] = aliases[source["source"]]
            del source["source"]
    rng = random.Random(args.blind_seed + 2)
    for pair_data in float_pairs:
        number = len(review) + 1
        sources = [("a", Path(pair_data["a"])), ("b", Path(pair_data["b"]))]
        rng.shuffle(sources)
        answer, audio_names = [], {}
        for label, (source_label, source) in zip(("A", "B"), sources):
            name = f"pair-{number:03d}-{label}.wav"
            benchmark.write_private(out / "escuta" / name, source.read_bytes())
            audio_names[label] = name
            answer.append({"label": label, "source_label": source_label, "sample_id": aliases[str(source)],
                           "sha256": digest(source), "format": "preserved FLOAT or original/quantized PCM16; see sample receipt"})
        review.append({"pair": number, "case": pair_data["case"], "text": pair_data["text"], **audio_names})
        key["pairs"].append({"pair": number, **pair_data["_identities"], "sources": answer})
    key_path.write_bytes(benchmark.json_bytes(portable(key)))
    key_path.rename(hidden / "respostas.json")
    for row in review:
        row["scores"] = {label: {criterion: None for criterion in CRITERIA} for label in ("A", "B")}
        row["preference"], row["notes"] = None, ""
    (out / "escuta/review.json").write_bytes(benchmark.json_bytes(review))
    (out / "escuta/index.html").write_text(listening_html(review), encoding="utf-8")
    sheet = io.StringIO()
    writer = csv.writer(sheet, delimiter=";")
    writer.writerow(["par", "amostra", "criterio", "nota"])
    for row in review:
        for label in ("A", "B"):
            for criterion in CRITERIA:
                writer.writerow([row["pair"], label, criterion, ""])
        writer.writerow([row["pair"], "par", "preferencia", ""])
        writer.writerow([row["pair"], "par", "notas", ""])
    benchmark.write_private(out / "ficha-vazia.csv", ("\ufeff" + sheet.getvalue()).encode())
    write_json(hidden / "amostras-e-conversoes.json", portable(samples))
    write_json(hidden / "metricas-sintese.json", portable(metrics))
    write_json(out / "corpus.json", corpus)
    qwen_cards = [f'<article><h2>{html.escape(case)}</h2><p>{html.escape(texts[case])}</p><audio controls preload="none" src="coorte_qwen/{case}.wav"></audio></article>' for case in qwen]
    benchmark.write_private(out / "coorte_qwen.html", ('<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Coorte Qwen</title><h1>Coorte Qwen 0.6B atual — nove textos</h1><p>Áudios para escuta. Tempos não são benchmark exclusivo contra a candidata.</p>' + "".join(qwen_cards)).encode())
    benchmark.write_private(out / "index.html", b'<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=escuta/index.html"><a href="escuta/index.html">Abrir escuta privada</a></html>')
    readme = f"""ESCUTA PRIVADA AEGISVOICE — pacote offline

1. Extraia TODO o ZIP e abra index.html no navegador. Não requer internet.
2. Ouça todos os pares em ordem e volume fixo, sem filtros automáticos.
3. Preencha as fichas no HTML e clique Baixar minhas notas JSON/CSV. Confira
   a pasta de downloads. Alternativa: abra ficha-vazia.csv numa planilha.
4. Abra a coorte identificada somente depois da rodada cega, se desejar.
5. Guarde suas notas antes de abrir {HIDDEN}/respostas.json.

São {len(review)} pares cegos e {len(qwen)} áudios da coorte identificada.
Todas as saídas concluídas encontradas foram incluídas, sem escolher a melhor
seed. Os controles acústicos comparam original, entrada e reconstrução.
Neles, identidade significa retenção da faixa de origem; não se pede avaliar
uma identidade vocal nova aprendida por treinamento.

As taxas e os níveis foram preservados. Conversões FLOAT→PCM16 são explícitas,
sem ganho, corte, limiter ou denoiser; erro limitado a um LSB. Se houver overload,
o FLOAT original fica preservado e a limitação de reprodução é registrada;
não se corta o pico para favorecer a amostra. O suporte FLOAT depende do
navegador. Os originais numéricos, hashes, métricas e identidades ficam na
pasta de respostas. Diferenças nativas de duração/padding foram preservadas.

Uma candidata foi gerada em CPU e a coorte operacional veio de serviço
aquecido com DSP. Este pacote permite ESCUTA. Seus tempos não provam qual
arquitetura é mais rápida na A2000 e não são um benchmark exclusivo de GPU.

Não há pesos, ambientes, credenciais, caminhos absolutos do host ou corpus de
treinamento neste arquivo. Mantenha o pacote e suas notas privados.
"""
    benchmark.write_private(out / "LEIA-ME.txt", readme.encode())
    codec_receipts = {}
    for label, path in [("vocos", root / "f5/roundtrip-treated-02/results.json"),
                        ("nanocodec", root / "magpie/roundtrip-treated-v1/receipt.json")]:
        data = read_json(path)
        codec_receipts[label] = portable({k: v for k, v in data.items() if k not in {"observations", "failures"}})
    write_json(hidden / "recibos-acusticos.json", codec_receipts)
    summary_path = directory / "results.json"
    if summary_path.is_file():
        data = read_json(summary_path)
        # Scalar/nested whitelists only: no raw runtime config, paths or logs.
        safe = {k: data[k] for k in ("candidate", "action", "device", "rate", "streaming",
                                    "streaming_evidence", "normalization", "versions", "success",
                                    "provenance", "failures") if k in data}
        if "failures" in safe:
            safe["failures"] = [{"type": row.get("type", "unknown"),
                                 "message": re.sub(r"(?:/[A-Za-z0-9_.-]+){2,}", "<caminho_privado>", row.get("message", ""))}
                                for row in safe["failures"]]
        write_json(hidden / "recibo-geracao.json", portable(safe))
    asr_path = root / "asr/limited-transcripts.json"
    if asr_path.is_file():
        data = read_json(asr_path)
        hashes_to_ids = {row["original_sha256"]: row["sample_id"] for row in samples}
        selected = []
        for row in data.get("attempts", []):
            if row.get("audio_sha256") not in hashes_to_ids or row.get("expected_text") not in texts.values():
                continue  # Never import transcripts for unrelated private audio.
            safe = {k: row[k] for k in ("candidate", "case", "audio_sha256", "expected_text", "source_rate",
                                        "source_seconds", "source_peak", "asr_input_sha256", "recognized_text",
                                        "segments", "duration_seconds", "success", "wall_seconds") if k in row}
            safe["sample_id"] = hashes_to_ids[row["audio_sha256"]]
            selected.append(portable(safe))
        safe = {k: data[k] for k in ("settings", "versions", "model_download_receipt_sha256", "note",
                                    "last_load_seconds") if k in data}
        safe["attempts"] = selected
        write_json(hidden / "asr-limitado.json", portable(safe))
    manifest = {p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "manifesto-sha256.json", manifest)
    archive.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                p.chmod(0o600)
                zipped.write(p, arcname=p.relative_to(out).as_posix())
    archive.chmod(0o600)
    return {"zip": str(archive), "zip_sha256": digest(archive), "zip_bytes": archive.stat().st_size,
            "index": str(out / "index.html"), "blind_pairs": len(review), "qwen_cases": len(qwen),
            "f5_observations": [{"case": row[0], "nfe": row[1], "repetition": row[2]} for row in f5]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--require-nfe", type=int, nargs="+", choices=(8, 16, 32))
    parser.add_argument("--blind-seed", type=int, default=40733133)
    args = parser.parse_args(argv)
    print(json.dumps(package(args), indent=2))


if __name__ == "__main__":
    main()
