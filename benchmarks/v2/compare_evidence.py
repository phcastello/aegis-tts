#!/usr/bin/env python3
"""Compact reproducible comparison; reads saved evidence only, never audio/models."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from training.f5_asr_metrics import plain_word_tokens


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def median(rows,name):
    v=[r[name] for r in rows if r.get(name) is not None]
    return statistics.median(v) if v else None

def align(reference,hypothesis):
    """Token edit alignment: evidence for review, not an auditory judgment."""
    a=plain_word_tokens(reference);b=plain_word_tokens(hypothesis)
    d=[list(range(len(b)+1))]
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,d[-1][j]+1,d[-1][j-1]+(x!=y)))
        d.append(row)
    i=len(a);j=len(b);ops=[]
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:ops.append(dict(operation='substitution',expected=a[i-1],recognized=b[j-1],reference_index=i-1))
            i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:
            ops.append(dict(operation='ASR_deletion_candidate',expected=a[i-1],reference_index=i-1));i-=1
        else:
            ops.append(dict(operation='ASR_insertion_candidate',recognized=b[j-1],reference_index=i));j-=1
    ops.reverse()
    repeated=[]
    for n in (3,4,5):
        ref_ngrams=collections.Counter(tuple(a[k:k+n]) for k in range(len(a)-n+1))
        hyp_ngrams=collections.Counter(tuple(b[k:k+n]) for k in range(len(b)-n+1))
        repeated += [dict(tokens=list(k),count=v,reference_count=ref_ngrams[k]) for k,v in hyp_ngrams.items() if v>max(1,ref_ngrams[k])]
    return dict(edits=ops,repeated_ngrams_candidates=repeated,expected_last_word=a[-1] if a else None,
        recognized_last_word=b[-1] if b else None,last_word_equal=bool(a and b and a[-1]==b[-1]),
        caution='ASR alignment only. Number/acronym/name orthography can mimic omission/repetition; compare actual audio before attributing a TTS error.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-root',default='/var/tmp/aegis-v2-work/bench');p.add_argument('--asr',required=True);p.add_argument('--qwen-asr',required=True)
    p.add_argument('--private-report',required=True);p.add_argument('--markdown',required=True);a=p.parse_args()
    root=Path(a.bench_root);names=['qwen-warm','f5-nfe8','f5-nfe16','vits-base'];bench={}
    for name in names:bench[name]=[json.loads(x) for x in (root/name/'results.jsonl').read_text().splitlines()]
    receipts=read(a.qwen_asr)['attempts']+read(a.asr)['attempts'];signals=[];groups={}
    for row in receipts:
        if not row.get('success'):continue
        signals.append(dict(candidate=row['candidate'],case=row['case'],cohort=row['cohort'],artifact=row.get('artifact'),
            wav_sha256=row['audio_sha256'],comparisons={kind:align(row['expected_'+kind],row['hypothesis']) for kind in row['comparisons']}))
        for kind,score in row['comparisons'].items():
            key=(row['candidate'],row['cohort'],kind)
            s=groups.setdefault(key,dict(words=0,edits=0,characters=0,character_edits=0,cases=[]))
            s['words']+=len(score['reference_tokens']);s['edits']+=score['word_edits'];s['characters']+=score['reference_characters'];s['character_edits']+=score['character_edits'];s['cases'].append(row['case'])
    aggregate=[]
    for key,s in sorted(groups.items()):
        aggregate.append(dict(candidate=key[0],cohort=key[1],policy=key[2],**s,wer=s['edits']/max(1,s['words']),cer=s['character_edits']/max(1,s['characters'])))
    private=dict(evidence_sha256={str(Path(a.asr)):sha(a.asr),str(Path(a.qwen_asr)):sha(a.qwen_asr)},ASR_aggregate=aggregate,signals=signals,
        limitations=['ASR-only edit candidates; no listening performed','VITS base Portuguese dialect/accent cannot be rejected solely from ASR','Do not compare different cohorts/policies','PCM-rejected FLOAT is diagnostic and not a usable product output'])
    Path(a.private_report).write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n');Path(a.private_report).chmod(0o600)
    lines=['# Comparação inicial local, 03/11/2026','',
        'Medições na mesma A2000, validação comum e regra digital congelada. Qwen inclui gateway e DSP; F5/VITS medem motor e arquivos, incluindo conversões e fechamento. Não há transporte HTTP hipotético nos motores. Os modelos e repetições falhos ficam no denominador. Esta tabela não aprova identidade, pronúncia ou uma candidata para produção.','',
        '| Caso | Qwen TTFA p50 s | F5 NFE8 TTFA p50 s | F5 NFE16 TTFA p50 s | VITS base TTFA p50 s |',
        '| --- | ---: | ---: | ---: | ---: |']
    case_ids=[r['case'] for r in bench['qwen-warm'] if r['repetition']==0]
    for case in case_ids:
        values=[]
        for name in names:
            rs=[r for r in bench[name] if r['case']==case and r['outcome']=='ok'];m=median(rs,'ttfa_estimated_playback_ms')
            values.append((f'{m/1000:.3f}' if m is not None else 'indisponível')+(f' (n={len(rs)})' if len(rs)!=3 else ''))
        lines.append('| '+case+' | '+' | '.join(values)+' |')
    lines += ['', 'P95 indisponível com n=3 por caso. TTFA é reprodução FIFO estimada até início digital de fala, sem atraso do dispositivo. Ver também a disponibilidade dos samples de fala nos recibos, sem confundir as duas métricas.','',
        '| Sistema | PCM válidos / tentativas | RTF mediano observado* | Máximo de memória observado |',
        '| --- | ---: | ---: | --- |']
    for name in names:
        rs=bench[name];ok=[r for r in rs if r['outcome']=='ok'];mem=max((r.get('peak_allocated_mib') or 0 for r in rs),default=0)
        lines.append(f'| {name} | {len(ok)} / {len(rs)} | {median(ok,"rtf"):.3f} | '+(f'{mem:.1f} MiB allocator PyTorch' if mem else '10.674 MiB dispositivo, amostrado')+' |')
    lines += ['', '*A mediana RTF desta linha mistura os sete tamanhos e serve só como descrição; decisões usam cada frase. Picos de allocator e de dispositivo têm escopos diferentes. Não comparar diretamente os números de memória entre esses escopos.','',
        'F5 NFE8 teve duas conversões rejeitadas: pergunta/r0, pico FLOAT 1,14484 com 22 samples fora da faixa; siglas/r2, pico 1,03475 com um sample fora. Não houve clipping oculto. O ASR da pergunta usa FLOAT bruto e fica em coorte diagnóstica separada, sem TTFA de produto. F5 NFE16 completou 21/21 PCM, mas não alcançou o ganho material selado de 400 ms em nenhum caso. VITS base é um controle de idioma/velocidade antes de especializar a identidade.','',
        '## ASR auxiliar na repetição zero','',
        '| Sistema / coorte | Casos | Edições/palavras verbatim | WER auxiliar | CER auxiliar |',
        '| --- | ---: | ---: | ---: | ---: |']
    for s in aggregate:
        if s['policy']!='verbatim':continue
        lines.append(f'| {s["candidate"]} / {s["cohort"]} | {len(s["cases"])} | {s["edits"]}/{s["words"]} | {100*s["wer"]:.2f}% | {100*s["cer"]:.2f}% |')
    lines += ['', 'Comparar percentuais somente com os mesmos casos. As formas por extenso e as entradas efetivas de cada frontend permanecem separadas no recibo privado; números/siglas escritos de outro modo podem aumentar WER sem indicar pronúncia errada. ASR também pode errar com o timbre robótico. Não foi feita avaliação auditiva. Um sotaque português distinto não deve ser tratado como falha do modelo apenas por esse reconhecedor.','',
        'Os sinais de omissão, inserção, última palavra e n-gramas repetidos são alinhamentos de transcrição para revisão, sem afirmar que o áudio contém esses defeitos. Detalhes privados: `'+a.private_report+'`. Fontes: `'+str(root)+'`, `'+a.asr+'`, `'+a.qwen_asr+'`.','',
        'Os três roundtrips BigVGAN e os três VITS posterior→decoder têm textos e alvos diferentes da validação inédita e entre si. Seus controles de fonte estão na mesma coorte correspondente; não misturar esses percentuais com a síntese texto→fala nem entre codecs.']
    Path(a.markdown).write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(markdown=a.markdown,private_report=a.private_report,ASR_observations=len(receipts)),ensure_ascii=False))
if __name__=='__main__':main()
