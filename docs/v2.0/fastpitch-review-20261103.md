# Revisão independente do runner FastPitch — 03/11/2026

Revisão somente leitura de `training/fastpitch_experiment.py` e NVIDIA DLE
fixado em `729963dd47e7c8bd462ad10bfac7a7b0b604e6dd`. Nenhuma GPU foi
iniciada pela revisão; as correções pertencem ao responsável pelo runner.

## Problema BF16 encontrado e correção verificada

O upstream converte fronteiras cumulativas de duração para o dtype do encoder
em `regulate_len`. BF16 perde precisão inteira acima de256frames: pode
omitir/repetir tokens. Promover somente a loss para FP32 não resolve isso.

Execução CPU da função upstream com400 tokens one-hot e duração1: FP32/FP16
mantiveram400 tokens; BF16 deixou72tokens sem frames e36 com mais de um
frame (máximo3). Recibo:
`/var/tmp/aegis-v2-work/fastpitch/independent-review-20261103.json`.

O agente aplicou wrappers fora da árvore NVIDIA hashada: entradas e
acumuladores de `regulate_len`/`average_pitch` em FP32, somente valores
acústicos finais convertidos. Verificação independente, BF16/autocast CPU
ligado/desligado: durações FP32 `[257,1,257]` produzem515frames exatos;
400tokens com duração1 produzem400frames exatos, erro0 nos quatro casos.
Recibo com hash do runner:
`/var/tmp/aegis-v2-work/fastpitch/independent-review-patched-20261103.json`.
Labels já quantizados não recuperariam257 a partir de256; o prior FP32 atual
promove a atenção e os labels. Os pilotos anteriores são engenharia e não
comprovam qualidade da receita corrigida.

## Conferências e limites

- Forward sob autocast configurável; outputs promovidos a FP32 antes de
  losses/CTC. FP16 usa GradScaler; BF16 não usa. Há rejeição de loss/gradiente
  não finito e limite para overflows consecutivos.
- MAS usa `mas_width1` upstream, com correção somente de dispositivo/conversão
  CPU. Prior beta-binomial tem orientação e parâmetros upstream. CTC treina
  atenção; pressão de binarização cresce após warm-up.
- **Mel compatível comprovado:** um segundo real da validação Aegis produziu
  `[100,93]` igual bit a bit ao extrator BigVGAN fixado, erro0,0. FFT1024,
  hop256, padding384, Hann, magnitude epsilon1e-9, Slaney/librosa, log1e-5.
  Os mels Vocos não foram usados com esse vocoder.
- Pitch mean/std usam somente voiced do treino da etapa; zeros ficam zeros.
  Energia L2 do mel log corresponde ao upstream. WORLD não comprova
  vozeamento/identidade robótica. Interpolação F0 entre zero/voiced merece
  inspeção se a fala apresentar instabilidade.
- Inicialização PT→alvo: strict model, mesma arquitetura/vocab/mel, buffers
  pitch da etapa nova e optimizer/scheduler/scaler/RNG/dados novos. A mudança
  das estatísticas requer novo gate de pronúncia e identidade.
- Resume restaura pesos, optimizer, scheduler, scaler, RNG, passo/época/cursor.
  Permutação é seed+época. Mudança de recursos exige flag/recibo. SIGINT/SIGTERM
  finalizam update completo e gravam atomicamente; OOM/NaN preservam o último
  checkpoint anterior. Não recuperam estado parcialmente atualizado.
- Best é hardlink do objeto selecionado antes de próximas trocas atômicas de
  last; permanece íntegro. Margem de gravação/retensão são limitadas; a
  supervisão externa ainda garante GPU exclusiva e orçamento global.

**Loss de validação usa duração MAS e pitch reais.** Mel/loss melhor não
comprova geração texto→fala. São necessárias sondas inéditas sem teacher
forcing, diagnósticos de duração/alinhamento e gate de idioma antes da voz.
O teste final permanece fora dessas decisões.

Foi solicitado incluir agenda de binarização e critérios de parada no config
retomável, além de optimizer/dados. Duplicações de IDs na preparação pública
foram comunicadas e precisam de rejeição/quarentena antes do cache longo.
Os originais permanecem preservados.

## VITS: piloto e retomada observados

Filas051/052 terminaram com exit0: passo3/cursor3 foi salvo e outro processo
retomou até passo4/cursor4, com atualização real do encoder. Pico alocado:
1.706,976MiB. Checkpoint passo4:886.864.812bytes,
SHA-256 `b60037df82013ded6a0d5062d9ba570ebab085eec9552d6119f7fb74e9a72bec`.
Três sondas pós-resume foram finitas, sem overshoot. Isso qualifica o
mecanismo; o baseline de idioma necessita de adaptação PT-BR separada.
