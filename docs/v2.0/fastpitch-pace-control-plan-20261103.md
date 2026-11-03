# Controle CPU de duração no FastPitch8000

Este é o registro do plano anterior à execução. O
[controle e a limpeza foram concluídos](fastpitch-pace-control-result-20261103.md)
posteriormente; na preparação descrita aqui não havia síntese ou ASR executados.
A hipótese é que
`pace0.8` expanda as durações de grafemas/tokens previstas em aproximadamente1,25×
e reduza trechos curtos/omitidos. Um ratio médio de duração0,7704 passaria a cerca de
0,963 antes do arredondamento. Isso não comprova a causa do erro de idioma;
o teste compara somente pace no mesmo checkpoint8000, CPU FP32.

A API pinada aceita `infer(inputs,pace=...)`; o regulador não-JIT usa
`(duration.float()/pace+0.5).long()`. O helper novo
`training/v2_fastpitch_pace_control.py` reutiliza Backend, carga estrita e
bounded_regulator existentes, sem alterar o treinador ou benchmark. Ambos
os `--action plan` passaram sem importar Torch: rascunho pendente e plano
final vinculado ao8000 exato.

O checkpoint foi preservado por hardlink próprio do retainer8000, após
conferir passo, sidecar, tamanho549.622.954 bytes e SHA
`ae9834fb562562a405febfcaa325f57711779aeadda8ebfa933e12ce02c922c8`.
Não se substitui por outro passo. O plano vinculado fica em
`/var/tmp/aegis-v2-work/fastpitch/pace-control-step8000-eval-v2/plan.json`.
O checkpoint fica fora desse output, em
`pace-control-step8000-preparation-v2/checkpoint8000/`, evitando contá-lo
como novo áudio ou duplicar seus dados.

O desenho pré-fixado tem sete frases de validação, n1 por pace,14 gerações.
A ordem1.0/0.8 alterna por caso; o seed40733133 é reiniciado a cada geração.
O texto efetivo de refinement foi conferido contra os sete casos do7000.
Não há warmup, alteração de pesos, teste final, alvo50 ou mudança de pace
do produto. FP32 CPU também difere do benchmark GPU/BF16: seus tempos
não serão usados como latência da A2000.

Há dois modelos residentes, acústico e vocoder, numa única instância de backend.
O orçamento é5GiB de RSS, ainda uma estimativa de carga/ativações, com
monitor próprio amostrado0,25s e abortamento do processo de controle.
O preflight exige8GiB disponíveis, teto global40GB mais50MB de saída e
reserva física5GiB. São mantidos somente os14 PCM24k16bit: o limite60s/caso
gera no máximo40.320.616 bytes de WAVs, mais JSON, abaixo de50MB. Não se
gravam cópias FLOAT; hashes e contagens finitas/overshoot/saturação são
registrados antes da conversão, sem ganho ou clipping adicional. O clamp
intrínseco do BigVGAN continua explícito. O limite de tempo é1800s.

Falhas continuam no denominador14. Se houver ausência/invalidade de áudio,
o relatório deve declarar o conjunto incompleto; não se compara apenas o
subconjunto que passou. O helper guarda jobs ASR com aliases por pace e
`input_audio_sha256`, e não chama o reconhecedor. A execução TTS e o ASR
sequencialCPU2/nice19 receberam aprovação técnica do operador, condicionada
ao peerreview e ao término do ASR público8000 já em execução.

SHA atual do helper:
`67bee5515049dae75d61898783efd86090e2bb26555fec917f007299de8cd13e`.
Nenhum modelo foi construído, nenhum CUDA/serviço foi usado e o treino segue.
