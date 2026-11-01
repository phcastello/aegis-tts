# Nova voz: preparação sem treino nem substituição da produção

Estado em 01/11/2026: **scripts preparados; nenhum checkpoint/ONNX baixado,
nenhum fine-tuning, reconstrução neural ou deploy da nova voz executado**.
A instrução posterior do proprietário exige autorização explícita antes dessas
operações. O áudio `master_processed.wav` também ainda não existe.
Os testes locais usam somente sinais sintéticos pequenos. Eles verificam o
pipeline de software, sem demonstrar identidade, efeito ou pronúncia neural.
A entrega de latência da voz atual continua independente.

## Seleção limitada a duas opções

| Opção | Evidência e decisão |
| --- | --- |
| FastPitch + HiFi-GAN/NeMo | Primeira avaliação. O [catálogo oficial](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/checkpoints.html) e a [lista implementada de FastPitch](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/tts/models/fastpitch.py) oferecem en/de/es/zh; não fornecem uma base FastPitch PT-BR comprovada. O inglês ARPABET não cobre o frontend brasileiro. Exigiria inventário fonético PT-BR, embeddings adequados, normalização, realinhamento e treinamento linguístico prévio. O plano de 1–2 horas não é garantia suficiente para esse trabalho. Por esse impedimento concreto não a escolhemos como primeira receita executável. |
| **Piper VITS pt_BR-cadu-medium** | Segunda e única alternativa: checkpoint PT-BR treinável, frontend `pt-br`, exportação ONNX e decoder HiFi-GAN no mesmo modelo. Escolhida para o **piloto de viabilidade**, sujeito a reconstrução e revisão do efeito. Não é uma voz final aprovada nem escolha de provedor/voz do dataset. |

O [tutorial Riva indicado](https://docs.nvidia.com/deeplearning/riva/archives/2-24-0/public/tutorials/tts-finetune-nemo.html)
mostra a adaptação de gerador e vocoder, mas seu exemplo é LJSpeech inglês.
O [GLaDOS](https://github.com/davesarmoury/GLaDOS) demonstra esse tipo de separação;
seus comandos/checkpoints antigos não foram copiados. Riva não será necessário.
A perda de efeitos no codec Qwen observada pelo proprietário é uma restrição:
não se propõe fine-tuning desse gerador com o codec inalterado.

## Base, licença e representação

A [card cadu](https://huggingface.co/rhasspy/piper-voices/blob/main/pt/pt_BR/cadu/medium/MODEL_CARD)
identifica PT-BR, um locutor, 22.050 Hz e dataset CC0. Os repositórios de
[checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/main/README.md)
e de [vozes ONNX](https://huggingface.co/rhasspy/piper-voices/blob/main/README.md)
declaram MIT. O código [OHF Piper](https://github.com/OHF-Voice/piper1-gpl/blob/639388b6317fc4731e91d53da42aea68fd4166ff/COPYING)
é GPL-3.0-or-later. Esses componentes têm licenças distintas; conservar avisos
se forem distribuídos. O card também registra origem em lessac inglês, mas a
base escolhida **já foi ajustada para PT-BR**, com config `espeak.voice=pt-br`;
não estamos tratando um checkpoint inglês como português.

Os identificadores e hashes dos artefatos públicos ficam em
[`training/recipe.json`](../training/recipe.json). O checkpoint tem 845.888.150
bytes; o ONNX de referência, 62.950.044 bytes. Só metadados, fontes e cards
públicos pequenos foram consultados. Pesos futuros e todos os WAVs/manifests
privados ficam fora do Git.

VITS não oferece o caminho literal mel → vocoder separado do FastPitch.
Ele usa espectrograma linear no encoder posterior durante treino, espaço
latente contínuo, alinhamento monotônico texto/áudio e decoder HiFi-GAN;
o mel participa da perda de reconstrução. Não há codec discreto Qwen.
Isso torna a alternativa compacta, mas **não prova que preserve o efeito**.
Gerador textual, posterior, fluxo e decoder precisam aprender o áudio tratado;
o treino preparado não congela o decoder nem usa áudio-base como alvo final.

A verificação futura é `python -m training.reconstruct`, com pesos locais e
amostra **sintética já tratada**; nunca uma referência pessoal protegida.
O script roda em CPU com uma thread e semente fixa, exporta A (alvo resampled)
e B (posterior → decoder), hashes, duração, pico, RTF e erro espectral.
Não baixa pesos nem começa pelo modo padrão. Exige `--execute` e identificação
da autorização futura. O proprietário compara os pares privados: timbre,
aperiodicidade/aspereza, quantização tonal, sibilantes e ataques. Métricas
espectrais não homologam o efeito. Se o decoder suavizar o efeito, o piloto
precisa adaptá-lo e repetir esse teste antes de investir no treino completo.
A comparação neural real permanece pendente; não há resultado inventado.

## Entrada, frontend e treino futuro

`training.prepare_dataset` importa `processed/dataset.jsonl` do pipeline único
Melodyne. Exige `quality_status=accepted`, `alignment_status=aligned`,
`content_review=approved`, hash, duração e origem no master tratado. Pendências
ficam no relatório; não entram no treino. O script preserva `group_id` e os
splits decididos no corpus, mantém testes finais fora do trainer e reamostra
explicitamente para 22.050 Hz. Não normaliza volume, não muda velocidade nem
remove margens. Texto revisado permanece em NFC; `spoken_text`, se fornecido,
precisa ser a transcrição realmente aprovada, não uma correção automática.

```bash
python3 -m training.prepare_dataset \
  --manifest /CAMINHO_PRIVADO/processed/dataset.jsonl \
  --output /CAMINHO_PRIVADO/training-input
python3 -m training.train_piper \
  --prepared /CAMINHO_PRIVADO/training-input \
  --checkpoint /PESOS_LOCAIS/cadu.ckpt \
  --output /CAMINHO_PRIVADO/piper-pilot
```

O segundo comando **apenas imprime o plano**, sem carregar pesos ou treinar.
No futuro, execução requer `--execute --owner-authorization ID
--exclusive-gpu-window ID`, áudio tratado revisado e checkpoint verificado.
O programa verifica memória livre e nunca interrompe produção por conta própria.

Antes de autorizar o treino, preparar um ambiente Python 3.12 separado:

```bash
python3.12 -m venv /AMBIENTE_PRIVADO/piper-venv
/AMBIENTE_PRIVADO/piper-venv/bin/pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r training/requirements.lock
```

A instalação não foi executada neste trabalho. O lock fixa também dependências
transitivas; o Torch CUDA 12.1 foi escolhido para o driver 535 deste host.
O alinhador Cython deve ser compilado no ambiente separado a partir da revisão
OHF fixada na receita: seguir `build_monotonic_align.sh` e o build de extensão
na [documentação dessa revisão](https://github.com/OHF-Voice/piper1-gpl/blob/639388b6317fc4731e91d53da42aea68fd4166ff/docs/TRAINING.md).
Conservar `pip freeze`, commit, compilador e log do primeiro build autorizado;
não instalar NeMo/Riva ou dependências no container atual.

O frontend é eSpeak-ng PT-BR e mapa IPA de 256 símbolos.
`python -m training.frontend_check --manifest /PRIVADO/training-input/trainval.jsonl
--voice-config /CONFIG_LOCAL/cadu.onnx.json --output /PRIVADO/frontend.json`
verifica cada fonema contra `phoneme_id_map` e falha em símbolos desconhecidos.
A verificação neural/eSpeak não foi executada: o ambiente isolado ainda não foi
instalado; a mecânica de cobertura passou com fonemas sintéticos. Números, datas, siglas e nomes requerem normalização
revisada e a mesma regra na inferência. As expansões efetivas do provedor podem
diferir do pedido: aprovação de conteúdo é independente de alinhamento temporal.
O alinhamento monotônico interno do VITS não substitui essa checagem.
A avaliação deve incluir ç, acentos, vogais nasais, lh/nh, r/rr, encontros,
primeiras consoantes e empréstimos técnicos, além dos textos reservados.

A receita usa 22.050 Hz, FFT/janela 1024, hop 256, 80 mels, fmin 0, fmax Nyquist,
janela Hann, espectrograma sem center e log natural com piso 1e-5 na perda mel.
Encoder e vocoder compartilham exatamente esses parâmetros. `recipe.json` é
fonte legível da configuração. Não aplicar outro normalizador de mel entre eles.

`train_piper.py` preserva a partição explícita, pois o DataModule upstream faz
`random_split`. A visualização periódica usa somente validação; o teste final
não entra na seleção de checkpoint. A cópia inicial é verificada por hash e
compatibilidade de **todos** os pesos do gerador; um módulo vocal aleatório não
é aceito como ajuste válido. O modelo e os discriminadores aprendem juntos,
com retomada por `last.ckpt` e identidade do manifesto. Dez épocas iniciais são
um ensaio operacional seguido de avaliação, não promessa de convergência.
Não confundir os latentes contínuos VITS com os mels previstos no método
FastPitch: este último exigiria ajuste adicional de HiFi-GAN com mels previstos
**apenas do treino**, sempre separados de validação/teste; essa etapa não é
transferida mecanicamente para VITS.

## Recursos, API e critérios de promoção

Host confirmado: RTX A2000 12 GB, 11.514 MiB visíveis; produção já ocupa
aproximadamente 10.674 MiB. Não cabe um treino junto da inferência atual.
As [notas de hardware Piper](https://github.com/OHF-Voice/piper1-gpl/blob/639388b6317fc4731e91d53da42aea68fd4166ff/docs/TRAINING.md)
registram experiências com 8 GB, mas isso não garante esta receita nessa GPU.
O ensaio autorizado futuro começa com batch 2, uma thread/worker, acumulação 4,
FP32 e mede pico real antes de prolongar. A janela exclusiva precisa ser
combinada; o script exige pelo menos 9.000 MiB livres. Não foi usada GPU aqui.

`training.piper_adapter.PiperUnitAdapter` é um adaptador candidato **inativo**:
ONNX CPU, uma thread por sessão, um cálculo sequencial por vez, sem rede,
sem WORLD. Recebe uma unidade textual completa, gera PCM, reamostra internamente
22.050 → 24.000 Hz e emite blocos de até 50 ms. Cada sentença é calculada antes
de emitir seus blocos; isso não é streaming dentro da sentença. Não há promessa
de latência antes de medir o modelo candidato no endpoint real.
Cancelamento descarta o resultado do cálculo iniciado e interrompe as próximas
emissões; não se promete abortar um kernel ONNX já iniciado. O serviço existente
continua responsável por filas limitadas, prioridade, request_id, SSE, WAV e
ordem. O adaptador tem teto de texto/duração e nenhum estado acústico partilhado
entre unidades. Seu método assíncrono implementa a assinatura atual `AudioAdapter`
(`SpeechSegment`, evento de cancelamento → iterador PCM), com cálculo fora do
event loop. A ligação em `SpeechService` permanece inativa até uma futura
implementação autorizada; nenhum endpoint novo é exigido do cliente.

Perfil novo: `AegisVoicev2.0-PiperPTBR-pilot`; o perfil protegido atual não muda.
Só promover após A/B privado aprovado pelo proprietário, pronúncia PT-BR,
números/datas/nomes/frases novas sem omissões/repetições, efeito reconhecível,
sem clipping/cortes/estalos, primeira palavra audível e EOF medidos, CPU/RAM,
cancelamento/fila e unidades sucessivas no serviço real. Até lá, WORLD permanece
na voz atual e a candidata não substitui produção.
