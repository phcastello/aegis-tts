# VoxCPM2: preparação do streaming nativo

Preparado em 2026-11-03, somente com testes CPU. Não carregamos pesos, não
inicializamos CUDA, não executamos ASR, não enfileiramos GPU e não alteramos os
probes compartilhados, os helpers comuns ou a produção. A execução real será
um job serial autorizado do operador; os números desta preparação não são
medidas de velocidade do modelo.

Novos arquivos: `training/v2_voxcpm_streaming_probe.py` e
`tests/test_v2_voxcpm_streaming_probe.py`. Recibo e log CPU privados:
`/var/tmp/aegis-v2-work/voxcpm2/streaming-preparation/{cpu-proof.json,cpu-tests.log}`.
O recibo fixa hashes do código novo e dos helpers importados, NumPy1.26.4,
SciPy1.17.1 e fonte limpa no commit
`cce58a5b59303c9bd63d12f23afe6a49a4a80c59`. `torch_imported=false` e nenhuma
biblioteca libtorch/libcuda carregada no processo dos testes.

## Semântica da fonte oficial

A API usada é `VoxCPM2Model.generate_streaming`, diretamente no motor, sem o
wrapper de normalização/denoising. O método `_inference` entrega apenas o patch
latente novo; `_generate` o passa para `audio_vae.streaming_decode()`. O VAE
mantém estados das convoluções causais entre patches e restaura os forwards e
limpa os estados quando o contexto fecha. Não há overlap de áudio para
recortar nem crossfade acrescentado pelo adaptador. `streaming_prefix_len=4`
limita a história latente mantida pelo gerador; não exige acumular quatro
patches antes de falar.

A configuração local verificada tem patch_size4 e produto dos strides do
decoder1920: cada patch corresponde a7680 samples48k, ou160ms. A primeira
disponibilidade depende de prefill, DiT, VAE, cópia para CPU e resampling. Esse
tamanho descreve a granularidade; não demonstra que o motor acompanha a
reprodução. O adaptador registra os comprimentos realmente recebidos.

Fontes primárias pinadas:

- [Modelo, optimize e geração](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py)
  — optimize279, geração468/661 e yield de patch novo1105.
- [VAE causal com estado](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/audiovae/audio_vae_v2.py)
  — streaming_decode475 e contexto504–590.

## PCM24k com estado e flush

`HalfRateFIR` usa o mesmo FIR de41 taps Kaiser beta10 da chamada de referência
`scipy.signal.resample_poly(float32,1,2,window=('kaiser',10.))`. Executa convolução
causal com histórico limitado a40/41 samples e fase global fixa. Retém o atraso
de20 samples48k (0,4167ms), compensa os10 outputs de atraso e só em EOF normal
estende a entrada com20 zeros. O total termina exatamente em ceil(N/2). O
tempo para obter a entrada futura necessária está incluído na chegada PCM;
não se deslocam timestamps para esconder atraso do filtro.

Nos testes, a concatenação do streaming ficou **bit a bit igual** ao float32
batch em13 comprimentos de0 a10003 samples, chunks irregulares e impulso. O
PCM resultante também ficou idêntico byte a byte. Os primeiros20 samples
nativos não produzem PCM prematuro. Isso prova a equivalência do resampler
nessas versões; a equivalência entre os próprios VAE batch e streaming ainda
precisa de prova no modelo real.

Não há ganho, normalização ou clipping silencioso. Amostra não finita ou PCM
fora da faixa invalida a tentativa. Cancelar fecha o gerador e descarta o
histórico do FIR, sem flush artificial; EOF normal faz flush. Os testes também
verificam fechamento em falha do gerador, sink, cancelamento antes do primeiro
passo e cancelamento após um chunk, sem avançar o próximo passo do modelo.

## Medição e execução preparada

Cada tentativa começa antes de criar/avançar o gerador. Registra chegada de
cada chunk nativo e de cada PCM, incluindo FIR, quantização e disponibilidade
no consumidor local. Os mesmos `pcm_metrics` e regra de onset/FIFO do benchmark
comum calculam primeiro PCM, fala disponível, TTFA estimado, lacunas, EOF e
RTF. O atraso do dispositivo e transporte HTTP não são medidos nem supostos.
WAVs e JSONs de auditoria são gravados depois do EOF cronometrado do stream.

O caminho compilado chama `model.optimize()` oficial e exige quatro wrappers
compilados. Conta replays reais de CUDAGraph nas três chamadas excluídas de
cada caso e restaura o hook antes das repetições quentes. Cada caso, inclusive
o longo, recebe suas próprias três chamadas excluídas; primeira compilação e
caso novo têm rótulos separados. O resumo exige prova de replay real para
rotular o caminho como verificado. Pesos, buffers e KV caches devem permanecer
na GPU integralmente; a flag `--topology base` não cria LoRA.

O runner exige pai GPUworker, GPU vazia antes da carga, `pause-asr`, verificação
de CPU ASR ociosa, hashes locais e cache/space budget. Usa os helpers existentes
apenas por import, sem modificá-los. O watchdog de orçamento limita cache a
512MiB por padrão e preserva reserva de5GiB e teto40GB. A margem declarada é de pelo menos1GiB;
não há download nem ambiente novo. Traces e WAVs desses dois casos custam
poucos MB; tempo e pico reais de GPU permanecem pendentes.

Comando preparado para o operador enfileirar, sem executar diretamente:

```sh
/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m training.v2_voxcpm_streaming_probe --action execute --gpu-authorized \
  --topology base --optimize --nfe 4 --matmul-precision highest \
  --cases v_reply v_long --repetitions 3 \
  --run /var/tmp/aegis-v2-work/voxcpm2/base-streaming-nfe4-controlled \
  --cache /var/tmp/aegis-v2-work/voxcpm2/compile-cache --cache-limit-mib 512
```

NFE6/10 e TF32 `high` são configurações separadas e precisam de run novo. O
modo `merged` exige novamente ASR e comparação de áudio após recarregar os
pesos selecionados; não presume equivalência numérica ou de identidade do
LoRA. A opção `--cancel-after-native-chunks N` é diagnóstico explícito e suas
tentativas canceladas não contam como benchmark completo.

Dez testes CPU passaram. Plano padrão `--action plan` não carrega modelo.

```sh
CUDA_VISIBLE_DEVICES='' \
  /var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python \
  -m unittest tests.test_v2_voxcpm_streaming_probe -v
```

Decisão futura: RTF longo deve ficar abaixo de1, junto a TTFA, lacunas e
conteúdo íntegro. RTF curto acima de1 por custo inicial não descarta sozinho
streaming nativo. Nenhum resultado de velocidade vale se a pronúncia ou a
identidade robótica regredir. NFE4/6/10 são candidatos acústicos distintos;
clareza, conteúdo e timbre ainda não foram julgados nesta preparação.

## Evidência local que motiva o diagnóstico

Os runs por arquivo executados pelo operador na A2000 têm três repetições
quentes de `v_reply`, base sem LoRA, compilação oficial e seed fixo. Os arquivos
`summary.json` e `results.jsonl` de cada diretório abaixo foram lidos para esta
tabela; são medições do host, separadas da prova CPU do adaptador.

| Run em `/var/tmp/aegis-v2-work/voxcpm2/` | EOF p50 s | Áudio s | RTF p50 | Nativo por passo base-LM, amortizado ms |
| --- | ---: | ---: | ---: | ---: |
| `base-compiled-controlled` (NFE10/highest) | 3,3185 | 1,76 | 1,8855 | 331,30 |
| `nfe06-highest-controlled` | 2,5837 | 1,44 | 1,7943 | 322,48 |
| `nfe04-highest-controlled` | 3,1211 | 1,76 | 1,7734 | 311,65 |
| `nfe10-high-controlled` | 3,2998 | 1,76 | 1,8749 | 329,52 |

O NFE4 mediu33 chamadas do estimador DiT por11 patches (3 por patch), e o
NFE10/high mediu99 por11 patches (9 por patch). A redução não resolveu o RTF
por arquivo. O custo amortizado por passo base-LM caiu cerca de331 para312ms;
ele inclui prefill, DiT, VAE e outros custos da chamada inteira, portanto não
isola um kernel AR nem identifica a causa no hardware. Ainda não houve perfil
que permita atribuir o limite a largura de memória, driver ou outra causa
específica. NFE6 também alterou a duração gerada; a mudança exige checagem de
conteúdo, não demonstra ganho de clareza.

A janela076a não foi inserida: o job077 FastPitch já havia retomado às09:39:43.
O operador preservou o treino e reservou o streaming para um próximo
checkpoint íntegro,4000 ou6000. Nenhum job foi enfileirado por este subagente.
