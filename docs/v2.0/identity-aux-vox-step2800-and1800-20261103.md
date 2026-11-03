# VoxCPM2: controles auxiliares dos passos 2800 e 1800

O passo2800 recuperou a comparação ASR de12/135 edições observada em1800,
após19/135 em2600. Os descritores de voz variam por frase e não oferecem um
desempate consistente. Os dois checkpoints permanecem candidatos; esta análise
não aprova inteligibilidade, timbre robótico ou identidade por escuta.

Foram feitos somente dois passes novos de38 clipes CPU, um por checkpoint.
Cada manifest mantém idênticos31 itens anteriores: três âncoras reservadas,
Qwen processado, F5 especializado, Vox base e LoRA inicial. Substitui apenas
os sete clipes da etapa alvo. Os sete WAVs e o checkpoint de cada marco foram
vinculados ao SHA/passo do quality monitor antes de criar hardlinks próprios.
Os resultados800/1400, o controle0586 e seus recibos selados foram preservados.

As comparações abaixo reutilizam os resultados800/1400 já arquivados e o ASR
existente. São as mesmas sete frases de validação, frontend original e geração
eager NFE10; não são o benchmark HTTP com frontend numérico/reserva320ms.
Nenhuma nova síntese, ASR, instalação, mudança de treino ou uso de GPU ocorreu.

| Marco | ASR verbatim: edições/135 | Cosseno mediano auxiliar | F0 mediano dos clipes (Hz) | DP dos medianos F0 (Hz) | Centroide mediano (Hz) | Energia8–12kHz mediana |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 800, anterior | 14 (10,37%) | 0,988384 | 206,37 | 10,71 | 904,62 | 2,385% |
| 1400, anterior | 13 (9,63%) | 0,987949 | 226,06 | 12,50 | 940,92 | 2,528% |
| 1800, novo passe | 12 (8,89%) | 0,988285 | 205,32 | 11,23 | 1064,19 | 3,339% |
| 2800, novo passe | 12 (8,89%) | 0,987572 | 193,53 | 8,24 | 1028,22 | 3,197% |
| Três âncoras reais, textos diferentes | — | — | 200,06 | 2,56 | 897,64 | 2,962% |

O cosseno de2800 aumenta em4/7 casos contra1400 e3/7 contra800; sua mediana
não aumenta. O de1800 supera2800 em4/7,1400 em6/7 e800 em5/7. São diferenças
descritivas, sem limiar de identidade ou incerteza estatística calibrada.
O espectro de2800 fica menos distante das âncoras que1800 nesses dois
agregados de centroide/alta frequência, mas o texto e a fonética também mudam
essas medidas; proximidade numérica não estabelece caráter metálico.

Nas células seguintes, a ordem é **800 / 1400 / 1800 / 2800**. O recibo privado
retém os demais descritores e hashes por caso nos quatro marcos.

| Caso de validação | Cosseno auxiliar | Edições ASR verbatim |
| --- | --- | --- |
| v_command | 0,982151 / 0,981696 / 0,984112 / 0,985547 | 2 / 1 / 1 / 1 |
| v_reply | 0,988976 / 0,984423 / 0,986421 / 0,982987 | 1 / 0 / 0 / 0 |
| v_pressure | 0,988909 / 0,987949 / 0,989384 / 0,989860 | 3 / 2 / 3 / 2 |
| v_acronyms | 0,988384 / 0,988439 / 0,989377 / 0,989137 | 1 / 2 / 1 / 2 |
| v_question | 0,987854 / 0,987972 / 0,988285 / 0,987572 | 1 / 0 / 0 / 0 |
| v_sibilants | 0,986459 / 0,985087 / 0,984247 / 0,985373 | 1 / 1 / 1 / 1 |
| v_long | 0,991161 / 0,990350 / 0,991814 / 0,989914 | 5 / 7 / 6 / 6 |

O empate de12/135 não significa os mesmos erros:1800 tem uma edição a mais
em pressão e uma a menos em siglas. A sequência ASR já disponível em
1800/2000/2200/2400/2600/2800 é12/14/15/16/19/12 edições. Não há evidência
de regressão ASR monotônica; uma loss de validação menor também não resolve
o conflito entre esses indicadores e a identidade que o usuário espera.

Foi usado o helper imutável `training/v2_identity_aux.py`,
SHA`c9cc80bdd2ccad2b49e17b56a17b28d3b28907dd7fbdc19cf1aeb1edbbbc663f`.
O encoder local pertence ao Qwen e pode favorecer esse baseline; as âncoras
têm palavras diferentes. A análise iguala RMS ativo a−23dBFS, sem limitar ou
alterar os WAVs originais; isso não é normalização perceptual LUFS. YIN usa
máscara de energia, pode incluir erros de oitava/trechos surdos, e a energia
de bandas não mede sibilantes alinhadas nem robótica. Nenhum áudio foi ouvido.

Execuções CPU2/nice19/CUDA desativada:2800 em20,022s/RSS998,883MiB;
1800 em19,542s/RSS1003,680MiB. Ambas registram CUDA não inicializada.
Nenhuma âncora foi usada como prompt e o teste final de seis frases e os50
trechos reservados continuam fechados.

Artefatos privados mantidos em `/var/tmp/aegis-v2-work/identity-aux/`:

- `manifest-vox-step{1800,2800}38.private.json` e `vox-step{1800,2800}38/`:
  resultados38, embeddings, resumo por caso e recibo da execução.
- `snapshots/voxcpm2-step1800-ebb8e11f79f4/`: checkpoint
  SHA`808fc4342dc90e1924da10c7533b81ba70df3a55c52fe92b2f3115d422c9cb4a`.
- `snapshots/voxcpm2-step2800-4217a27bd24f/`: checkpoint
  SHA`b60e4d7e970f123e6dff1674747ddf768e27ebe01ce7e9b8a66ad17100469059`.
- Recibos1800/2800, respectivamente:
  SHA`d280a30882e9c1348201604ef00d3949900e44df94200745cfee6e7301b28df7` e
  SHA`8e7198724aa0f210c6222261cb6aee1f9a1614d2996e5702d576cf68fd4d8c82`.

Os checkpoints têm54.741.414 bytes cada e são hardlinks verificados dos
originais exatos; não são cópias de outro passo. Os resumos privados guardam
política ASR, hashes dos insumos e comparação completa800/1400/1800/2800.
Preservar ambos para a próxima decisão é justificável; escolher uma voz exige
avaliar clareza/identidade além desses indicadores auxiliares.
