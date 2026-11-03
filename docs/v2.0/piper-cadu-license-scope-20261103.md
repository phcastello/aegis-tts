# Piper Cadu: permissões verificadas e linhagem pendente

A linha pode continuar como experimento privado de pesquisa. A documentação
verificada não sustenta certificar uso irrestrito ou redistribuição comercial
dos pesos derivados. A questão concreta é a autorização da linhagem Lessac;
a GPL do runtime, por si só, não impede executar/modificar privadamente.

| Camada | Fonte primária | Evidência e limite |
| --- | --- | --- |
| Checkpoint Cadu | [`rhasspy/piper-checkpoints@f0a4c43…`](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/f0a4c43d31b4e533feac35ae3ac25a56eebab8e8) | O publicador declara MIT no repositório. Preservar licença e atribuição; isso não resolve sozinho a autorização dos materiais antecedentes. |
| Card Cadu | [MODEL_CARD pinado](https://huggingface.co/datasets/rhasspy/piper-checkpoints/blob/f0a4c43d31b4e533feac35ae3ac25a56eebab8e8/pt/pt_BR/cadu/medium/MODEL_CARD) | CC0 aparece na seção Dataset, não como licença explícita dos pesos; declara fine-tuning de Lessac medium. |
| Corpus Cadu | [OHF voice-datasets, README pinado](https://github.com/OHF-Voice/voice-datasets/blob/1692a7c509e35eceb5d3bd6a7581a104629ded0f/README.md) | Nabu Casa/OHF declara CC0 para os dados e solicita citação do repositório. Não se transfere automaticamente essa declaração aos pesos ancestrais. |
| Runtime usado | [Piper1.8, COPYING pinado](https://github.com/OHF-Voice/piper1-gpl/blob/639388b6317fc4731e91d53da42aea68fd4166ff/COPYING) | GPL-3.0-or-later. Execução/modificação privada e serviço local sem entregar cópias não exigem publicação automática; distribuição do runtime modificado requer cumprir notices/licença/fonte correspondente. A licença tampouco cobre automaticamente toda saída do programa. |

O arquivo local Cadu tem845.888.150bytes, SHA
`abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a`.
A revisão acima não alterou esse peso, o ambiente ou os resultados anteriores.

O [card Lessac medium pinado](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33a10e145e109b886d661984201ac3c4/en/en_US/lessac/medium/MODEL_CARD)
aponta Blizzard2013. O [contrato publicado pelos titulares](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html)
limita os materiais enumerados, como gravações, textos, recortes e labels,
à pesquisa e exclui desenvolvimento comercial. Ele não nomeia expressamente
checkpoints neurais ou derivados por fine-tuning. Portanto, o texto não prova
sozinho uma proibição automática dos pesos Cadu, nem fornece a autorização
específica que falta para sua linhagem.

O mantenedor explica que Piper não acrescenta restrições aos modelos, mas
deixa sem solução a incidência de licenças antecedentes; essa resposta não é
autorização dos titulares Lessac/Voice Factory. [Discussão primária271](https://github.com/rhasspy/piper/discussions/271#discussioncomment-7570436).

O checkpoint Lessac ancestral exato de Cadu não foi identificado. Não usar
um step de outro locutor como prova dessa identidade. A condição pendente é
uma autorização publicada aplicável ao ancestral e seus derivados, com
escopo operacional/comercial e redistribuição esclarecidos.

Decisão operacional: os experimentos privados continuam possíveis com os
insumos já obtidos. Não publicar os pesos nem descrevê-los como liberados
irrestritamente. Caso esta linha vença a avaliação técnica, distinguir o
resultado local de pesquisa das permissões de uso/distribuição demonstradas.
Não foi encontrada proibição expressa específica de servir esses pesos
localmente; isso também não equivale a autorização irrestrita certificada.

Foi uma rodada focada em fontes primárias e pins já usados, sem nova busca
de arquitetura, download de pesos, instalação, ASR, síntese ou GPU.
