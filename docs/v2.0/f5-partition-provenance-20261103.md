# Proveniência da partição do F5 especializado

A exclusão do teste histórico de 50 itens do treino de especialização do F5 está **comprovada pela cadeia local preservada**. Esse teste já foi avaliado historicamente após a seleção automática do checkpoint. Não há prova de ausência de influência humana ou de escolhas posteriores. O novo teste comum de seis frases continua fechado.

O checkpoint especializado `b3faf799…` corresponde à época 29/update 3741. Seu header safetensors registra o fingerprint `657c4d46…`, reproduzido exatamente a partir do recibo histórico `cpu-check.json` e confirmado por `training-result.json`. Os manifests preparados têm os hashes históricos e coincidem por IDs, grupos e hashes originais de áudio com a partição preservada: 1025 itens de treino, 55 de validação e 50 de teste. As interseções entre os splits são zero nesses três critérios. A igualdade aritmética entre épocas e updates é apenas uma verificação auxiliar.

O trainer preservado usa somente o treino para gradientes e a validação para escolher o checkpoint, ajustar o aprendizado e parar. A preparação e a auditoria de entrada verificaram todos os splits, incluindo o teste. Os callbacks de síntese usaram o corpus histórico de arquitetura, sem chamar a avaliação dos 50 itens. A exclusão comprovada refere-se à especialização registrada, sem afirmar a composição do pré-treino da base.

Depois da escolha automática da época 29, o pipeline executou a avaliação final histórica e gravou resultados para os 50 itens, tanto na base quanto no checkpoint especializado. Os recibos e a sequência de execução confirmam essa exposição posterior à escolha. O código observado não retorna esses resultados ao treino ou à parada; isso não demonstra que os resultados nunca influenciaram decisões posteriores. Portanto, esses 50 itens não devem ser apresentados como um teste historicamente intocado.

A proposta antiga de partição por montagens posteriores de áudio foi descartada e não deve ser confundida com a partição preservada. Seus conjuntos propostos de validação/teste incluem itens usados pelo F5 no treino.

A auditoria leu somente o header do checkpoint e metadados, projetando IDs e hashes dos manifests sem decodificar transcrições. Não abriu áudio do teste histórico, não abriu nem calculou hash do novo corpus final, e não executou Torch, GPU, TTS ou ASR. Nenhum peso, split ou critério foi alterado. Este documento não publica IDs, transcrições, áudio ou resultados individuais do teste.

Prova privada preservada em `operations/f5-partition-provenance-20261103`, sob a raiz privada do laboratório: `AUDIT.md` SHA-256 `98dcec48d6886c6e30061e1096cb285bf2dd9bd5b0d495cb9d7d96f74edb4d30`; `provenance.private.json` SHA-256 `64691d0a00830170c572479c8bd25f7196611d5847af2221789eb28111f6e25f`; recibo SHA-256 `867f8ad7ea737e24904618eb1fa96c4843bd171eca953023d803378b0f535920`.
