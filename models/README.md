# Modelo treinado

Esta pasta já contém os dois modelos versionados no repositório:

- `lstm_final.keras` — treinado pelo notebook 02, artefato de origem
- `lstm_final.onnx` — derivado do `.keras`, é o que a **API carrega**

A API não usa o `.keras`: servir em ONNX dispensa o TensorFlow no
runtime, o que reduz a imagem de 1,64 GB para 495 MB e a memória
residente de 722 MB para 159 MB.

Depois de retreinar, reexporte:

```bash
python scripts/export_onnx.py
```

O script verifica a equivalência entre os dois e falha se a diferença
passar de `1e-4`, para o ONNX nunca divergir em silêncio do modelo
treinado.

Para armazenar modelos grandes no Git, prefira Git LFS ou um
repositório de artefatos.
