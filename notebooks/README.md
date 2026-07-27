# Notebooks

Ordem recomendada de execução:

1. `01_coleta_e_preprocessamento.ipynb`
2. `02_modelo_lstm.ipynb`
3. `03_deploy_api.ipynb`
4. `04_monitoramento_e_escalabilidade.ipynb`

Os notebooks 01 e 02 geram os arquivos necessários para a API.

Todo o pré-processamento vive no **01**: coleta, limpeza, log-retorno,
split temporal, escalonamento e janelamento. O **02** carrega os arrays
prontos e apenas constrói, treina e avalia o modelo — não repete
nenhuma transformação. Por isso o 02 depende do 01 ter sido executado
antes: os intermediários de treino não são versionados.

A última seção do **02** exporta o modelo para ONNX, que é o formato
servido pela API — os notebooks 03 e 04 carregam o `.onnx`, não o
`.keras`. Sem essa etapa, retreinar deixaria a API servindo a versão
anterior em silêncio.

## Execute a partir da raiz do projeto

Os caminhos dos artefatos são relativos (`artifacts/…`, `models/…`), e o
diretório de trabalho do kernel precisa ser a raiz do repositório, não a
pasta `notebooks/`. Rodando do lugar errado, os artefatos vão parar em
`notebooks/artifacts/` e a API não enxerga nada do que foi gerado.

- **JupyterLab:** use o `./run-jupyter-lab.sh` na raiz.
- **VS Code:** configure `"jupyter.notebookFileRoot": "${workspaceFolder}"`.

Para conferir, rode numa célula:

```python
import os; os.getcwd()
```
