# Changelog

Registro das mudanças relevantes do projeto.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/).
O projeto ainda não usa tags de versão, portanto as entradas são
organizadas por data.

## [2026-07-26]

### Adicionado

- Modelo e artefatos de inferência versionados no repositório
  (`models/lstm_final.keras`, `artifacts/ret_scaler.pkl`,
  `artifacts/inference_meta.pkl`, `artifacts/metrics.pkl` e
  `artifacts/AAPL_clean.csv`), permitindo executar a API logo após o
  clone, sem rodar os notebooks.
- Este `CHANGELOG.md`, com o histórico reconstruído a partir dos
  commits anteriores.

### Alterado

- Imagem de inferência trocou `tensorflow` por `tensorflow-cpu`, que
  não arrasta as bibliotecas CUDA da NVIDIA — inúteis, já que a API
  executa em CPU.
- `Dockerfile` remove `tensorflow/include` (287 MB de headers C++,
  necessários somente para compilar ops customizadas) após o
  `pip install`.

  Imagem final em 1,64 GB, com 257 MB de memória residente em
  execução. Validado em container: healthcheck `healthy`, `/ready`
  com modelo e scaler carregados, `/metrics` exportando e `/predict`
  devolvendo os mesmos valores da imagem anterior.

  Os notebooks seguem instalando o `tensorflow` completo, já que o
  treino pode se beneficiar de GPU.

### Corrigido

- README principal e os READMEs de `models/` e `artifacts/` afirmavam
  que os artefatos não eram versionados e precisavam ser copiados
  manualmente. A documentação passou a descrever o conteúdo já
  presente e a explicar o papel do `.gitignore`.

## [2026-07-14]

### Adicionado

- API FastAPI (`app/`) servindo o modelo, com os endpoints `/predict`,
  `/health`, `/ready`, `/feedback` e `/metrics`, além de validação por
  schemas Pydantic e configuração por variáveis de ambiente.
- Inferência recursiva sobre log-retornos no `Predictor`, com
  reconstrução do preço por `P_hat = P_prev * exp(pred_return)`,
  serialização das chamadas ao TensorFlow e aquecimento do grafo na
  inicialização.
- Instrumentação Prometheus: latência HTTP, duração da inferência,
  CPU e memória residente do processo, prontidão do modelo e erro
  preditivo alimentado pelo `/feedback`.
- Stack de observabilidade com Prometheus e Grafana via
  `docker-compose.yml`, incluindo dashboard e regras de alerta.
- `Dockerfile` com usuário não-root e healthcheck.
- Notebooks 03 (deploy) e 04 (monitoramento e escalabilidade),
  testes automatizados com predictor falso, script de teste de carga,
  `Makefile` e README do projeto.

### Alterado

- Notebooks movidos para `notebooks/`.
- Alvo do modelo passou a ser o log-retorno. Os artefatos foram
  renomeados de acordo: `scaler.pkl` virou `ret_scaler.pkl` e
  `meta.pkl` virou `inference_meta.pkl`.

### Removido

- Artefatos intermediários de treino (`X_train.npy`, `X_valid.npy`,
  `y_train.npy`, `y_valid.npy`) e o checkpoint `lstm_best.keras`,
  que não são usados na inferência.

## [2026-06-21]

### Adicionado

- Notebook 01, com coleta via `yfinance` (AAPL), limpeza, análise
  exploratória, split temporal, escalonamento e criação das janelas.
- Notebook 02, com a arquitetura LSTM, busca de hiperparâmetros,
  treino final e avaliação por MAE, RMSE e MAPE contra um baseline
  naïve.
- Primeira versão do modelo treinado e dos artefatos de apoio.
