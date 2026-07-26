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
- Seção 8.1 do notebook 02, com diagnósticos no espaço de retorno:
  acurácia direcional, R² contra prever zero, correlação entre
  previsto e real, razão de desvios e proporção de altas. MAE, RMSE e
  MAPE em preço são dominados por P_{k-1} e não distinguem uma rede
  que aprendeu algo de uma que devolve sempre ~0.
- Chaves `ganho_vs_naive_%`, `return_diagnostics` e `n_validacao` no
  `artifacts/metrics.pkl`, documentadas em `artifacts/README.md`.

### Alterado

- O `/predict` devolvia sempre `"symbol": "AAPL"`, vindo do
  `inference_meta`, independentemente dos preços recebidos. Como o
  modelo consome log-retornos, que são adimensionais, a API aceita a
  série de qualquer ação — e rotulava todas como AAPL.

  A resposta passou a separar `symbol`, a ação que a requisição diz
  representar, de `model_trained_on`, a ação em que o modelo foi
  treinado. Quando divergem, o modelo está sendo aplicado a uma ação
  diferente da que viu no treino, e isso fica explícito. O
  `PredictRequest` ganhou um campo opcional `symbol`, normalizado para
  maiúsculas sem espaços, que apenas rotula a resposta: não troca de
  modelo nem altera a previsão.

  Em `/health` e `/ready` o campo `symbol` virou `model_trained_on`.
  Quebra de contrato para quem lia esses campos; o `load_test.py`
  consome apenas `min_prices`, que não mudou.

  Os notebooks 03 e 04 não importam de `app/`: mantêm cópias próprias
  da API para ficarem autocontidos, e por isso seguem executando sem
  erro, mas as saídas gravadas neles ainda exibem o campo `symbol`
  antigo. Alinhá-los exige reexecutar os dois notebooks, e não apenas
  editar as células.
- Todo o pré-processamento passou a viver no notebook 01. Antes o 01
  escalonava preços com `MinMaxScaler` e gravava janelas que o 02
  nunca lia: o 02 recarregava o CSV e refazia tudo com log-retorno e
  `StandardScaler`. Eram dois pré-processamentos incompatíveis, e o
  cabeçalho do 02 declarava como pré-requisito arquivos que ele não
  abria.

  O 01 agora calcula log-retorno, faz o split, ajusta o `ret_scaler` e
  janela a série, gravando também `base_valid.npy`, `true_valid.npy` e
  `valid_dates.npy` — os preços de referência e as datas que o 02
  precisa para avaliar em escala real. O 02 apenas carrega os arrays e
  treina. O `ret_scaler.pkl` passou a ser gravado pelo 01, que é quem o
  ajusta; o 02 só o consome.

  O modelo **não** foi retreinado: os arrays gerados pelo 01
  refatorado são idênticos, elemento a elemento, aos que o 02
  calculava internamente, e o `ret_scaler` reproduz o `mean_` e o
  `scale_` do artefato já versionado.
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
- O exemplo de `PredictRequest` trazia apenas 4 preços, então o
  "Try it out" do Swagger falhava com HTTP 422 no estado padrão. O
  exemplo passou a conter os 61 fechamentos reais que o modelo atual
  exige, e a descrição do campo `prices` aponta para o `min_prices`
  do `/ready`.
- `FeedbackRequest` não declarava exemplo, então o Swagger montava um
  a partir do `gt=0` e exibia `1` nos dois campos — erro absoluto zero,
  que não ilustra o propósito do endpoint. O exemplo passou a usar o
  D+1 devolvido pelo exemplo de `PredictRequest` (297.71) contra um
  preço observado de 298.01, e os dois campos ganharam descrição.
- A avaliação do notebook 02 decidia o resultado apenas pelo RMSE e
  declarava que a LSTM superava o baseline com um ganho de 0,3%,
  enquanto o modelo perde em MAE (-0,60%) e MAPE (-0,72%) para o
  mesmo random walk. O veredito passou a comparar as três métricas em
  conjunto e a nomear o empate: em 426 dias de validação, diferenças
  abaixo de 1% são ruído.

  Os diagnósticos explicam o resultado: a rede reproduz 22% da
  volatilidade real e prevê alta em 75,6% dos dias contra 53,5% de
  altas de fato, o que derruba a acurácia direcional para 48,7% —
  abaixo do acaso. O modelo não foi retreinado, apenas reavaliado.
- `make lint` falhava com quatro violações pré-existentes em `app/`
  (`UP037`, `UP035` e dois `I001`). Corrigidas: anotação de retorno
  sem aspas em `Settings.from_env`, `Iterable` importado de
  `collections.abc` e ordenação dos blocos de import.

### Removido

- `artifacts/scaler.pkl`, o `MinMaxScaler` sobre preços que o notebook
  01 gravava e nenhuma etapa consumia desde que o alvo passou a ser o
  log-retorno.

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
