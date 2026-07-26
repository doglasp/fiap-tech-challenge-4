# Tech Challenge Fase 4 — LSTM para previsão de ações

Projeto de Machine Learning Engineering que cobre a coleta e o
pré-processamento dos preços históricos, o treinamento de uma rede
LSTM, a exportação dos artefatos, o deploy em FastAPI e o
monitoramento com Prometheus e Grafana.

A implementação foi preparada para o modelo treinado nos notebooks,
que prevê **log-retornos** e reconstrói o preço pela fórmula:

```text
preço_previsto = preço_anterior × exp(retorno_previsto)
```

## Estrutura

```text
.
├── app/
│   ├── config.py
│   ├── main.py
│   ├── metrics.py
│   ├── predictor.py
│   └── schemas.py
├── artifacts/
├── models/
├── monitoring/
│   ├── alert_rules.yml
│   ├── prometheus.yml
│   └── grafana/
├── notebooks/
├── scripts/load_test.py
├── tests/test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## 1. Modelo e artefatos

Os arquivos abaixo já vêm versionados no repositório, então basta
clonar para executar a API:

```text
models/
├── lstm_final.keras   ← treinado pelo notebook 02
└── lstm_final.onnx    ← servido pela API

artifacts/
├── ret_scaler.pkl
├── inference_meta.pkl
├── metrics.pkl
└── AAPL_clean.csv
```

A API carrega o **ONNX**, não o `.keras`. O `.keras` é o artefato de
origem, produzido pelo treino; o `.onnx` é derivado dele e serve à
inferência, o que tira o TensorFlow da imagem de produção:

| | TensorFlow (`.keras`) | ONNX (`.onnx`) |
|---|---|---|
| Imagem Docker | 1,64 GB | **495 MB** |
| Memória residente | 722 MB | **159 MB** |
| Artefato | 818 KB | 268 KB |

O `metrics.pkl` guarda a avaliação do modelo — métricas de preço,
comparação com o baseline naïve e diagnósticos no espaço de retorno,
descritos em [artifacts/README.md](artifacts/README.md) — e o CSV
serve aos exemplos e ao teste de carga.

Para regerá-los do zero, execute os notebooks 01 e 02 **nessa ordem**:
o 01 concentra todo o pré-processamento e grava os arrays de treino
que o 02 consome, sem repetir nenhuma transformação. Esses arrays
intermediários não são versionados, então o 02 não roda sozinho num
clone novo.

Depois de retreinar, **reexporte o ONNX**, senão a API continua
servindo o modelo antigo:

```bash
pip install -r requirements-dev.txt
python scripts/export_onnx.py
```

O script converte o `.keras` e verifica a equivalência em 512 janelas
aleatórias, abortando se a diferença passar de `1e-4`. Na versão atual
a diferença máxima é de `3.7e-07`.

## 2. Execução local sem Docker

Requer Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

uvicorn app.main:app       --host 0.0.0.0       --port 8000
```

No Windows PowerShell, ative o ambiente com:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Execução completa com Docker Compose

Crie o arquivo de ambiente:

```bash
cp .env.example .env
```

Troque a senha do Grafana e inicie a stack:

```bash
docker compose up --build -d
docker compose ps
```

Serviços:

| Serviço | Endereço |
|---|---|
| API | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| Métricas | `http://localhost:8000/metrics/` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

O dashboard **Tech Challenge — API LSTM** é provisionado
automaticamente no Grafana.

Para acompanhar a inicialização:

```bash
docker compose logs -f api
```

## 4. Endpoints

| Método | Rota | Função |
|---|---|---|
| `GET` | `/` | índice com links para os demais recursos |
| `GET` | `/health` | estado da aplicação e uso de recursos; sempre HTTP 200 |
| `GET` | `/ready` | prontidão do modelo; HTTP 503 enquanto não carregou |
| `POST` | `/predict` | previsão de preços a partir de fechamentos |
| `POST` | `/feedback` | registra o preço observado para medir o erro |
| `GET` | `/metrics/` | métricas no formato Prometheus |
| `GET` | `/docs` | documentação interativa (Swagger) |

A barra final de `/metrics/` importa: o endpoint é montado como uma
sub-aplicação, então `/metrics` sem barra apenas redireciona.

### Índice

```bash
curl http://localhost:8000/
```

```json
{
  "name": "Tech Challenge Fase 4 — API de Previsão LSTM",
  "version": "2.0.0",
  "docs": "/docs",
  "health": "/health",
  "readiness": "/ready",
  "metrics": "/metrics/"
}
```

### Saúde

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_ready": true,
  "model_trained_on": "AAPL",
  "window_size": 60,
  "min_prices": 61,
  "resources": {
    "system_cpu_percent": 0.0,
    "system_memory_percent": 51.8,
    "process_cpu_percent": 0.2,
    "process_cpu_normalized_percent": 0.025,
    "process_rss_mb": 158.9
  },
  "startup_error": null
}
```

O `/health` responde HTTP 200 mesmo com o modelo indisponível: nesse
caso `status` vira `degraded`, `model_ready` vira `false` e
**`startup_error` traz a exceção que impediu o carregamento** — é o
primeiro lugar a olhar quando a API sobe sem servir previsões.

O `/ready` é o oposto: só responde 200 quando o modelo está pronto, e
HTTP 503 enquanto não estiver. Use-o como *readiness probe*.

```bash
curl http://localhost:8000/ready
```

```json
{
  "status": "ready",
  "model_loaded": true,
  "scaler_loaded": true,
  "model_trained_on": "AAPL",
  "window_size": 60,
  "min_prices": 61
}
```

### Previsão

A quantidade mínima de preços é informada por `/health` e `/ready` no
campo `min_prices`. No modelo atual a janela é 60, portanto são
necessários pelo menos 61 fechamentos.

O comando abaixo monta o corpo com os 61 últimos fechamentos do CSV
versionado, evitando montar a lista à mão. Rode-o a partir da raiz do
projeto, com o ambiente da seção 2 ativado — fora dele, troque
`python` por `python3`:

```bash
python -c "import csv,json;r=list(csv.DictReader(open('artifacts/AAPL_clean.csv')));print(json.dumps({'prices':[round(float(x['Close']),2) for x in r[-61:]],'horizon':1,'symbol':'AAPL'}))" > /tmp/payload.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @/tmp/payload.json
```

O Swagger também traz um exemplo completo já preenchido, bastando
clicar em *Try it out*.

Resposta:

```json
{
  "symbol": "AAPL",
  "model_trained_on": "AAPL",
  "last_price": 298.01,
  "horizon": 1,
  "predictions": [297.71]
}
```

### Erros de validação

O `/predict` recusa entradas inválidas com HTTP 422, mas **o campo
`detail` tem dois formatos diferentes**, conforme onde a validação
falhou. Quem consome a API programaticamente precisa tratar os dois.

Regras de schema — tamanho, sinal e faixa — são verificadas pelo
Pydantic, e o `detail` vem como **lista de objetos**:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "horizon"],
      "msg": "Input should be less than or equal to 30",
      "input": 99
    }
  ]
}
```

Regras que dependem do modelo carregado — como a quantidade mínima de
preços, que varia com a janela — são verificadas pelo `Predictor`, e o
`detail` vem como **string**:

```json
{
  "detail": "São necessários pelo menos 61 preços (janela=60); recebidos 3."
}
```

Um cliente robusto deve checar o tipo antes de exibir a mensagem:

```python
detalhe = resposta.json()["detail"]
if isinstance(detalhe, list):
    mensagens = [erro["msg"] for erro in detalhe]
else:
    mensagens = [detalhe]
```

### Sobre os campos `symbol` e `model_trained_on`

O modelo consome **log-retornos**, que são adimensionais, portanto a
API aceita a série de qualquer ação — não há restrição a um conjunto
predefinido de papéis.

Por isso a resposta separa duas informações que não devem ser
confundidas:

- `symbol` é a ação que a requisição diz representar. É opcional e
  meramente informativo; quando omitido, volta nulo, pois a API não
  tem como deduzir a origem dos preços.
- `model_trained_on` é a ação em que o modelo carregado foi treinado.

Quando os dois divergem, o modelo está sendo aplicado a uma ação
diferente da que viu no treino. A previsão é produzida mesmo assim,
mas a divergência fica explícita na resposta e deve ser levada em
conta na interpretação do resultado.

### Sobre o campo `horizon`

O modelo prevê **um** passo à frente. Horizontes maiores são obtidos
por recursão: a previsão de D+1 entra na janela como se fosse um dia
observado, e o processo se repete. O erro, portanto, se acumula.

Medido sobre os mesmos 426 dias de validação do notebook 02:

| Horizonte | MAPE | MAPE do naïve | Variação prevista | Variação real |
|---|---|---|---|---|
| D+1 | 1,19% | 1,19% | 0,21% | 1,19% |
| D+2 | 1,78% | 1,78% | 0,36% | 1,78% |
| D+3 | 2,27% | 2,29% | 0,49% | 2,28% |
| D+5 | 3,09% | 3,13% | 0,71% | 3,13% |
| D+10 | 4,20% | 4,23% | 1,32% | 4,25% |
| D+20 | 5,89% | 5,93% | 2,68% | 6,03% |

Duas leituras importam:

- **O erro cresce aproximadamente com a raiz do horizonte**, o padrão
  de um passeio aleatório, e o modelo não supera o baseline naïve em
  nenhum horizonte.
- **As duas últimas colunas quantificam o achatamento.** Em D+5 o
  modelo projeta uma variação média de 0,71% enquanto o preço varia
  3,13% de fato. A projeção longa é uma deriva suave, não uma
  previsão da dinâmica do papel.

Ou seja, `horizon` alto é aceito pela API, mas o resultado deve ser
lido como tendência achatada, não como preço esperado. Os números
acima são reproduzíveis:

```bash
python scripts/eval_horizon.py
```

O horizonte 1 do script reproduz o MAE, o RMSE e o MAPE gravados no
`artifacts/metrics.pkl`, o que verifica a replicação do split.

### Feedback da previsão

Quando o preço real estiver disponível, registre-o para acompanhar
o erro do modelo:

```bash
curl -X POST http://localhost:8000/feedback       -H "Content-Type: application/json"       -d '{
    "predicted_price": 297.71,
    "actual_price": 298.01
  }'
```

## 5. Monitoramento

A API expõe, entre outras, as métricas:

- `lstm_api_http_request_duration_seconds`;
- `lstm_api_http_requests_total`;
- `lstm_model_inference_duration_seconds`;
- `lstm_api_process_cpu_normalized_percent`;
- `lstm_api_process_resident_memory_bytes`;
- `lstm_model_absolute_percentage_error`;
- `lstm_model_ready`.

O Prometheus avalia alertas para:

- indisponibilidade da API;
- falha de readiness do modelo;
- latência p95 acima de 500 ms;
- taxa de erros 5xx acima de 1%;
- CPU e memória elevadas;
- erro preditivo p95 acima de 5%.

As regras ficam visíveis no Prometheus. Para envio de notificações,
conecte um Alertmanager.

## 6. Testes automatizados

Os testes usam um predictor falso, portanto não precisam carregar o
runtime de inferência nem o arquivo do modelo.

```bash
pip install -r requirements-dev.txt
pytest
```

## 7. Teste de carga

Com a API em execução:

```bash
python scripts/load_test.py       --prices-file artifacts/AAPL_clean.csv       --requests 100       --concurrency 10       --horizon 1
```

O script apresenta throughput, taxa de erro e latências média, p50,
p95, p99 e máxima.

## 8. Escalabilidade

O contêiner usa um único worker do Uvicorn porque:

- cada processo mantém uma cópia do modelo em memória;
- o cliente Python do Prometheus exige configuração especial para
  métricas multiprocess;
- a inferência é serializada por processo, para que a sessão ONNX não
  receba chamadas concorrentes.

Para aumentar capacidade, prefira réplicas do contêiner atrás de um
balanceador de carga. Ajuste `ORT_NUM_INTRAOP_THREADS` e
`ORT_NUM_INTEROP_THREADS` com base nos testes do ambiente.

Medido nesta máquina com o `load_test.py`, sem erros em nenhum caso:

| Cenário | Throughput | p95 |
|---|---|---|
| 200 req, concorrência 10, D+1 | 236 req/s | 76 ms |
| 60 req, concorrência 10, D+30 | 58 req/s | 188 ms |

Ambos ficam abaixo do limiar de 0,5 s da regra `LSTMApiHighP95Latency`.

## 8.1 Publicar em nuvem

O contêiner lê a porta de `$PORT` (padrão 8000) e roda como UID 1000,
que é o esperado pelas plataformas de hospedagem. Com 495 MB de imagem
e 159 MB de memória residente, cabe nas camadas gratuitas que limitam
o contêiner a 512 MB.

```bash
docker run -e PORT=7860 -p 7860:7860 lstm-api:onnx
```

Só a API sobe: Prometheus e Grafana continuam no `docker-compose.yml`
local, então o `/metrics` fica exposto mas sem coletor em nuvem.

## 9. Checklist da entrega

- notebooks 01 a 04 executados;
- modelo e artefatos gerados;
- API respondendo em `/predict`;
- imagem Docker construída;
- Prometheus coletando `api:8000`;
- dashboard carregado no Grafana;
- teste de carga registrado;
- README publicado no repositório;
- vídeo demonstrando API, métricas e dashboard;
- link da API, caso seja publicada em nuvem.

## Observação sobre os artefatos

O modelo e os artefatos de inferência **já estão versionados** neste
repositório, portanto a API funciona logo após o clone, sem executar
os notebooks.

O `.gitignore` continua ignorando `models/*` e `artifacts/*` para
evitar commits acidentais de execuções locais. Ao regerar os
artefatos e desejar publicá-los, force a inclusão:

```bash
git add -f models/lstm_final.keras models/lstm_final.onnx artifacts/*.pkl artifacts/*.csv
```

Os arquivos atuais são pequenos (o `.keras` tem cerca de 800 KB e o
`.onnx`, 270 KB) e cabem no Git comum. Para modelos maiores, prefira
Git LFS, uma release do GitHub ou um armazenamento de objetos.
