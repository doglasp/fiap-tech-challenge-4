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

## 1. Gerar o modelo e os artefatos

Execute os notebooks 01 e 02 na ordem. Ao final, a raiz do projeto
precisa conter:

```text
models/
└── lstm_final.keras

artifacts/
├── ret_scaler.pkl
├── inference_meta.pkl
└── AAPL_clean.csv
```

O CSV é necessário apenas para os exemplos e o teste de carga. A
API utiliza o modelo, o scaler e os metadados.

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

### Saúde

```bash
curl http://localhost:8000/health
```

O endpoint pode retornar `degraded` quando algum artefato estiver
ausente. O `/ready` retorna HTTP 503 até que o modelo esteja pronto.

```bash
curl http://localhost:8000/ready
```

### Previsão

A quantidade mínima de preços é informada por `/health` e
`/ready`. No modelo atual, a janela é 60, portanto são necessários
pelo menos 61 fechamentos.

```bash
curl -X POST http://localhost:8000/predict       -H "Content-Type: application/json"       -d '{
    "prices": [
      270.10, 271.30, 270.85, 272.12, 273.04,
      272.80, 274.15, 275.01, 274.60, 276.20
    ],
    "horizon": 1
  }'
```

O exemplo acima é ilustrativo e precisa ser ampliado até a
quantidade mínima exigida pelo modelo.

Resposta:

```json
{
  "symbol": "AAPL",
  "last_price": 298.01,
  "horizon": 1,
  "predictions": [297.71]
}
```

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
TensorFlow nem o arquivo `.keras`.

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
- a inferência é serializada por processo para evitar concorrência
  imprevisível no TensorFlow.

Para aumentar capacidade, prefira réplicas do contêiner atrás de um
balanceador de carga. Ajuste `TF_NUM_INTRAOP_THREADS` e
`TF_NUM_INTEROP_THREADS` com base nos testes do ambiente.

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

O `.gitignore` não versiona automaticamente os arquivos de modelo e
os artefatos binários. Para incluí-los no repositório, utilize Git
LFS, uma release do Git ou um armazenamento de objetos.
