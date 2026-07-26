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
└── lstm_final.keras

artifacts/
├── ret_scaler.pkl
├── inference_meta.pkl
├── metrics.pkl
└── AAPL_clean.csv
```

A API utiliza o modelo, o scaler e os metadados. O `metrics.pkl`
guarda a avaliação do modelo — métricas de preço, comparação com o
baseline naïve e diagnósticos no espaço de retorno, descritos em
[artifacts/README.md](artifacts/README.md) — e o CSV serve aos
exemplos e ao teste de carga.

Para regerá-los do zero, execute os notebooks 01 e 02 **nessa ordem**:
o 01 concentra todo o pré-processamento e grava os arrays de treino
que o 02 consome, sem repetir nenhuma transformação. Esses arrays
intermediários não são versionados, então o 02 não roda sozinho num
clone novo.

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
    "horizon": 1,
    "symbol": "AAPL"
  }'
```

O exemplo acima é ilustrativo e precisa ser ampliado até a
quantidade mínima exigida pelo modelo. O Swagger já traz um exemplo
completo, com os 61 fechamentos.

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

O modelo e os artefatos de inferência **já estão versionados** neste
repositório, portanto a API funciona logo após o clone, sem executar
os notebooks.

O `.gitignore` continua ignorando `models/*` e `artifacts/*` para
evitar commits acidentais de execuções locais. Ao regerar os
artefatos e desejar publicá-los, force a inclusão:

```bash
git add -f models/lstm_final.keras artifacts/*.pkl artifacts/*.csv
```

Os arquivos atuais são pequenos (o `.keras` tem cerca de 800 KB) e
cabem no Git comum. Para modelos maiores, prefira Git LFS, uma
release do GitHub ou um armazenamento de objetos.
