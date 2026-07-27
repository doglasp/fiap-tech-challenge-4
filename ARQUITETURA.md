# Arquitetura

Como o projeto está organizado, o que cada peça faz e por que as
fronteiras estão onde estão. Para instruções de execução, veja o
[README](README.md).

## Princípio que organiza tudo

O projeto separa dois mundos que raramente convivem bem:

- **Offline** — os notebooks treinam e produzem artefatos. Custam
  minutos, exigem TensorFlow e rodam na máquina de quem desenvolve.
- **Online** — a API serve previsões. Custa milissegundos, não conhece
  TensorFlow e roda em contêiner.

A fronteira entre eles são os **artefatos versionados**. O offline só
escreve; o online só lê. Nada além de arquivos atravessa essa linha, e
é por isso que a imagem de produção não carrega o aparato de treino.

```mermaid
flowchart LR
    subgraph OFF["Offline — desenvolvimento"]
        NB["Notebooks 01 e 02<br/>TensorFlow"]
    end

    subgraph ART["Fronteira — artefatos no git"]
        A1["lstm_final.onnx"]
        A2["ret_scaler.pkl"]
        A3["inference_meta.pkl"]
    end

    subgraph ON["Online — produção"]
        API["API FastAPI<br/>ONNX Runtime"]
    end

    NB -->|escreve| ART
    ART -->|lê| API
```

## Visão geral dos componentes

```mermaid
flowchart TB
    CLI["Cliente<br/>navegador, curl, script"]

    subgraph COMPOSE["docker compose"]
        API["api :8000<br/>FastAPI + Uvicorn"]
        PROM["prometheus :9090"]
        GRAF["grafana :3000"]
    end

    subgraph VOL["Artefatos na imagem"]
        MODEL["lstm_final.onnx"]
        SCALER["ret_scaler.pkl"]
        META["inference_meta.pkl"]
    end

    CLI -->|"POST /predict"| API
    CLI -->|"GET /docs"| API
    API --> MODEL
    API --> SCALER
    API --> META

    PROM -->|"raspa /metrics/ a cada 15s"| API
    GRAF -->|"consulta"| PROM
    CLI -->|"abre painéis"| GRAF
```

O Prometheus alcança a API pelo nome de serviço `api:8000`, que só
existe dentro da rede do Compose. Do navegador, o mesmo endpoint é
`localhost:8000` — a separação entre rede interna e externa é
intencional.

## Camadas da aplicação

O pacote `app/` tem 986 linhas divididas em cinco módulos, cada um com
uma responsabilidade única:

| Módulo | Linhas | Responsabilidade |
|---|---|---|
| `main.py` | 393 | rotas HTTP, ciclo de vida, middleware de métricas |
| `metrics.py` | 209 | registro Prometheus e coleta de recursos |
| `predictor.py` | 185 | carga dos artefatos e inferência recursiva |
| `schemas.py` | 161 | contratos de entrada e saída, validação |
| `config.py` | 37 | configuração por variáveis de ambiente |

```mermaid
flowchart TB
    subgraph HTTP["Camada HTTP"]
        MAIN["main.py<br/>rotas e middleware"]
        SCH["schemas.py<br/>validação Pydantic"]
    end

    subgraph DOM["Camada de inferência"]
        PRED["predictor.py<br/>janela, recursão, reconstrução"]
    end

    subgraph INFRA["Infraestrutura"]
        CFG["config.py<br/>variáveis de ambiente"]
        MET["metrics.py<br/>registro Prometheus"]
    end

    MAIN --> SCH
    MAIN --> PRED
    MAIN --> MET
    MAIN --> CFG
    PRED --> CFG
```

A dependência é sempre de fora para dentro: o `predictor.py` não sabe
que existe HTTP, e o `schemas.py` não sabe que existe modelo. É o que
permite os testes trocarem o `Predictor` real por um falso sem tocar em
nada da camada HTTP.

## Fluxo de treino

O notebook 01 é a **única** etapa que transforma dados. O 02 apenas
consome os arrays que ele grava.

```mermaid
flowchart TB
    YF["yfinance<br/>AAPL 2018-2026"]

    subgraph NB1["Notebook 01 — pré-processamento"]
        CLEAN["limpeza<br/>ordenar, deduplicar, faltantes"]
        LOG["log-retorno<br/>r = ln(P_t / P_t-1)"]
        SPLIT["split temporal 80/20<br/>sem embaralhar"]
        SCALE["StandardScaler<br/>ajustado só no treino"]
        WIN["janelas de 60"]
    end

    subgraph NB2["Notebook 02 — modelo"]
        SEARCH["busca de hiperparâmetros"]
        TRAIN["treino final<br/>early stopping"]
        EVAL["avaliação<br/>preço, retorno, horizonte"]
        EXP["exportação ONNX"]
    end

    OUT["models/ e artifacts/"]

    YF --> CLEAN --> LOG --> SPLIT --> SCALE --> WIN
    WIN -->|"X_train, X_valid,<br/>base_valid, true_valid"| SEARCH
    SEARCH --> TRAIN --> EVAL --> EXP --> OUT
    SCALE -->|"ret_scaler.pkl"| OUT
```

Duas fronteiras importam aqui:

**O scaler é ajustado apenas no treino.** Ajustá-lo na série inteira
vazaria a distribuição futura para dentro do conjunto de validação.

**A exportação para ONNX é parte do notebook 02**, não um passo
manual. Sem isso, retreinar deixaria a API servindo o modelo anterior
em silêncio, já que ela lê o `.onnx` e o treino escreve o `.keras`.

## Fluxo de uma previsão

```mermaid
sequenceDiagram
    participant C as Cliente
    participant M as middleware
    participant R as rota /predict
    participant S as PredictRequest
    participant P as Predictor

    C->>M: POST /predict
    M->>M: inicia cronômetro
    M->>S: valida corpo
    alt schema inválido
        S-->>C: 422 (detail = lista)
    end
    S->>R: requisição válida
    R->>P: predict(prices, horizon)
    alt preços insuficientes
        P-->>C: 422 (detail = string)
    end
    P->>P: log-retornos da janela
    P->>P: escalona
    loop horizon vezes
        P->>P: infere próximo retorno
        P->>P: preço = anterior x exp(retorno)
        P->>P: desliza a janela com a previsão
    end
    P-->>R: lista de preços
    R->>M: resposta
    M->>M: registra latência e status
    M-->>C: 200 + previsões
```

O laço é o ponto central: o modelo prevê **um** passo. Horizontes
maiores realimentam a própria previsão, e por isso o erro se acumula —
comportamento medido e documentado no README.

Os dois caminhos de 422 explicam por que o campo `detail` tem formatos
diferentes: a validação de schema acontece no Pydantic, antes da rota;
a validação que depende do modelo carregado acontece dentro do
`Predictor`.

## Ciclo de vida e prontidão

```mermaid
stateDiagram-v2
    [*] --> Subindo
    Subindo --> Carregando: lifespan inicia
    Carregando --> Pronto: artefatos ok + warm_up
    Carregando --> Degradado: exceção no carregamento
    Pronto --> [*]: shutdown
    Degradado --> [*]: shutdown

    note right of Degradado
        /health responde 200 com status degraded
        e startup_error preenchido
        /ready responde 503
        /predict responde 503
    end note
```

A distinção entre `/health` e `/ready` é deliberada: o primeiro
sempre responde, para que a causa da falha seja legível; o segundo
recusa enquanto o modelo não estiver utilizável, para servir de
*readiness probe* a orquestradores.

O `warm_up` executa uma inferência com entrada zerada na
inicialização, para que a primeira requisição real não pague o custo
de aquecimento da sessão.

## Observabilidade

```mermaid
flowchart LR
    subgraph FONTES["Instrumentação na API"]
        MW["middleware HTTP<br/>latência, status, em andamento"]
        INF["inferência<br/>duração"]
        RES["amostrador<br/>CPU e memória a cada 5s"]
        FB["/feedback<br/>erro do modelo"]
    end

    REG["registro Prometheus<br/>exposto em /metrics/"]
    PROM["Prometheus<br/>raspa a cada 15s"]
    ALERT["7 regras de alerta"]
    GRAF["Grafana<br/>11 painéis provisionados"]

    MW --> REG
    INF --> REG
    RES --> REG
    FB --> REG
    REG --> PROM
    PROM --> ALERT
    PROM --> GRAF
```

O `/feedback` é o que distingue esta stack de um monitoramento apenas
de infraestrutura: ele fecha o ciclo entre previsão e realidade,
permitindo medir o **erro do modelo em produção**, e não só o tempo de
resposta.

O amostrador de recursos roda como tarefa assíncrona dentro do ciclo
de vida da aplicação, com intervalo configurável por
`RESOURCE_SAMPLE_SECONDS`.

O caminho `/metrics` é montado como sub-aplicação, então a barra final
importa: `/metrics` redireciona, `/metrics/` responde.

## Empacotamento

```mermaid
flowchart TB
    subgraph IMG["Imagem Docker — 495 MB"]
        PY["python:3.12-slim"]
        DEPS["onnxruntime, fastapi,<br/>uvicorn, scikit-learn"]
        APP["app/"]
        ART["models/ e artifacts/"]
    end

    RUN["Contêiner<br/>UID 1000, porta $PORT<br/>159 MB residentes"]

    PY --> DEPS --> APP --> ART --> RUN
```

Três decisões deixaram a imagem publicável em camadas gratuitas:

**ONNX no lugar do TensorFlow.** O runtime de treino pesava 1,9 GB
para executar uma rede de 66 mil parâmetros. Servir em ONNX levou a
imagem de 1,64 GB para 495 MB e a memória de 722 MB para 159 MB.

**Porta lida de `$PORT`.** As plataformas de hospedagem injetam a
porta em tempo de execução, em vez de aceitar uma fixa.

**Usuário UID 1000.** É o que essas plataformas esperam, e o `COPY`
já entrega os arquivos com o dono correto, evitando uma camada extra
de `chown` recursivo.

O `scikit-learn` permanece na imagem porque o `ret_scaler.pkl` é um
`StandardScaler` serializado — desserializá-lo exige a biblioteca.

## Topologias de execução

```mermaid
flowchart TB
    subgraph LOCAL["Local — docker compose"]
        LA["api"] --- LP["prometheus"] --- LG["grafana"]
    end

    subgraph NUVEM["Produção — Render"]
        RA["api"]
        RN["sem coletor"]
    end

    LOCAL -.->|"mesmo Dockerfile"| NUVEM
```

A mesma imagem serve aos dois cenários. A diferença é o entorno: em
nuvem sobe apenas a API, porque a plataforma executa um contêiner por
serviço e o `docker-compose.yml` não se aplica. O `/metrics/` continua
exposto, mas sem coletor — o monitoramento é demonstrado localmente.

## Contratos de dados

Os artefatos são a interface entre etapas, e cada um tem dono único:

| Artefato | Produzido por | Consumido por | Conteúdo |
|---|---|---|---|
| `AAPL_clean.csv` | notebook 01 | exemplos, testes de carga | série limpa |
| `ret_scaler.pkl` | notebook 01 | notebook 02, API | `StandardScaler` dos log-retornos |
| `inference_meta.pkl` | notebook 02 | API | símbolo, janela, alvo, reconstrução |
| `lstm_final.keras` | notebook 02 | `export_onnx.py` | modelo de origem |
| `lstm_final.onnx` | `export_onnx.py` | **API** | modelo servido |
| `metrics.pkl` | notebook 02 | documentação | avaliação e diagnósticos |

O `Predictor` valida esse contrato ao subir: se a janela declarada nos
metadados não bater com a que o grafo ONNX espera, ele recusa carregar
com uma mensagem que aponta o `export_onnx.py`. Isso impede que um
retreino parcial coloque no ar um modelo incompatível com seus
próprios metadados.

## Decisões estruturais

**Log-retorno como alvo, não preço.** O preço tem tendência: a escala
de 2018 não é a de 2026. A variação relativa é adimensional e
aproximadamente estacionária, o que torna o alvo comparável em todo o
período — e faz a API aceitar a série de qualquer ação.

**Um worker por contêiner, inferência serializada.** Cada processo
carrega uma cópia do modelo, e o cliente Python do Prometheus exige
configuração especial para métricas multiprocesso. A serialização por
`Lock` dá latência previsível; para vazão, replicam-se contêineres
atrás de um balanceador.

**Notebooks autocontidos.** Os notebooks 03 e 04 mantêm cópias
próprias do `Predictor` em vez de importar de `app/`, para que rodem
isoladamente como material didático. O custo é manter as cópias
alinhadas quando a API muda — um trabalho real, já exercido duas
vezes.

**Artefatos versionados apesar do `.gitignore`.** As pastas `models/`
e `artifacts/` são ignoradas para evitar commits acidentais de
execuções locais, mas os artefatos de inferência foram incluídos com
`git add -f`. É o que permite clonar e subir a API sem treinar nada.

## Mapa do repositório

```text
.
├── app/                    aplicação servida em produção
│   ├── config.py             variáveis de ambiente
│   ├── main.py               rotas, ciclo de vida, middleware
│   ├── metrics.py            registro Prometheus
│   ├── predictor.py          inferência recursiva
│   └── schemas.py            contratos de entrada e saída
├── notebooks/              pipeline offline
│   ├── 01_…                  coleta e pré-processamento
│   ├── 02_…                  treino, avaliação e exportação
│   ├── 03_…                  deploy da API
│   └── 04_…                  monitoramento e escalabilidade
├── models/                 modelo de origem e modelo servido
├── artifacts/              scaler, metadados, métricas e série limpa
├── monitoring/             configuração de Prometheus e Grafana
│   ├── alert_rules.yml       7 regras
│   ├── prometheus.yml        alvo de coleta
│   └── grafana/              datasource e dashboard provisionados
├── scripts/
│   ├── export_onnx.py        converte e verifica equivalência
│   ├── eval_horizon.py       erro por horizonte de previsão
│   └── load_test.py          teste de carga concorrente
├── tests/                  suíte com predictor falso
├── apresentacao/           deck e o script que o gera
├── Dockerfile              imagem da API
└── docker-compose.yml      API + Prometheus + Grafana
```
