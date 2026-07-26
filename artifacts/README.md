# Artefatos de inferência

Esta pasta já contém os artefatos versionados no repositório:

- `ret_scaler.pkl` — scaler dos log-retornos, gerado pelo notebook 01
- `inference_meta.pkl` — símbolo, janela e alvo do modelo, notebook 02
- `metrics.pkl` — avaliação do modelo, notebook 02
- `AAPL_clean.csv` — série limpa, notebook 01; usada nos exemplos e no
  teste de carga

O nome do CSV muda conforme o símbolo escolhido.

## Intermediários de treino (não versionados)

O notebook 01 concentra todo o pré-processamento e entrega ao 02 os
arrays abaixo. Eles são regerados ao executar o 01 e ficam de fora do
git por serem grandes e derivados:

| Arquivo | Conteúdo |
|---|---|
| `X_train.npy` / `y_train.npy` | janelas de treino e o log-retorno escalonado alvo |
| `X_valid.npy` / `y_valid.npy` | idem para a validação |
| `base_valid.npy` | `P_{k-1}` de cada janela de validação |
| `true_valid.npy` | `P_k` realizado, para avaliar em escala de preço |
| `valid_dates.npy` | datas da validação, para os gráficos |
| `meta.pkl` | símbolo, período, janela, `train_ratio`, alvo e reconstrução |

`base_valid` e `true_valid` existem porque `y` está em retorno
escalonado: sem os preços de referência não dá para reconstruir
`P̂ = P_{k-1} · exp(r̂)` nem comparar com o valor realizado.

## Conteúdo do `metrics.pkl`

| Chave | Descrição |
|---|---|
| `lstm` | MAE, RMSE e MAPE do modelo, em escala de preço |
| `naive_baseline` | as mesmas métricas para o random walk (retorno zero) |
| `ganho_vs_naive_%` | ganho percentual da LSTM sobre o baseline, por métrica; negativo = pior |
| `return_diagnostics` | acurácia direcional, R² contra prever zero, correlação, razão de desvios e proporção de altas |
| `n_validacao` | número de dias avaliados |
| `horizon_metrics` | erro da previsão recursiva por horizonte, indexado por `"1"`, `"2"`, `"3"`, `"5"`, `"10"` e `"20"` |
| `best_hyperparams` | `units`, `dropout` e `lr` da configuração vencedora |

Os diagnósticos de retorno existem porque MAE/RMSE/MAPE em preço são
dominados por P_{k-1}: mesmo prevendo retorno zero o erro fica em torno
de 1%, o que faz qualquer modelo parecer bom. As chaves do
`return_diagnostics` medem o que a rede de fato prevê.

O `horizon_metrics` existe porque a API aceita `horizon` até 30 e
produz esses passos por recursão, enquanto `lstm` e `naive_baseline`
cobrem apenas D+1. Cada horizonte traz `n`, `MAE`, `RMSE`, `MAPE`,
`MAE_naive`, `MAPE_naive` e o par `variacao_prevista_%` /
`variacao_real_%`, que compara o movimento projetado com o observado
e quantifica o achatamento. O horizonte `"1"` reproduz os valores de
`lstm`, o que serve de verificação. A seção 7.4 do notebook 02 gera
essas chaves; `scripts/eval_horizon.py` recalcula tudo a partir do CSV
versionado, sem depender dos intermediários.
