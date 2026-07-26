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
