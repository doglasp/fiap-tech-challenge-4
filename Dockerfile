FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# UID 1000 é o que as plataformas de hospedagem costumam usar ao rodar o
# contêiner; criar o usuário antes dos COPY evita um chown recursivo, que
# duplicaria os arquivos numa nova camada.
RUN useradd --create-home --uid 1000 apiuser

WORKDIR /opt/lstm-api

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY --chown=apiuser:apiuser app ./app
COPY --chown=apiuser:apiuser models ./models
COPY --chown=apiuser:apiuser artifacts ./artifacts

USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/ready', timeout=4)"

# Forma shell para expandir $PORT, que as plataformas de deploy injetam;
# o exec entrega os sinais ao uvicorn em vez de deixá-los no shell.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
