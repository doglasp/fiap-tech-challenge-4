FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         PIP_NO_CACHE_DIR=1         TF_CPP_MIN_LOG_LEVEL=2         MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /opt/lstm-api

COPY requirements.txt .
RUN python -m pip install --upgrade pip         && python -m pip install -r requirements.txt         && rm -rf /usr/local/lib/python3.12/site-packages/tensorflow/include

COPY app ./app
COPY models ./models
COPY artifacts ./artifacts

RUN useradd --create-home --uid 10001 apiuser         && chown -R apiuser:apiuser /opt/lstm-api

USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3       CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=4)"

CMD ["uvicorn", "app.main:app",          "--host", "0.0.0.0",          "--port", "8000",          "--workers", "1"]
