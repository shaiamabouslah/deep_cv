FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/ src/
COPY static/ static/
COPY params.yaml .
COPY models/ models/
COPY data/official/ data/official/
COPY data/monitoring/ data/monitoring/
COPY reports/ reports/

ENV MONITORING_DB=/app/data/monitoring.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
