# Small image for the keyless demo: in-memory BM25 search + offline extractive answers.
# No database, no model downloads, no API key. Works on Render, Hugging Face Spaces, Fly.io, Koyeb...
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DDC_PUBLIC_DEMO=1 DDC_MAX_UPLOAD_MB=10 LLM_PROVIDER=demo

WORKDIR /srv
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install . && useradd --create-home appuser
USER appuser

# Render and Fly set $PORT; Hugging Face Spaces expects 7860.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips='*'"]
