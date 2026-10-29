# Shopping Concierge API (+ widget, webhooks). Build: docker build -t concierge .
# Run (data, embeddings, imported stores and the LLM cache stay outside the image):
#   docker run -p 8000:8000 --env-file .env -e CONCIERGE_ALLOWED_ORIGINS=https://mystore.com \
#     -v ./data:/app/data -v ./artifacts:/app/artifacts -v ./stores:/app/stores concierge
# Put an HTTPS reverse proxy (e.g. Caddy, nginx) in front and set app.trust_proxy: true in config.yaml.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the embedding model is downloaded once at build time, not on every start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY config.yaml .
COPY src/ src/
COPY app/ app/

RUN useradd --create-home --uid 1000 concierge && mkdir -p data artifacts stores && chown -R concierge /app
USER concierge

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
