FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

# hdbscan builds from source on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU torch first, so sentence-transformers doesn't pull the multi-GB CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# bake the embedder and cross-encoder in, so containers work offline and start fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY pyproject.toml alembic.ini ./
COPY migrations migrations
COPY src src
COPY scripts scripts
COPY data/validation data/validation
RUN pip install --no-deps -e . && python -m src.data.synthetic_kb_builder

ENTRYPOINT ["umbra"]
CMD ["--help"]
