# IsipFin — one image, one service (docs/adr/ADR-008-packaging-docker-ci.md).
#
# Two stages so the runtime image carries no toolchain: `builder` compiles a venv from the
# pinned requirements; `runtime` copies that venv and only the files the app actually reads.
# Runtime state (FAISS/BM25 artifacts, uploaded PDFs, chat.sqlite3) lives under `/app/data`,
# which compose mounts as a named volume — a rebuild never loses ingested docs or history.

# --- builder ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Every pin ships manylinux wheels, but a wheel-less platform falls back to a source build
# (reportlab and faiss-cpu both have C extensions). The toolchain stays in this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# --- runtime ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONPATH=/app gives the same `import src.*` resolution the native editable install
# provides, for `streamlit run app.py`, `python scripts/build_index.py`, and pytest alike.
# DATA_DIR is the runtime state root (src/store/db.py); the other paths are resolved from
# the module, so they land inside the mounted /app/data by themselves.
# FASTEMBED_CACHE_PATH keeps the local-embedding weights on the mounted /app/data volume.
# Without it fastembed defaults to $TMPDIR/fastembed_cache (fastembed/common/utils.py:
# `define_cache_dir`), i.e. /tmp inside the container: the ~150 MB BAAI/bge-small-en-v1.5
# fetch would be repeated on every `down`/rebuild, because /tmp is not the volume. With it,
# the download happens once per data volume and later ingests/queries read it offline.
# MOCK_LLM=1 never builds a fastembed model, so that keyless path downloads nothing.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    DATA_DIR=/app/data \
    FASTEMBED_CACHE_PATH=/app/data/models \
    LOG_JSON=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Only what the app reads. `docs/prompts/` is loaded at runtime by `load_prompt`, and the
# upload path imports `scripts/build_index.py`; `tests/` ships so `docker compose exec app
# pytest -q` and `scripts/eval_rag.py` (which reads tests/data/golden_qa.jsonl) work too.
# `.env` and `data/` are deliberately absent: secrets never enter the image, and state
# belongs in the volume.
COPY app.py pyproject.toml ./
COPY .streamlit/ ./.streamlit/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY docs/ ./docs/

# Drop host build caches that the context may carry, create the state dir, and run unrooted.
# The named volume inherits this directory's ownership on first mount.
RUN find /app -type d -name __pycache__ -prune -exec rm -rf {} + \
    && mkdir -p /app/data \
    && groupadd --gid 10001 appuser \
    && useradd --create-home --uid 10001 --gid appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Streamlit's own health endpoint. ProxyHandler({}) keeps a proxy env var out of the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request as u; o = u.build_opener(u.ProxyHandler({})); sys.exit(0 if o.open('http://127.0.0.1:8501/_stcore/health', timeout=4).read().strip() == b'ok' else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
