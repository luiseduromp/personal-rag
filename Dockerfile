FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHROMA_DB_PATH=/chroma-db \
    DOCS_PATH=/rag-docs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY . .

RUN mkdir -p ${CHROMA_DB_PATH} ${DOCS_PATH}

RUN ln -sf ${CHROMA_DB_PATH} /app/app/chroma_db && \
    ln -sf ${DOCS_PATH} /app/app/docs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]