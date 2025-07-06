import os
from pathlib import Path

if os.getenv("ENV", "development") == "development":
    from dotenv import load_dotenv

    load_dotenv()

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5
EMBEDDINGS_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_raw_base_dir = os.getenv("BASE_DIR", "app")
BASE_DIR = Path(_raw_base_dir).resolve()
DATA_DIR = os.path.join(BASE_DIR, "docs")
DATABASE_DIR = os.path.join(BASE_DIR, "chroma_db")
