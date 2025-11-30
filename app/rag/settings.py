import os
from pathlib import Path

if os.getenv("ENV", "development") == "development":
    from dotenv import load_dotenv

    load_dotenv()

LLM_MODEL = "gpt-5.1"
TEMPERATURE = 0.5
EMBEDDINGS_MODEL = "text-embedding-3-small"

CHUNK_SIZE = {"en": 350, "es": 460}
CHUNK_OVERLAP = {"en": 50, "es": 60}

_raw_base_dir = os.getenv("BASE_DIR", "app")
BASE_DIR = Path(_raw_base_dir).resolve()
DATA_DIR = os.path.join(BASE_DIR, "docs")
DATABASE_DIR = os.path.join(BASE_DIR, "chroma_db")

API_URL = os.getenv("API_URL", "")
CDN_URL = os.getenv("CDN_URL", "")
