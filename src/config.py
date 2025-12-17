from pathlib import Path
import os
from dotenv import load_dotenv

# Carga variables de entorno (OPENAI_API_KEY, etc.)
load_dotenv()

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------
# Paths (todas ABSOLUTAS)
# ---------------------------

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Corpus intermedio (salida de tu ingesta / parsing)
CORPUS_PATH = PROCESSED_DIR / "corpus.parquet"

# Chunks separados por hito (Opción A)
CHUNKS_HITO1_PATH = PROCESSED_DIR / "chunks_hito1.parquet"
CHUNKS_HITO2_PATH = PROCESSED_DIR / "chunks_hito2.parquet"

# (Backwards compat si alguna parte antigua usa esto)
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"

# Dataset de evaluación (tu app puede buscar aquí)
EVAL_SET_PATH = PROCESSED_DIR / "eval_set.json"

# Carpeta del índice Chroma (si la usas)
CHROMA_DIR = ROOT_DIR / "chroma_db_bioactives"

# ---------------------------
# Embeddings y modelo
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY no encontrado. Ponlo en .env.")

EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------
# Qdrant
# ---------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    print("[WARN] QDRANT_URL no encontrado. Asegúrate de definirlo en .env")
if not QDRANT_API_KEY:
    print("[INFO] QDRANT_API_KEY no encontrado (puede ser opcional si usas Qdrant local sin auth).")

# ---------------------------
# Debug
# ---------------------------
DEBUG_CONFIG = False
if DEBUG_CONFIG:
    print("ROOT_DIR:", ROOT_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("CORPUS_PATH:", CORPUS_PATH)
    print("CHUNKS_HITO1_PATH:", CHUNKS_HITO1_PATH)
    print("CHUNKS_HITO2_PATH:", CHUNKS_HITO2_PATH)
    print("EVAL_SET_PATH:", EVAL_SET_PATH)
