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

# Carpeta donde se guardan los PDFs crudos
RAW_DIR = ROOT_DIR / "data" / "raw"

# Carpeta donde guardamos los chunks .parquet
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Carpeta del índice Chroma (ABSOLUTA, clave para Streamlit!)
CHROMA_DIR = ROOT_DIR / "chroma_db_bioactives"

# Dataset de evaluación
EVAL_SET_PATH = PROCESSED_DIR / "eval_set.json"


# ---------------------------
# Embeddings y modelo
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY no encontrado. Ponlo en .env.")

# Modelo de embeddings (ajústalo a tu necesidad)
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
# Config flags
# ---------------------------

# Para debugging: imprimir rutas usadas
DEBUG_CONFIG = False

if DEBUG_CONFIG:
    print("ROOT_DIR:", ROOT_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    print("CHROMA_DIR:", CHROMA_DIR)
    print("EVAL_SET_PATH:", EVAL_SET_PATH)
