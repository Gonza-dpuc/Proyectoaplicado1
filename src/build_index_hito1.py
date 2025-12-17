import os
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import CHUNKS_HITO1_PATH

load_dotenv()

QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "bioactives_hito1"

if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError(
        "ERROR: OPENAI_API_KEY no está configurado. Crea un archivo .env en la raíz.")

client_openai = OpenAI(api_key=OPENAI_API_KEY)


def get_uuid_from_string(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def detect_doc_type(source_str: str) -> str:
    s = (source_str or "").lower()
    if "bioassay" in s or "assay" in s:
        return "bioassay"
    if "pubmed" in s or "abstract" in s:
        return "abstract"
    if "compound" in s or "dictionary" in s:
        return "compound_info"
    if "review" in s or "metabolomics" in s:
        return "literature_review"
    return "unknown"


def build_index_hito1():
    print("[HITO1] Cargando chunks procesados...")
    chunks_path = Path(CHUNKS_HITO1_PATH)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No existe {chunks_path}. Ejecuta: python -m src.chunking")

    df = pd.read_parquet(chunks_path)
    print(f"[HITO1] Chunks cargados: {len(df)}")

    # Columnas esperadas (pero mantenemos robustez)
    text_col = "content" if "content" in df.columns else (
        "text" if "text" in df.columns else None)
    source_col = "source_file" if "source_file" in df.columns else (
        "source" if "source" in df.columns else None)
    if text_col is None or source_col is None:
        raise KeyError(
            f"Columnas inesperadas. Disponibles: {df.columns.tolist()}")

    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if client_qdrant.collection_exists(collection_name=COLLECTION_NAME):
        client_qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print("[HITO1] Colección previa eliminada.")

    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536, distance=models.Distance.COSINE),
    )
    print(f"[HITO1] Colección '{COLLECTION_NAME}' creada.")

    ids = []
    documents_list = []
    payloads = []

    for idx, row in df.iterrows():
        doc_id = row.get("doc_id")
        original_id = row.get("chunk_id", f"chunk_{idx}")

        text = row.get(text_col, "")
        source = row.get(source_col, "")

        chunk_index = row.get("chunk_index", idx)
        total_chunks = row.get("total_chunks", None)

        doc_type = detect_doc_type(str(source))
        point_id = get_uuid_from_string(str(original_id))

        ids.append(point_id)
        documents_list.append(str(text))

        payloads.append({
            "chunk_id": str(original_id),
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chunk_index": int(chunk_index) if chunk_index is not None else 0,
            "total_chunks": int(total_chunks) if total_chunks is not None else None,
            "content": str(text),
            "source_file": os.path.basename(str(source)),
        })

    BATCH_SIZE = 200
    total_docs = len(documents_list)
    print(
        f"[HITO1] Indexando batches de {BATCH_SIZE} (total: {total_docs})...")

    for i in range(0, total_docs, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_docs)
        batch_docs = documents_list[i:batch_end]
        batch_ids = ids[i:batch_end]
        batch_payloads = payloads[i:batch_end]

        emb = client_openai.embeddings.create(
            model=EMBEDDING_MODEL, input=batch_docs)
        vectors = [e.embedding for e in emb.data]

        points = [
            models.PointStruct(
                id=batch_ids[j], vector=vectors[j], payload=batch_payloads[j])
            for j in range(len(batch_docs))
        ]
        client_qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[HITO1] Batch {i}-{batch_end} ✅")

    print(f"[HITO1] ✅ Indexación completada en '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    print("=== [HITO1] Construcción del índice vectorial BioActives (Qdrant) ===")
    build_index_hito1()
    print("=== [HITO1] Proceso completado ===")
