import os
import re
import uuid
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import CHUNKS_HITO2_PATH

load_dotenv()

QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "bioactives_hito2"

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


def extract_chemical_metadata(text: str):
    meta = {}
    if not isinstance(text, str) or not text:
        return meta

    mz_match = re.search(
        r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if mz_match:
        try:
            meta["mz"] = float(mz_match.group(1))
        except ValueError:
            pass

    rt_match = re.search(
        r"(?:rt|retention time)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if rt_match:
        try:
            meta["rt"] = float(rt_match.group(1))
        except ValueError:
            pass

    return meta


def build_index():
    print("[HITO2] Cargando chunks procesados...")
    chunks_path = Path(CHUNKS_HITO2_PATH)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No existe {chunks_path}. Ejecuta: python -m src.chunking")

    df = pd.read_parquet(chunks_path)
    print(f"[HITO2] Chunks cargados: {len(df)}")

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
        print(f"[HITO2] Colección anterior '{COLLECTION_NAME}' eliminada.")

    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536, distance=models.Distance.COSINE),
    )
    print(f"[HITO2] Colección '{COLLECTION_NAME}' creada.")

    # índices payload para filtros numéricos (self-query)
    for field in ("mz", "rt"):
        try:
            client_qdrant.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.FLOAT,
            )
        except Exception:
            pass

    ids = []
    documents_list = []
    payloads = []

    print("[HITO2] Preparando documentos y metadatos...")

    for idx, row in df.iterrows():
        doc_id = row.get("doc_id")
        original_id = row.get("chunk_id", f"chunk_{idx}")

        text = row.get(text_col, "")
        source = row.get(source_col, "")

        chunk_index = row.get("chunk_index", idx)
        total_chunks = row.get("total_chunks", None)

        doc_type = detect_doc_type(str(source))
        point_id = get_uuid_from_string(str(original_id))

        chem_meta = extract_chemical_metadata(str(text))

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
            "mz": chem_meta.get("mz"),
            "rt": chem_meta.get("rt"),
        })

    BATCH_SIZE = 200
    total_docs = len(documents_list)
    print(
        f"[HITO2] Indexando batches de {BATCH_SIZE} (total: {total_docs})...")

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
        print(f"[HITO2] Batch {i}-{batch_end} ✅")

    print(f"[HITO2] ✅ Indexación completada en '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    print("=== [HITO2] Construcción del índice vectorial BioActives (Qdrant) ===")
    build_index()
    print("=== [HITO2] Proceso completado ===")
