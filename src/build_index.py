import sys
import os

# Agregar la raíz del proyecto al sys.path para permitir importaciones absolutas (from src...)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
from openai import OpenAI
from src.schemas import ChunkMetadata

# Validación de API Key
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError(
        "ERROR: OPENAI_API_KEY no está configurado.\n"
        "Crea un archivo .env en la raíz del proyecto con:\n"
        "OPENAI_API_KEY=tu_clave"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------------------------------------------------
# Factory Pattern para creación de Metadatos
# -------------------------------------------------------------------------
class ChunkFactory:
    @staticmethod
    def _detect_doc_type(source_str: str) -> str:
        s = source_str.lower()
        if "bioassay" in s or "assay" in s: return "bioassay"
        if "pubmed" in s or "abstract" in s: return "abstract"
        if "compound" in s or "dictionary" in s: return "compound_info"
        if "review" in s or "metabolomics" in s: return "literature_review"
        return "unknown"

    @staticmethod
    def create_metadata(row, default_chunk_index: int) -> ChunkMetadata:
        source = str(row["source"])
        doc_type = ChunkFactory._detect_doc_type(source)
        return ChunkMetadata(
            doc_id=str(row["doc_id"]),
            source=source,
            doc_type=doc_type,
            chunk_index=int(row.get("chunk_index", default_chunk_index))
        )

# Construcción del índice


def build_index():

    print("Cargando chunks procesados...")
    chunks_path = Path("data/processed/chunks.parquet")
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"No existe {chunks_path}. Primero ejecuta:\n"
            "    python src/chunking.py"
        )

    df = pd.read_parquet(chunks_path)
    print(f"Chunks cargados: {len(df)}")

    # Cliente Qdrant
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Obtener dimensión del embedding para configurar la colección
    print("Verificando dimensión del embedding...")
    dummy_emb = client.embeddings.create(input="test", model=EMBEDDING_MODEL).data[0].embedding
    vector_size = len(dummy_emb)

    # Recrear colección en Qdrant
    qdrant_client.recreate_collection(
        collection_name="bioactives_chunks",
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    print(f"Colección 'bioactives_chunks' recreada en Qdrant (size={vector_size}).")

    ids = []
    docs = []
    metas = []

    print("Preparando documentos y metadatos...")

    for idx, row in df.iterrows():
        doc_id = row["doc_id"]
        text = row["text"]
        source = row["source"]
        chunk_index = row.get("chunk_index", idx)
        unique_id = f"{doc_id}_chunk_{chunk_index}"

        # Usamos la Factory para crear y validar metadatos
        metadata_obj = ChunkFactory.create_metadata(row, idx)

        ids.append(unique_id)
        docs.append(text)
        # Convertimos a dict para Qdrant, pero ya sabemos que es válido gracias a Pydantic
        metas.append(metadata_obj.model_dump())

    # EMBEDDINGS EN BATCHES
    BATCH_SIZE = 200  # ajustable: 100, 200, 500

    print(f"Generando embeddings en batches de {BATCH_SIZE}...")

    for i in range(0, len(docs), BATCH_SIZE):
        batch_docs = docs[i: i + BATCH_SIZE]
        batch_ids = ids[i: i + BATCH_SIZE]
        batch_metas = metas[i: i + BATCH_SIZE]

        print(f"Procesando batch {i} – {i + len(batch_docs)} / {len(docs)}")

        # Crear los embeddings del batch con OpenAI
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_docs,
        )

        vectors = [emb.embedding for emb in response.data]

        # Preparar puntos para Qdrant
        points = []
        for j, vec in enumerate(vectors):
            # Generar UUID determinista basado en el ID original
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, batch_ids[j]))
            payload = batch_metas[j].copy()
            payload["text"] = batch_docs[j]  # Guardar texto en payload

            points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))

        # Subir a Qdrant
        qdrant_client.upsert(
            collection_name="bioactives_chunks",
            points=points
        )

    print("Índice vectorial creado correctamente.")


# Entry point
if __name__ == "__main__":
    print("=== Construcción del índice vectorial BioActives RAG ===")
    build_index()
    print("=== Proceso completado ===")
