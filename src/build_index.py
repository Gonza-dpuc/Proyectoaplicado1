import pandas as pd
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY
from openai import OpenAI

# Validación de API Key
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError(
        "ERROR: OPENAI_API_KEY no está configurado.\n"
        "Crea un archivo .env en la raíz del proyecto con:\n"
        "OPENAI_API_KEY=tu_clave"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Clasificación de tipo de documento


def detect_doc_type(source_str: str) -> str:
    s = source_str.lower()

    if "bioassay" in s or "assay" in s:
        return "bioassay"
    if "pubmed" in s or "abstract" in s:
        return "abstract"
    if "compound" in s or "dictionary" in s:
        return "compound_info"
    if "review" in s or "metabolomics" in s:
        return "literature_review"

    return "unknown"

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

    # Cliente Chroma
    client_chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=True),
    )

    # Embedding function consistente con retriever.py
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    # Limpiar colección previa
    try:
        client_chroma.delete_collection(name="bioactives_chunks")
        print("Colección previa 'bioactives_chunks' eliminada.")
    except Exception as e:
        print(
            f"No había colección previa o no se pudo eliminar limpiamente: {e}")

    # Crear colección nueva con el MISMO embedding_function que usa retriever.py
    collection = client_chroma.create_collection(
        name="bioactives_chunks",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )
    print("Colección nueva 'bioactives_chunks' creada con embedding_function OpenAI.")

    ids = []
    docs = []
    metas = []

    print("Preparando documentos y metadatos...")

    for idx, row in df.iterrows():
        doc_id = row["doc_id"]
        text = row["text"]
        source = row["source"]
        chunk_index = row.get("chunk_index", idx)

        doc_type = detect_doc_type(str(source))

        unique_id = f"{doc_id}_chunk_{chunk_index}"

        ids.append(unique_id)
        docs.append(text)
        metas.append(
            {
                "doc_id": doc_id,
                "source": source,
                "doc_type": doc_type,
                "chunk_index": int(chunk_index),
            }
        )

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

        # Agregar a Chroma
        collection.add(
            ids=batch_ids,
            embeddings=vectors,
            metadatas=batch_metas,
            documents=batch_docs,
        )

    print("Índice vectorial creado correctamente.")


# Entry point
if __name__ == "__main__":
    print("=== Construcción del índice vectorial BioActives RAG ===")
    build_index()
    print("=== Proceso completado ===")
