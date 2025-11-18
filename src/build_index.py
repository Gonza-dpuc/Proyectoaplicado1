import pandas as pd
from pathlib import Path
import chromadb
from chromadb.config import Settings
from config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_API_KEY
from openai import OpenAI


# Validación de API Key
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError(
        "ERROR: OPENAI_API_KEY no está configurado.\n"
        "Crea un archivo .env en la raíz del proyecto y define:\n"
        "OPENAI_API_KEY=tu_clave_aquí"
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# Detectar tipo de documento
def detect_doc_type(source_str: str) -> str:
    """
    Clasifica el documento según el nombre del archivo o la ruta.
    Esto permite futuros filtros basados en tipo de evidencia.
    """

    source_lower = source_str.lower()

    if "bioassay" in source_lower or "assay" in source_lower:
        return "bioassay"

    if "pubmed" in source_lower or "abstract" in source_lower:
        return "abstract"

    if "compound" in source_lower or "dictionary" in source_lower:
        return "compound_info"

    if "review" in source_lower or "metabolomics" in source_lower:
        return "literature_review"

    return "unknown"


# Función principal de construcción del índice
def build_index():
    print("Cargando chunks procesados...")

    chunks_path = Path("data/processed/chunks.parquet")
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"ERROR: No existe el archivo {chunks_path}.\n"
            "Primero ejecuta:\n"
            "    python src/chunking.py"
        )

    df = pd.read_parquet(chunks_path)
    print(f"Chunks cargados: {len(df)}")

    # Crear cliente Chroma DB
    client_chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=True)
    )

    collection = client_chroma.get_or_create_collection(
        name="bioactives_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    # Limpiar colección (evitar duplicados)
    print("Limpiando colección existente para evitar duplicados...")
    collection.delete(where={})
    print("Colección vacía.")

    # Preparar datos
    ids = []
    documents = []
    metadatas = []

    print("Creando embeddings y metadatos...")

    for idx, row in df.iterrows():

        doc_id = row["doc_id"]
        text = row["text"]
        source = row["source"]
        chunk_index = row.get("chunk_id", idx)

        # Clasificación del documento
        doc_type = detect_doc_type(str(source))

        unique_id = f"{doc_id}_chunk_{chunk_index}"

        ids.append(unique_id)
        documents.append(text)
        metadatas.append(
            {
                "doc_id": doc_id,
                "source": source,
                "chunk_index": int(chunk_index),
                "doc_type": doc_type,
            }
        )

    # Crear embeddings en batch
    print("Generando embeddings...")

    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=documents
    )

    vectors = [e.embedding for e in embedding_response.data]

    print(f"Subiendo {len(vectors)} vectores al índice...")

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas
    )

    print("Índice vectorial creado correctamente.")


# Entry point
if __name__ == "__main__":
    print("=== Construcción del índice vectorial BioActives RAG ===")
    build_index()
    print("=== Proceso completado ===")
