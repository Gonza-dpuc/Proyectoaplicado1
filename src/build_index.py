import pandas as pd
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

# Cargar variables de entorno
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL: QDRANT_URL = QDRANT_URL.strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL="text-embedding-3-small"
COLLECTION_NAME = "bioactives_hito1"

# Validación de API Key
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() == "":
    raise ValueError(
        "ERROR: OPENAI_API_KEY no está configurado.\n"
        "Crea un archivo .env en la raíz del proyecto."
    )

client_openai = OpenAI(api_key=OPENAI_API_KEY)

# --- Funciones Auxiliares ---

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

def get_uuid_from_string(string_id: str) -> str:
    """
    Qdrant requiere UUIDs o enteros para los IDs de los puntos.
    Generamos un UUID determinista basado en tu ID original (string).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))

def extract_chemical_metadata(text: str) -> dict:
    """Extrae m/z y RT del texto usando Regex para indexación numérica."""
    meta = {}
    # Busca patrones como: m/z 449.1, mz:449.107, mass 449.1
    mz_match = re.search(r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if mz_match:
        meta["mz"] = float(mz_match.group(1))
    
    # Busca patrones como: RT 8.2, rt:8.2 min
    rt_match = re.search(r"(?:rt|retention time)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if rt_match:
        meta["rt"] = float(rt_match.group(1))
    return meta

# --- Construcción del índice ---

def build_index():
    print("Cargando chunks procesados...")
    chunks_path = Path("data/processed/chunks.parquet")
    if not chunks_path.exists():
        raise FileNotFoundError(f"No existe {chunks_path}.")

    df = pd.read_parquet(chunks_path)
    print(f"Chunks cargados: {len(df)}")

    # 1. Inicializar cliente Qdrant (Soporte Cloud/Local)
    client_qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 2. Re-crear la colección
    print(f"Verificando colección '{COLLECTION_NAME}'...")
    
    # Verificamos si existe y la borramos para empezar de cero (igual que tu script anterior)
    if client_qdrant.collection_exists(collection_name=COLLECTION_NAME):
        client_qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print("Colección previa eliminada.")

    # Crear colección definiendo el tamaño del vector
    # text-embedding-3-small y ada-002 usan 1536 dimensiones.
    client_qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536, 
            distance=models.Distance.COSINE
        ),
    )
    print(f"Colección '{COLLECTION_NAME}' creada exitosamente.")

    # 3. Preparar datos
    ids = []
    documents_list = [] # Guardamos texto para generar embeddings
    payloads = []       # Metadatos para Qdrant

    print("Preparando documentos y metadatos...")

    for idx, row in df.iterrows():
        doc_id = row["doc_id"]
        text = row["text"]
        source = row["source"]
        chunk_index = row.get("chunk_index", idx)

        doc_type = detect_doc_type(str(source))
        
        # ID original (String)
        original_id = f"{doc_id}_chunk_{chunk_index}"
        
        # ID para Qdrant (UUID)
        point_id = get_uuid_from_string(original_id)

        # Extraer metadatos químicos (NUEVO)
        chem_meta = extract_chemical_metadata(text)

        ids.append(point_id)
        documents_list.append(text)
        
        # En Qdrant, el texto del chunk va DENTRO del payload (metadata)
        payloads.append({
            "chunk_id": original_id,       # Estandarizado (antes original_id)
            "content": text,               # Estandarizado (antes text)
            "source_file": os.path.basename(str(source)), # Estandarizado (antes source)
            "doc_id": doc_id,
            "doc_type": doc_type,
            "chunk_index": int(chunk_index),
            "mz": chem_meta.get("mz"), # Campo numérico para filtro
            "rt": chem_meta.get("rt")  # Campo numérico para filtro
        })

    # 4. Embeddings en Batch y Subida
    BATCH_SIZE = 200
    print(f"Generando embeddings e indexando en batches de {BATCH_SIZE}...")

    total_docs = len(documents_list)
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch_end = i + BATCH_SIZE
        batch_docs = documents_list[i: batch_end]
        batch_ids = ids[i: batch_end]
        batch_payloads = payloads[i: batch_end]

        print(f"Procesando batch {i} – {min(batch_end, total_docs)} / {total_docs}")

        try:
            # Generar embeddings con OpenAI
            response = client_openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_docs,
            )
            vectors = [emb.embedding for emb in response.data]

            # Crear estructuras de Puntos para Qdrant
            points = [
                models.PointStruct(
                    id=bid,
                    vector=vec,
                    payload=bpay
                )
                for bid, vec, bpay in zip(batch_ids, vectors, batch_payloads)
            ]

            # Subir a Qdrant (Upsert)
            client_qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            
        except Exception as e:
            print(f"Error en el batch {i}: {e}")
            # Opcional: break o continue dependiendo de lo estricto que quieras ser

    print("Índice vectorial en Qdrant creado correctamente.")

# Entry point
if __name__ == "__main__":
    print("=== Construcción del índice vectorial BioActives (Qdrant) ===")
    build_index()
    print("=== Proceso completado ===")