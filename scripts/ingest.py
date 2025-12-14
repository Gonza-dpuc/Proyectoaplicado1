import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Configurar path para que encuentre 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Cargar variables de entorno (.env)
load_dotenv()

from src.ingestion.factory import DocumentLoaderFactory
from src.ingestion.embedders import OpenAIAdapter
from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService

# Configuración
PDF_DIR = str(ROOT_DIR / "data" / "raw")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "bioactives_v1")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # <--- CORRECCIÓN 1: Leer la Key

def run_ingestion():
    print("🚀 Iniciando Pipeline de Ingesta...")
    print(f"🔍 DEBUG URL: '{QDRANT_URL}'")
    # 1. Preparar Componentes
    openai_adapter = OpenAIAdapter()
    
    # Inicializamos la conexión a la DB (Bridge) con la API KEY
    # <--- CORRECCIÓN 2: Pasarla al constructor
    db_impl = QdrantImpl(
        collection_name=COLLECTION_NAME, 
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY 
    )
    print(f"🔍 DEBUG: Intentando conectar a URL: '{QDRANT_URL}'")
    rag_service = VectorStoreService(db_impl=db_impl)

    all_chunks = []
    
    # 2. Carga y Chunking (Factory)
    if not os.path.exists(PDF_DIR):
        print(f"❌ Error: No existe la carpeta {PDF_DIR}")
        return

    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"📄 Encontrados {len(files)} archivos PDF.")

    for filename in files:
        file_path = os.path.join(PDF_DIR, filename)
        try:
            loader = DocumentLoaderFactory.get_loader(file_path)
            chunks = loader.load_and_chunk(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Error procesando {filename}: {e}")

    if not all_chunks:
        print("No se generaron chunks. Revisa tus PDFs.")
        return

    print(f"✅ Total chunks generados: {len(all_chunks)}")

    # 3. Embedding (Adapter)
    print("🧠 Generando Embeddings (esto puede tardar)...")
    dense_vectors = openai_adapter.embed_chunks(all_chunks)
    
    if len(dense_vectors) != len(all_chunks):
        print("❌ Error: Discrepancia entre chunks y vectores.")
        return

    for i, chunk in enumerate(all_chunks):
        chunk.dense_vector = dense_vectors[i]

    # 4. Indexación (Bridge Service)
    print("💾 Subiendo a Qdrant...")
    rag_service.index_chunks(all_chunks)
    
    print("🎉 Ingesta completada exitosamente.")

if __name__ == "__main__":
    run_ingestion()