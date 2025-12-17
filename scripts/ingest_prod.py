import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import models # Necesario para el filtro de chequeo

# Configurar path para que encuentre 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Cargar variables de entorno
load_dotenv()

from src.ingestion.factory import DocumentLoaderFactory
from src.ingestion.embedders import OpenAIAdapter
from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService

# --- CONFIGURACIÓN ---
PROD_COLLECTION_NAME = "bioactives_prod"
PROD_PDF_DIR = str(ROOT_DIR / "data" / "prod_raw")
BATCH_SIZE_FILES = 5

def run_prod_ingestion_resume():
    print(f"🏭 INICIANDO INGESTA (MODO RESUME / NO-DUPLICADOS)")
    print("===================================================")

    # 1. Configuración de Red
    raw_url = os.getenv("QDRANT_URL")
    if not raw_url:
        print("❌ Error: Falta QDRANT_URL")
        return
    raw_url = raw_url.strip()

    # 2. Inicializar Servicios
    try:
        openai_adapter = OpenAIAdapter()
        
        # Instancia para subida
        db_impl = QdrantImpl(
            collection_name=PROD_COLLECTION_NAME, 
            url=raw_url, 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        rag_service = VectorStoreService(db_impl=db_impl)
        
        # Cliente crudo para verificar existencia (Chequeo rápido)
        client = db_impl.client 
        
        print("✅ Servicios conectados.")
    except Exception as e:
        print(f"❌ Error inicializando: {e}")
        return

    # 3. Listar Archivos Locales
    if not os.path.exists(PROD_PDF_DIR):
        print(f"❌ No existe carpeta: {PROD_PDF_DIR}")
        return

    all_files = [f for f in os.listdir(PROD_PDF_DIR) if f.endswith(".pdf")]
    total_files = len(all_files)
    print(f"📚 Archivos en carpeta local: {total_files}")

    # 4. Procesamiento
    files_processed_count = 0
    files_skipped_count = 0

    for i in range(0, total_files, BATCH_SIZE_FILES):
        batch_files = all_files[i : i + BATCH_SIZE_FILES]
        batch_chunks_to_process = []
        
        print(f"\n🔹 Revisando Lote {i//BATCH_SIZE_FILES + 1}...")

        # A. Filtrado (¿Ya existe?) y Carga
        for filename in batch_files:
            
            # --- LÓGICA DE VERIFICACIÓN (RESUME) ---
            try:
                # Preguntamos a Qdrant si hay AL MENOS 1 chunk con este source_file
                # Nota: Buscamos por el nombre limpio (filename)
                count_result = client.count(
                    collection_name=PROD_COLLECTION_NAME,
                    count_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=filename)
                            )
                        ]
                    )
                )
                
                if count_result.count > 0:
                    print(f"   ⏭️ Saltando (Ya existe): {filename}")
                    files_skipped_count += 1
                    continue # Saltamos al siguiente archivo
                    
            except Exception as e:
                print(f"   ⚠️ No se pudo verificar existencia de {filename}, se intentará procesar. Error: {e}")

            # --- SI NO EXISTE, PROCESAMOS ---
            file_path = os.path.join(PROD_PDF_DIR, filename)
            try:
                print(f"   📖 Leyendo NUEVO: {filename}...", end="\r")
                loader = DocumentLoaderFactory.get_loader(file_path)
                chunks = loader.load_and_chunk(file_path)
                
                # Limpieza de ruta
                for chunk in chunks:
                    if chunk.source_file:
                        chunk.source_file = os.path.basename(chunk.source_file)

                batch_chunks_to_process.extend(chunks)
            except Exception as e:
                print(f"\n   ⚠️ Error leyendo {filename}: {e}")
        
        if not batch_chunks_to_process:
            continue # Si todos en el lote ya existían, pasamos al siguiente lote

        # B. Embedding (Solo de lo nuevo)
        try:
            print(f"\n   🧠 Vectorizando {len(batch_chunks_to_process)} chunks nuevos...", end="\r")
            # El adaptador ya tiene el fix de mini-batches, así que no fallará por tamaño
            dense_vectors = openai_adapter.embed_chunks(batch_chunks_to_process)
            
            for idx, chunk in enumerate(batch_chunks_to_process):
                chunk.dense_vector = dense_vectors[idx]
        except Exception as e:
            print(f"\n   ❌ Error vectorizando lote: {e}")
            continue

        # C. Subida
        try:
            print(f"   💾 Subiendo a Qdrant...", end="\r")
            rag_service.index_chunks(batch_chunks_to_process)
            print("   ✅ Nuevos datos guardados.")
            files_processed_count += len(batch_files)
        except Exception as e:
            print(f"\n   ❌ ERROR SUBIENDO A QDRANT: {e}")

    print("\n" + "="*50)
    print("🏁 PROCESO FINALIZADO")
    print(f"⏭️  Archivos Omitidos (Ya existían): {files_skipped_count}")
    print(f"✅  Archivos Procesados (Nuevos):    {files_processed_count}")
    print("="*50)

if __name__ == "__main__":
    run_prod_ingestion_resume()