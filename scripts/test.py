import os
import sys

# Ajuste de path para importar módulos de src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from src.qdrant_impl import QdrantImpl
from openai import OpenAI

# 1. Cargar configuración
load_dotenv()
url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

def test_search():
    print("🔎 Iniciando prueba de búsqueda...")
    
    # 2. Conectar a la BD
    try:
        db = QdrantImpl(collection_name="bioactives_v1", url=url, api_key=api_key)
    except Exception as e:
        print(f"❌ Error conectando a Qdrant: {e}")
        return

    # 3. Tu pregunta
    query_text = "compuestos polifenólicos en bayas chilenas" 
    print(f"❓ Pregunta: '{query_text}'")
    
    try:
        # 4. Vectorizar la pregunta
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("🧠 Generando embedding...")
        response = client.embeddings.create(
            input=query_text,
            model="text-embedding-3-small"
        )
        query_vector = response.data[0].embedding
        
        # 5. Buscar en Qdrant
        print("🚀 Buscando en la base de datos...")
        results = db.query_data(
            query_vector=query_vector,
            filters=None,
            top_k=3 
        )
        
        # 6. Mostrar resultados formateados
        print(f"\n✅ --- SE ENCONTRARON {len(results)} FRAGMENTOS RELEVANTES ---")
        
        for i, res in enumerate(results):
            # Extraer datos usando las llaves REALES que descubrimos
            content = res.get("content", "❌ Sin texto")
            full_path = res.get("source_file", "Desconocido")
            
            # Limpiamos la ruta para ver solo el nombre del archivo
            filename = os.path.basename(full_path) 
            
            print(f"\n📄 [Resultado {i+1}]")
            print(f"   📂 Fuente: {filename}")
            print(f"   📝 Extracto: \"{content[:300]}...\"") # Primeros 300 caracteres
            print(f"   🔗 ID: {res.get('chunk_id')}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Error durante el proceso: {e}")

if __name__ == "__main__":
    test_search()