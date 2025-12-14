import chromadb
from src.config import CHROMA_DIR

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

print("=== Listado de colecciones existentes ===")
cols = client.list_collections()
for c in cols:
    print(f"- {c.name}")

if len(cols) == 0:
    print("No hay colecciones en este directorio.")
