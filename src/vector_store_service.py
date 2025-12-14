from typing import List, Dict, Any
from .vector_store_impl import VectorStoreImpl
from .models import ProcessedChunk

class VectorStoreService:
    """
    Abstracción (Bridge): Gestiona la lógica de negocio.
    Convierte objetos de dominio (ProcessedChunk) a formatos de BD y viceversa.
    """
    
    def __init__(self, db_impl: VectorStoreImpl):
        self._db_impl = db_impl

    def index_chunks(self, chunks: List[ProcessedChunk]):
        if not chunks:
            return

        # 1. Separar vectores de metadatos
        vectors = []
        payloads = []

        for chunk in chunks:
            # Validación defensiva: Si no hay vector denso, no podemos indexar (o usamos ceros)
            if not chunk.dense_vector:
                print(f"⚠️ Chunk {chunk.chunk_id} sin vector denso. Saltando.")
                continue
                
            vectors.append(chunk.dense_vector)
            
            # Serializamos el objeto Pydantic a diccionario para el payload
            # Excluimos los vectores para no duplicar datos en el payload
            payload = chunk.model_dump(exclude={'dense_vector', 'sparse_vector'})
            payloads.append(payload)

        # 2. Delegar a la implementación (QdrantImpl)
        if vectors:
            self._db_impl.index_data(vectors, payloads)

    def query(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[ProcessedChunk]:
        """
        Realiza la búsqueda y convierte los resultados crudos (dicts) 
        de nuevo a objetos ProcessedChunk.
        """
        # Delegar búsqueda
        raw_results = self._db_impl.query_data(query_vector, filters, top_k)
        
        # Rehidratar objetos
        chunks = []
        for r in raw_results:
            try:
                # Reconstruimos el objeto Pydantic desde el diccionario
                chunk = ProcessedChunk(**r)
                chunks.append(chunk)
            except Exception as e:
                print(f"Error rehidratando chunk desde DB: {e}")
                
        return chunks