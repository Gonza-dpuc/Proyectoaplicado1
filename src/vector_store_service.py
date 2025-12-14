import os  # <--- Necesario para limpiar la ruta
from typing import List, Dict, Any
from .vector_store_impl import VectorStoreImpl
# Asegúrate de importar tu modelo de datos si lo usas, ej:
# from .models import ProcessedChunk 

class VectorStoreService:
    """
    Abstracción (Bridge): Gestiona la lógica de negocio.
    Es el PORTERO: Todo lo que entra a la BD pasa por aquí.
    """
    
    def __init__(self, db_impl: VectorStoreImpl):
        self._db_impl = db_impl

    def index_chunks(self, chunks: List[Any]):
        """
        Recibe una lista de chunks (objetos ProcessedChunk), limpia sus metadatos
        y los envía a la base de datos.
        """
        if not chunks:
            return

        # 1. Separar vectores de metadatos y LIMPIAR DATOS
        vectors = []
        payloads = []

        for chunk in chunks:
            # Validación defensiva
            if not chunk.dense_vector:
                print(f"⚠️ Chunk {chunk.chunk_id} sin vector denso. Saltando.")
                continue
                
            vectors.append(chunk.dense_vector)
            
            # --- 🔥 CAMBIO PERMANENTE AQUÍ 🔥 ---
            # Antes de convertir a diccionario, limpiamos la ruta en el objeto
            if chunk.source_file:
                # Transforma "C:/Users/pepe/data/paper.pdf" -> "paper.pdf"
                chunk.source_file = os.path.basename(chunk.source_file)
            # -------------------------------------
            
            # Serializamos el objeto a diccionario para Qdrant
            # (Dependiendo de si usas Pydantic v1 o v2, puede ser .dict() o .model_dump())
            if hasattr(chunk, "model_dump"):
                payload = chunk.model_dump(exclude={'dense_vector', 'sparse_vector'})
            else:
                payload = chunk.__dict__.copy()
                if 'dense_vector' in payload: del payload['dense_vector']
                if 'sparse_vector' in payload: del payload['sparse_vector']

            payloads.append(payload)

        # 2. Delegar a la implementación (QdrantImpl)
        if vectors:
            self._db_impl.index_data(vectors, payloads)

    def query(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Any]:
        # ... (Tu código de query sigue igual) ...
        # Delegar búsqueda
        raw_results = self._db_impl.query_data(query_vector, filters, top_k)
        
        # Como usamos una clase genérica aquí para el ejemplo, retornamos raw o rehidratamos
        # Si tienes la clase ProcessedChunk importada, úsala aquí.
        # Por ahora devolvemos los diccionarios para no romper imports circulares si no tengo tu models.py
        return raw_results