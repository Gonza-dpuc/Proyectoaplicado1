import os  # <--- Necesario para limpiar la ruta
from typing import List, Dict, Any
from .vector_store_impl import VectorStoreImpl
import re
# Asegúrate de importar tu modelo de datos si lo usas, ej:
# from .models import ProcessedChunk 

def extract_chemical_metadata(text: str) -> dict:
    """Extrae m/z y RT del texto usando Regex para indexación numérica."""
    meta = {}
    if not text: return meta
    # Busca patrones como: m/z 449.1, mz:449.107, mass 449.1
    mz_match = re.search(r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if mz_match:
        meta["mz"] = float(mz_match.group(1))
    
    # Busca patrones como: RT 8.2, rt:8.2 min
    rt_match = re.search(r"(?:rt|retention time)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if rt_match:
        meta["rt"] = float(rt_match.group(1))
    return meta

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

            # --- ENRIQUECIMIENTO AUTOMÁTICO ---
            # Extraemos metadatos químicos del contenido antes de subir
            chem_meta = extract_chemical_metadata(payload.get("content", ""))
            payload.update(chem_meta)

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