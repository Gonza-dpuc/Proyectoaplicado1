from typing import List, Dict, Any
from .retrieval_strategy import RetrievalStrategy
from .vector_store_service import VectorStoreService
from .models import ProcessedChunk

class HybridSearchStrategy(RetrievalStrategy):
    """
    Implementación concreta de búsqueda híbrida.
    Fuente: Documento 04 - Retrieval y Strategy, Pag 13.
    """
    
    def __init__(self, store_service: VectorStoreService):
        # Inyección de dependencia: La estrategia usa el servicio (Bridge Abstraction)
        self._store = store_service 

    def retrieve_context(self, query: str, filters: Dict[str, Any], k: int) -> List[ProcessedChunk]:
        print(f"-> [Strategy]: Ejecutando Búsqueda HÍBRIDA con filtros: {filters}")
        
        # Delegamos la complejidad técnica al servicio (Bridge)
        # El servicio se encargará de vectorizar la query y llamar a la DB
        return self._store.query(query_text=query, filters=filters, top_k=k)