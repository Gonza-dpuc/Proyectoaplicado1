# hybrid_search_strategy.py
from typing import Dict, Any, List, Optional

from src.retrieval_strategy import RetrievalStrategy
from src.models import ProcessedChunk
from src.vector_store_service import VectorStoreService


class HybridSearchStrategy(RetrievalStrategy):
    def __init__(self, store: VectorStoreService):
        self.store = store

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedChunk]:
        # La Abstracción (VectorStoreService) se encarga de vectorizar y consultar
        return self.store.query_text(query_text=query, filters=filters or {}, top_k=k)
