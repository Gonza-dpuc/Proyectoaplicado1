# retriever.py (LEGACY/DEBUG ONLY)
"""
LEGACY / DEBUG:
Este archivo realiza query directa a Qdrant sin pasar por Bridge/Strategy.
NO debe ser usado por el pipeline principal evaluable del curso.
"""

from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None


class LegacyQdrantRetrieverDebug:
    def __init__(self, url: str, api_key: Optional[str], collection_name: str):
        if QdrantClient is None:
            raise ImportError(
                "qdrant_client no está instalado o no se pudo importar.")
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def query(self, vector: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # Útil para inspección rápida. No para arquitectura final.
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload for p in res.points]
