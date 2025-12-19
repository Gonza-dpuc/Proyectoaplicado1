from typing import Dict, Any, Optional, List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.vector_store_impl import VectorStoreImpl


class QdrantImpl(VectorStoreImpl):
    """
    Implementación concreta (Bridge) para Qdrant.
    Compatible con qdrant-client 1.16.x usando query_points().
    """

    def __init__(self, url: str, api_key: Optional[str], collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self._collection_ready = False

    # -----------------------------
    # Helpers
    # -----------------------------
    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        # Índices payload para filtros numéricos (no es obligatorio para funcionar, pero ayuda)
        for field in ("mz", "rt"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.FLOAT,
                )
            except Exception:
                pass

        self._collection_ready = True

    def _build_filters(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        if not filters:
            return None

        conditions = []

        def add_range_condition(key: str, spec: Dict[str, Any]) -> None:
            range_kwargs = {}
            for k in ("gte", "lte", "gt", "lt"):
                if k in spec and spec[k] is not None:
                    range_kwargs[k] = spec[k]
            if not range_kwargs:
                return

            conditions.append(
                models.FieldCondition(
                    key=key,
                    range=models.Range(**range_kwargs),
                )
            )

        for key, value in filters.items():
            if isinstance(value, dict):
                if key in ("mz", "rt"):
                    add_range_condition(key, value)
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    # -----------------------------
    # Bridge implementation
    # -----------------------------
    def index_data(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        if not vectors:
            return
        if len(vectors) != len(payloads):
            raise ValueError("vectors y payloads deben tener el mismo largo")

        vector_size = len(vectors[0])
        self._ensure_collection(vector_size=vector_size)

        points: List[models.PointStruct] = []
        for idx, (vec, payload) in enumerate(zip(vectors, payloads)):
            point_id = payload.get("chunk_id") or f"pt_{idx}"
            points.append(models.PointStruct(
                id=point_id, vector=vec, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def query_data(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """
        ESTE método es el que te faltaba: query_data().
        Si no existe (o está fuera de la clase), Python marca QdrantImpl como abstracta.
        """
        if not query_vector:
            return []

        self._ensure_collection(vector_size=len(query_vector))
        q_filter = self._build_filters(filters or {})

        # Qdrant 1.16.x: query_points
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
            with_vectors=False,
        )

        return [p.payload for p in getattr(res, "points", [])]
