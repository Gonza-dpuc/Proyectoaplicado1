from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from .vector_store_impl import VectorStoreImpl
import uuid

class QdrantImpl(VectorStoreImpl):
    def __init__(self, collection_name: str, url: str, api_key: Optional[str] = None):
        # Aseguramos limpieza nuevamente dentro de la clase por seguridad
        host = url.strip().replace("https://", "").replace("http://", "")
        if ":" in host:
            host = host.split(":")[0]
            
        print(f"-> [QdrantImpl] Host objetivo: '{host}'")

        self.client = QdrantClient(
            host=host,    # <--- USA HOST, NO URL
            port=6333,
            https=True,
            api_key=api_key,
            timeout=60    # <--- AUMENTAMOS TIMEOUT para cargas grandes
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                print(f"-> Creando colección '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="mz", 
                    field_schema=models.PayloadSchemaType.FLOAT
                )
        except Exception as e:
            print(f"⚠️ Aviso en _ensure_collection: {e}")

    def index_data(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        points = []
        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            point_id = meta.get("chunk_id", str(uuid.uuid4()))
            points.append(models.PointStruct(id=point_id, vector=vec, payload=meta))

        # Batching interno (Sube de a 50 vectores para ser más estable)
        batch_size = 50 
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

    # ... (El resto de métodos query_data y _build_filters igual que antes) ...
    def query_data(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        qdrant_filter = self._build_filters(filters)
        try:
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k
            ).points
            return [hit.payload for hit in search_result]
        except Exception as e:
            print(f"❌ Error en query_data: {e}")
            raise e

    def _build_filters(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        if not filters: return None
        conditions = []
        if "mz" in filters:
            conditions.append(models.FieldCondition(
                key="mz",
                range=models.Range(gte=filters["mz"].get("gte"), lte=filters["mz"].get("lte"))
            ))
        return models.Filter(must=conditions) if conditions else None