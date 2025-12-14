from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from .vector_store_impl import VectorStoreImpl
import uuid

class QdrantImpl(VectorStoreImpl):
    """
    Implementación del Bridge para Qdrant.
    Usa la lógica de conexión probada en debug_network.py
    """

    def __init__(self, collection_name: str, url: str, api_key: Optional[str] = None):
        # --- 1. LÓGICA DE LIMPIEZA (Idéntica al script de diagnóstico) ---
        # Tomamos la URL del .env y extraemos solo el HOST limpio.
        host = url.strip()
        host = host.replace("https://", "").replace("http://", "")
        if ":" in host:
            host = host.split(":")[0] 
            
        print(f"-> [QdrantImpl] Inicializando conexión a Host: '{host}'")

        # --- 2. CONEXIÓN EXPLÍCITA ---
        # Usamos host/port/https por separado para evitar errores de DNS en httpx
        self.client = QdrantClient(
            host=host,
            port=6333,
            https=True,
            api_key=api_key
        )
        
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                print(f"-> [QdrantImpl] Creando colección '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                )
                
                # Índice para filtros rápidos (Usamos "mz" plano)
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="mz", 
                    field_schema=models.PayloadSchemaType.FLOAT
                )
                print("-> [QdrantImpl] Índice creado para campo 'mz'.")
        except Exception as e:
            # Si falla aquí, es para avisar antes de intentar subir nada
            print(f"⚠️ Advertencia verificando colección: {e}")

    def index_data(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        points = []
        for i, (vec, meta) in enumerate(zip(vectors, metadata)):
            # ID determinista o aleatorio
            point_id = meta.get("chunk_id", str(uuid.uuid4()))
            
            points.append(models.PointStruct(
                id=point_id,
                vector=vec,
                payload=meta 
            ))

        # --- 3. SUBIDA POR LOTES (BATCHING) ---
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        print(f"-> [QdrantImpl] Subiendo {len(points)} puntos en {total_batches} lotes...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            except Exception as e:
                print(f"❌ Error crítico subiendo lote {i//batch_size + 1}: {e}")
                raise e
            
        print(f"-> [QdrantImpl] ✅ Carga completada exitosamente.")

    def query_data(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        # Construimos el filtro
        qdrant_filter = self._build_filters(filters)
        
        try:
            # ✅ USAMOS 'query_points' (Método moderno y disponible en tu versión)
            # Nota: El argumento correcto es 'query_filter', no 'filter'.
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,       # En versiones nuevas, 'query' recibe el vector
                query_filter=qdrant_filter, # <--- AQUÍ ESTABA EL ERROR
                limit=top_k
            ).points # .points extrae la lista de resultados del objeto respuesta
            
            # Retornamos solo el payload
            return [hit.payload for hit in search_result]
            
        except Exception as e:
            print(f"❌ Error en query_data: {e}")
            raise e

    def _build_filters(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        if not filters:
            return None
        conditions = []
        
        # Filtro de Rango para m/z
        if "mz" in filters:
            mz_filter = filters["mz"]
            conditions.append(models.FieldCondition(
                key="mz",
                range=models.Range(
                    gte=mz_filter.get("gte"),
                    lte=mz_filter.get("lte")
                )
            ))
            
        if not conditions:
            return None
        return models.Filter(must=conditions)