import os
from typing import List, Dict, Any
import qdrant_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class BioactivesRetriever:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        if url: url = url.strip()
        
        # Configuración Qdrant
        self.qdrant_client = qdrant_client.QdrantClient(
            url=url,
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Configuración OpenAI (para embedder la query)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        1. Convierte la query a vector.
        2. Busca en Qdrant.
        3. Retorna una lista limpia de resultados.
        """
        
        # 1. Generar Embedding de la pregunta
        # Nota: Usamos la API directa aquí para ser rápidos, pero podrías reusar tu OpenAIAdapter
        response = self.openai_client.embeddings.create(
            input=query,
            model=self.embedding_model
        )
        query_vector = response.data[0].embedding

        # 2. Buscar en Qdrant
        # Actualización: Usamos query_points compatible con qdrant-client v1.10+
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points

        # 3. Formatear salida para evaluación
        formatted_results = []
        for hit in search_result:
            payload = hit.payload
            formatted_results.append({
                "content": payload.get("content") or payload.get("text"), # Fallback por si cambió el nombre
                "source_file": payload.get("source_file") or payload.get("source"),
                "score": hit.score,
                "doc_type": payload.get("doc_type", "unknown"),
                "chunk_id": payload.get("chunk_id", "unknown")
            })

        return formatted_results