import os
from typing import List, Any
from openai import OpenAI
import time

class OpenAIAdapter:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Falta OPENAI_API_KEY en .env")
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"

    def embed_chunks(self, chunks: List[Any], batch_size: int = 100) -> List[List[float]]:
        """
        Genera embeddings para una lista de chunks.
        
        MEJORA CRÍTICA:
        Divide la lista de entrada en 'mini-lotes' (batch_size) más pequeños 
        para evitar el error 'max_tokens_per_request' de OpenAI.
        """
        all_embeddings = []
        total_chunks = len(chunks)
        
        # Si no hay chunks, retornamos lista vacía
        if total_chunks == 0:
            return []

        print(f"   🧠 [Adapter] Procesando {total_chunks} chunks en mini-lotes de {batch_size}...")

        # Procesamos en grupos pequeños (ej: de 100 en 100)
        for i in range(0, total_chunks, batch_size):
            # Recorte del lote actual
            mini_batch = chunks[i : i + batch_size]
            
            # Extraer solo el texto
            texts = [c.content for c in mini_batch]
            
            # Limpieza básica: OpenAI falla si el string está vacío
            texts = [t if t else " " for t in texts]

            try:
                # Llamada a la API (ahora segura por tamaño)
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                
                # Extraer vectores y agregarlos a la lista principal
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Opcional: Pequeña pausa para no saturar Rate Limits (RPM)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error en mini-lote {i}-{i+batch_size}: {e}")
                # En caso de error, podríamos lanzar la excepción o intentar reintentar.
                # Para este script, lanzamos para detener y revisar.
                raise e

        return all_embeddings