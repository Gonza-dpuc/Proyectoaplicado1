import os
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
from src.models import ProcessedChunk

class AbstractEmbedderAdapter(ABC):
    """Target: Interfaz estandarizada que el sistema espera."""
    @abstractmethod
    def embed_chunks(self, chunks: List[ProcessedChunk]) -> List[List[float]]:
        pass

class OpenAIAdapter(AbstractEmbedderAdapter):
    """Adaptee: Conecta con la API de OpenAI."""
    def __init__(self, model="text-embedding-3-small"):
        # Asegúrate de tener OPENAI_API_KEY en tu .env
        self.client = OpenAI() 
        self.model = model

    def embed_chunks(self, chunks: List[ProcessedChunk]) -> List[List[float]]:
        print(f"-> [OpenAIAdapter] Generando embeddings para {len(chunks)} chunks...")
        
        # OpenAI permite lotes (batches). Preparamos solo el texto.
        texts = [c.content.replace("\n", " ") for c in chunks]
        
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            # Extraemos los vectores en orden
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error en OpenAI Embedding: {e}")
            return []

class DummySparseAdapter(AbstractEmbedderAdapter):
    """
    Simulación de vectores dispersos (Sparse) para cumplir el requisito híbrido 
    sin instalar librerías complejas todavía (como Pinecone-text o Splade).
    """
    def embed_chunks(self, chunks: List[ProcessedChunk]) -> List[List[float]]:
        # Retorna vectores vacíos o aleatorios solo para que el código no rompa.
        # En producción, aquí iría BM25.
        vector_size = 1536 # Debe coincidir o ser manejado por Qdrant
        return [[0.0] * vector_size for _ in chunks]