from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .models import ProcessedChunk

class RetrievalStrategy(ABC):
    """
    Interfaz abstracta para estrategias de recuperación.
    Permite cambiar el algoritmo de búsqueda sin romper el código cliente (main.py).
    Fuente: Documento 04 - Retrieval y Strategy, Pag 13.
    """

    @abstractmethod
    def retrieve_context(self, query: str, filters: Dict[str, Any], k: int) -> List[ProcessedChunk]:
        """
        Método abstracto que debe implementar cualquier estrategia concreta.
        """
        pass