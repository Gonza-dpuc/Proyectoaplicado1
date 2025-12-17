from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class VectorStoreImpl(ABC):
    """
    Interfaz Bridge (Implementación): Define las operaciones de bajo nivel 
    que cualquier base de datos vectorial debe cumplir.
    """

    @abstractmethod
    def index_data(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """Sube vectores y metadatos a la base de datos."""
        pass

    @abstractmethod
    def query_data(self, query_vector: List[float], filters: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
        """Realiza la búsqueda vectorial y retorna los metadatos crudos."""
        pass
