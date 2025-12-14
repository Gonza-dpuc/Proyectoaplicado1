from abc import ABC, abstractmethod
from typing import List
# Asumiendo que ProcessedChunk está en src/models.py
# Si te da error de import, ajusta la ruta según donde creaste models.py
from src.models import ProcessedChunk 

class AbstractLoader(ABC):
    """
    Define el contrato de carga para todos los formatos.
    Cualquier clase que herede de esto DEBE implementar load_and_chunk.
    """
    @abstractmethod
    def load_and_chunk(self, file_path: str) -> List[ProcessedChunk]:
        pass