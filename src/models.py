# src/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ProcessedChunk(BaseModel):
    """
    Define la estructura de datos que fluye por todo el sistema RAG.
    Garantiza que el loader, el embedder y la base de datos hablen el mismo idioma.
    """
    chunk_id: str
    content: str
    source_file: str
    
    # Metadatos flexibles para bioactivos (mz, rt, formula, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Vectores (Opcionales, se llenan en el paso de embedding)
    # Importante: Optional para que no falle si el loader aún no generó embeddings
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[List[float]] = None