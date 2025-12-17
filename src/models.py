# models.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ProcessedChunk(BaseModel):
    """
    Contrato de datos del pipeline RAG.
    - content: texto del chunk
    - metadata: info para trazabilidad y filtros
    - dense_vector / sparse_vector: opcionales (según estrategia)
    """

    # Identidad / trazabilidad (útiles para benchmark y auditoría)
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None

    # Contenido principal
    content: str

    # Metadata general
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Embeddings / representación (si aplica)
    dense_vector: Optional[List[float]] = None
    # ej: {"indices":[...],"values":[...]}
    sparse_vector: Optional[Dict[str, Any]] = None

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata_dict(cls, v):
        return v or {}

    @field_validator("metadata")
    @classmethod
    def validate_mz_rt(cls, md: Dict[str, Any]) -> Dict[str, Any]:
        """
        Si metadata incluye 'mz' o 'rt', los normaliza a float si es posible.
        Esto es clave para filtros tipo Self-Query o filtros numéricos en DB vectorial.
        """
        for k in ("mz", "rt"):
            if k in md and md[k] is not None:
                try:
                    md[k] = float(md[k])
                except (TypeError, ValueError):
                    # Si no se puede convertir, lo dejamos como está
                    pass
        return md
