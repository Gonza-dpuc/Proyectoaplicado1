# src/models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any

class ProcessedChunk(BaseModel):
    """
    Define la estructura de datos que fluye por todo el sistema RAG.
    """
    chunk_id: str
    content: str
    source_file: str
    
    # Metadatos flexibles. 
    # Qdrant usará esto como "Payload".
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[List[float]] = None

    # --- AGREGADO: VALIDACIÓN AUTOMÁTICA DE TIPOS ---
    @field_validator('metadata')
    @classmethod
    def force_numeric_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asegura que campos críticos para filtrado (mz, rt) sean floats,
        incluso si vienen como strings desde un CSV/Parquet.
        """
        # Lista de campos que SIEMPRE deben ser numéricos para que Qdrant funcione
        numeric_fields = ['mz', 'rt', 'mass', 'retention_time']
        
        for key in numeric_fields:
            if key in v and v[key] is not None:
                try:
                    # Intentamos convertir a float. 
                    # Si es "449.1" (str) -> 449.1 (float)
                    v[key] = float(v[key])
                except (ValueError, TypeError):
                    # Si falla (ej: "N/A"), lo dejamos como None o lo borramos
                    v[key] = None
        
        return v