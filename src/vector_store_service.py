import os
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI

from src.vector_store_impl import VectorStoreImpl
from src.models import ProcessedChunk


def extract_chemical_metadata(text: str) -> Dict[str, Any]:
    """Extrae m/z y RT del texto usando Regex para indexación numérica."""
    meta: Dict[str, Any] = {}
    if not isinstance(text, str) or not text:
        return meta

    mz_match = re.search(
        r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if mz_match:
        try:
            meta["mz"] = float(mz_match.group(1))
        except ValueError:
            pass

    rt_match = re.search(
        r"(?:rt|retention time)\s*[:=]?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    if rt_match:
        try:
            meta["rt"] = float(rt_match.group(1))
        except ValueError:
            pass

    return meta


class VectorStoreService:
    """
    Abstracción del patrón Bridge.
    - Recibe ProcessedChunk (contrato del pipeline)
    - Serializa (vectors + payloads) para la implementación concreta (QdrantImpl, etc.)
    - Ofrece query_text(): vectoriza la query y consulta (para Strategy).
    """

    def __init__(self, db_impl: VectorStoreImpl, embedding_model: str = "text-embedding-3-small"):
        self._db_impl = db_impl
        self.embedding_model = embedding_model
        self._openai: Optional[OpenAI] = None  # lazy init

    # -------------------------
    # Embeddings (para query_text)
    # -------------------------
    def _get_openai_client(self) -> OpenAI:
        if self._openai is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY no está configurado (necesario para query_text).")
            self._openai = OpenAI(api_key=api_key)
        return self._openai

    def _embed_query(self, text: str) -> List[float]:
        client = self._get_openai_client()
        resp = client.embeddings.create(
            model=self.embedding_model, input=[text])
        return resp.data[0].embedding

    def query_text(self, query_text: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[ProcessedChunk]:
        """
        API para Strategy: recibe texto, lo vectoriza y consulta.
        """
        qvec = self._embed_query(query_text)
        return self.query(query_vector=qvec, filters=filters, top_k=top_k)

    # -------------------------
    # Indexación
    # -------------------------
    def index_chunks(self, chunks: List[ProcessedChunk]) -> None:
        vectors: List[List[float]] = []
        payloads: List[Dict[str, Any]] = []

        for chunk in chunks:
            if not chunk.dense_vector:
                continue

            # NO mutar el objeto original
            source_basename = os.path.basename(
                chunk.source_file) if chunk.source_file else None

            # Metadata segura (copia)
            md: Dict[str, Any] = dict(chunk.metadata or {})

            # Enriquecimiento automático (si no venían)
            extracted = extract_chemical_metadata(chunk.content)
            for k, v in extracted.items():
                md.setdefault(k, v)

            # Normalización numérica si aplica
            for k in ("mz", "rt"):
                if k in md and md[k] is not None:
                    try:
                        md[k] = float(md[k])
                    except (TypeError, ValueError):
                        pass

            payload: Dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": source_basename,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "content": chunk.content,
                "metadata": md,
            }

            # Flatten mz/rt para filtros directos en Qdrant
            if "mz" in md and md["mz"] is not None:
                payload["mz"] = md["mz"]
            if "rt" in md and md["rt"] is not None:
                payload["rt"] = md["rt"]

            vectors.append(chunk.dense_vector)
            payloads.append(payload)

        if vectors:
            self._db_impl.index_data(vectors, payloads)

    # -------------------------
    # Query vectorial (Bridge)
    # -------------------------
    def query(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[ProcessedChunk]:
        raw_results = self._db_impl.query_data(
            query_vector, filters or {}, top_k)

        results: List[ProcessedChunk] = []
        for payload in raw_results:
            if not isinstance(payload, dict):
                continue

            md = dict(payload.get("metadata") or {})

            # Si vienen mz/rt en raíz, también a metadata por consistencia
            if "mz" in payload and payload["mz"] is not None:
                md.setdefault("mz", payload["mz"])
            if "rt" in payload and payload["rt"] is not None:
                md.setdefault("rt", payload["rt"])

            results.append(
                ProcessedChunk(
                    chunk_id=payload.get("chunk_id"),
                    doc_id=payload.get("doc_id"),
                    source_file=payload.get("source_file"),
                    chunk_index=payload.get("chunk_index"),
                    total_chunks=payload.get("total_chunks"),
                    content=payload.get("content", ""),
                    metadata=md,
                    dense_vector=None,
                    sparse_vector=None,
                )
            )

        return results
