# src/rag_service.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from src.models import ProcessedChunk
from src.retrieval_strategy import RetrievalStrategy


class RAGService:
    """
    Servicio de alto nivel: orquesta retrieval + armado de contexto + generación.
    Mantiene compatibilidad con Streamlit vía answer_question().
    """

    def __init__(self, retriever: Any, openai_api_key: str):
        """
        retriever: puede ser:
          - un Strategy/Decorator con retrieve_context(...)
          - o un adaptador con retrieve(...) (como _StreamlitRetrieveAdapter)
        """
        self.retriever = retriever
        self.client = OpenAI(api_key=openai_api_key)

        # Modelo configurable por env var (para no hardcodear)
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    # -----------------------------
    # Compat con tu app_streamlit.py
    # -----------------------------
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        use_self_query: bool = True,
        use_repacking: bool = True,
        generate_response: bool = True,
    ) -> Dict[str, Any]:
        """
        Retorna un dict compatible con Streamlit:
        - answer: str
        - sources: List[str]
        - context_used: List[ProcessedChunk|dict]
        """
        filters = self._extract_filters_from_query(
            question) if use_self_query else None

        # Retrieval: soporta ambos contratos
        docs = self._retrieve(question, top_k=top_k, filters=filters)

        # Normalizar a ProcessedChunk (si vienen dicts)
        chunks = [self._to_chunk(d) for d in docs]

        if use_repacking:
            chunks = self._repack(chunks)

        sources = self._extract_sources(chunks)

        if not generate_response:
            return {
                "answer": "✅ Retrieval listo. (Modo: Solo Retrieval)",
                "sources": sources,
                "context_used": chunks,
            }

        context_text = self._build_context_text(chunks)
        prompt = self._build_prompt(question, context_text)

        answer = self._generate_answer(prompt)

        return {
            "answer": answer,
            "sources": sources,
            "context_used": chunks,
        }

    # (Opcional) Si tu código nuevo usa answer(), lo dejamos como alias limpio
    def answer(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        docs = self._retrieve(question, top_k=top_k, filters=filters)
        chunks = [self._to_chunk(d) for d in docs]
        context_text = self._build_context_text(chunks)
        prompt = self._build_prompt(question, context_text)
        return self._generate_answer(prompt)

    # -----------------------------
    # Internals
    # -----------------------------
    def _retrieve(self, query: str, top_k: int, filters: Optional[Dict[str, Any]]):
        # Caso A: estrategia nueva
        if hasattr(self.retriever, "retrieve_context"):
            return self.retriever.retrieve_context(query=query, k=top_k, filters=filters)
        # Caso B: adaptador legacy
        if hasattr(self.retriever, "retrieve"):
            return self.retriever.retrieve(query=query, top_k=top_k, filters=filters)

        raise AttributeError(
            "El retriever no implementa retrieve_context(...) ni retrieve(...).")

    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Self-query simple por regex.
        Genera rangos para mz/rt si aparecen.
        """
        filters: Dict[str, Any] = {}

        mz_match = re.search(
            r"(?:m\/z|mz)\s*[:=]?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
        rt_match = re.search(
            r"\b(rt|retention time)\b\s*[:=]?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)

        if mz_match:
            mz = float(mz_match.group(1))
            filters["mz"] = {"gte": mz - 0.05, "lte": mz + 0.05}

        if rt_match:
            rt = float(rt_match.group(2))
            filters["rt"] = {"gte": rt - 0.2, "lte": rt + 0.2}

        return filters

    def _to_chunk(self, d: Union[ProcessedChunk, Dict[str, Any]]) -> ProcessedChunk:
        if isinstance(d, ProcessedChunk):
            return d
        # payload dict
        md = d.get("metadata") or {}
        return ProcessedChunk(
            chunk_id=d.get("chunk_id"),
            doc_id=d.get("doc_id"),
            source_file=d.get("source_file"),
            chunk_index=d.get("chunk_index"),
            total_chunks=d.get("total_chunks"),
            content=d.get("content", ""),
            metadata=md,
        )

    def _repack(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """
        Repacking simple: agrupa por source_file para evitar saltos de contexto.
        """
        by_source: Dict[str, List[ProcessedChunk]] = {}
        for c in chunks:
            src = c.source_file or "unknown"
            by_source.setdefault(src, []).append(c)

        repacked: List[ProcessedChunk] = []
        for src, group in by_source.items():
            repacked.extend(group)
        return repacked

    def _extract_sources(self, chunks: List[ProcessedChunk]) -> List[str]:
        seen = set()
        sources = []
        for c in chunks:
            if c.source_file and c.source_file not in seen:
                seen.add(c.source_file)
                sources.append(c.source_file)
        return sources

    def _build_context_text(self, chunks: List[ProcessedChunk]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            src = c.source_file or "unknown"
            parts.append(f"[{i}] Source: {src}\n{c.content}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "Eres un asistente científico. Responde usando SOLO el contexto entregado.\n"
            "Si el contexto no contiene la respuesta, indícalo claramente.\n\n"
            f"Pregunta:\n{question}\n\n"
            f"Contexto:\n{context}\n\n"
            "Respuesta:"
        )

    def _generate_answer(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Responde de forma concisa y basada en evidencia del contexto."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
