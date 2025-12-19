# src/rag_service.py
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from src.models import ProcessedChunk


class RAGService:
    """
    Servicio de alto nivel: orquesta retrieval + armado de contexto + generación.
    Mantiene compatibilidad con Streamlit vía answer_question().

    Mejora clave (modo demo): si falla OpenAI (DNS/red), NO rompe la app.
    Retorna un mensaje claro + mantiene sources/contexto.
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

        # Retries para fallas transitorias de red/DNS
        self.max_generation_retries = int(os.getenv("OPENAI_GEN_RETRIES", "2"))

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

        # Fallback: si self-query dejó el retrieval vacío, reintentar sin filtros
        if use_self_query and filters and (not docs or len(docs) == 0):
            docs = self._retrieve(question, top_k=top_k, filters=None)

        # Normalizar a ProcessedChunk (si vienen dicts)
        chunks = [self._to_chunk(d) for d in docs]

        if use_repacking:
            chunks = self._repack(chunks)

        sources = self._extract_sources(chunks)

        # Modo solo retrieval
        if not generate_response:
            return {
                "answer": "Retrieval listo. (Modo: Solo Retrieval)",
                "sources": sources,
                "context_used": chunks,
                "generated_prompt": None,
            }

        context_text = self._build_context_text(chunks)
        prompt = self._build_prompt(question, context_text)

        # Tolerante a fallas de red/DNS: no romper el demo
        try:
            answer = self._generate_answer_with_retry(prompt)
        except Exception as e:
            answer = (
                "No se pudo generar la respuesta por un problema de conexión con OpenAI.\n\n"
                "El retrieval sí funcionó y se muestran las fuentes/contexto recuperados.\n\n"
                f"Detalle técnico: {type(e).__name__}: {e}"
            )

        return {
            "answer": answer,
            "sources": sources,
            "context_used": chunks,
            "generated_prompt": prompt,
        }

    # (Opcional) Alias limpio
    def answer(self, question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        docs = self._retrieve(question, top_k=top_k, filters=filters)
        chunks = [self._to_chunk(d) for d in docs]
        context_text = self._build_context_text(chunks)
        prompt = self._build_prompt(question, context_text)
        return self._generate_answer_with_retry(prompt)

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
            filters["mz"] = {"gte": mz - 0.2, "lte": mz + 0.2}

        if rt_match:
            rt = float(rt_match.group(2))
            filters["rt"] = {"gte": rt - 0.5, "lte": rt + 0.5}

        return filters

    def _to_chunk(self, d: Union[ProcessedChunk, Dict[str, Any]]) -> ProcessedChunk:
        if isinstance(d, ProcessedChunk):
            return d
        # Fix para Streamlit: si falla isinstance por recarga de módulos pero es objeto (no dict)
        if not isinstance(d, dict):
            return d

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
        for _, group in by_source.items():
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
            "Eres un asistente de investigación experto en fitoquímica, metabolómica y alimentos funcionales. "
            "Tu objetivo es responder a las consultas sintetizando la información del contexto proporcionado de manera breve y directa."
            "\n\n"
            "INSTRUCCIONES DE GENERACIÓN:\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. NO uses conocimientos previos externos. Si la respuesta no está en el contexto, di 'No cuento con información suficiente en los documentos procesados'.\n"
            "2. Sé preciso y técnico. Usa vocabulario científico (ej: menciona 'capacidad antioxidante', 'polifenoles', 'mecanismo de acción').\n"
            "3. CITA LAS FUENTES: Cuando hagas una afirmación, referencia el archivo de origen mencionado en el contexto.\n"
            "4. Si hay opiniones contradictorias en los fragmentos, menciónalas."
            "OTRAS REGLAS:\n   "
            "1. **Estilo Conciso**: Redacta una respuesta breve que integre los hallazgos. Evita rodeos innecesarios.\n"
            "2. **Contenido**: Si la consulta es sobre una feature química (m/z, RT), explica su posible identificación y bioactividad basándote en la evidencia del contexto. Si es una pregunta teórica, desarrolla una explicación técnica.\n"
            "3. **Uso de Evidencia**: Respalda tus afirmaciones citando las fuentes disponibles en el contexto (ej: 'Según el estudio [Archivo]...').\n"
            "4. **Manejo de Vacíos**: Si el contexto no tiene la respuesta exacta, no digas simplemente 'no hay información'. En su lugar, explica qué información relacionada sí está disponible o resume lo que los documentos mencionan sobre el tema general.\n"
            "5. **Tono**: Científico, preciso y profesional.\n"
            f"Pregunta:\n{question}\n\n"
            f"Contexto:\n{context}\n\n"
            "Respuesta:"
        )

    # -----------------------------
    # OpenAI generation (robusto)
    # -----------------------------
    def _generate_answer_with_retry(self, prompt: str) -> str:
        """
        Reintenta ante fallas transitorias (DNS/red).
        Si la red está inestable, esto hace el demo mucho más robusto.
        """
        last_err = None
        # retries + 1 intento inicial
        for attempt in range(1, self.max_generation_retries + 2):
            try:
                return self._generate_answer(prompt)
            except Exception as e:
                last_err = e
                # backoff exponencial simple
                wait = min(8, 2 ** attempt)
                time.sleep(wait)
        # si todos fallan
        raise last_err

    def _generate_answer(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "Responde de forma concisa, breve y directa."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,  # Opcional: Límite duro de tokens (aprox 300 palabras)
            seed=42,         # <--- Determinismo para respuestas consistentes
        )
        return resp.choices[0].message.content or ""
