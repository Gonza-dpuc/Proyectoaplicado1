# retrieval_strategy.py
from __future__ import annotations
import re

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.models import ProcessedChunk


class RetrievalStrategy(ABC):
    """Interfaz única (Strategy + Decorator-friendly)."""

    @abstractmethod
    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedChunk]:
        raise NotImplementedError


class RetrievalDecorator(RetrievalStrategy):
    """Base Decorator que mantiene la misma interfaz."""

    def __init__(self, inner: RetrievalStrategy):
        self.inner = inner

    def retrieve_context(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ProcessedChunk]:
        return self.inner.retrieve_context(query=query, k=k, filters=filters)


# -------------------------
# PRE-RETRIEVAL DECORATORS
# -------------------------


class SelfQueryDecorator(RetrievalDecorator):
    """
    Pre-retrieval: extrae filtros numéricos desde la query.
    (En vez de tener esto dentro de RAGService).
    """

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

        mz_match = re.search(
            r"(m\/z|mz)\s*[:=]?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)
        rt_match = re.search(
            r"\b(rt|retention time)\b\s*[:=]?\s*(\d+(?:\.\d+)?)", query, re.IGNORECASE)

        if mz_match:
            mz = float(mz_match.group(2))
            # ejemplo simple: rango estrecho; ajusta si tu pauta pide otro margen
            filters["mz"] = {"gte": mz - 0.01, "lte": mz + 0.01}

        if rt_match:
            rt = float(rt_match.group(2))
            filters["rt"] = {"gte": rt - 0.2, "lte": rt + 0.2}

        return filters

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        merged = dict(filters or {})
        merged.update(self._extract_filters(query))
        return self.inner.retrieve_context(query=query, k=k, filters=merged)


# Si ya tenías HyDE/StepBack/Decomposition, conviene que TODOS hereden de RetrievalDecorator.
# A continuación dejo un HyDE "skeleton" para que lo enchufes a tu generador real.

class HyDEDecorator(RetrievalDecorator):
    """
    Pre-retrieval: genera documento hipotético y consulta con ese texto.
    Requiere que el inner sea robusto a query larga.
    """

    def __init__(self, inner: RetrievalStrategy, hyde_generator):
        super().__init__(inner)
        self.hyde_generator = hyde_generator  # callable(query)->str

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        hypothetical_doc = self.hyde_generator(query)
        return self.inner.retrieve_context(query=hypothetical_doc, k=k, filters=filters)


# -------------------------
# POST-RETRIEVAL DECORATORS
# -------------------------

class ContextRepackerDecorator(RetrievalDecorator):
    """
    Post-retrieval: reordena/agrupa contexto (primacía/recencia) SIN cambiar el set recuperado.
    """

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        chunks = self.inner.retrieve_context(query=query, k=k, filters=filters)
        return self._repack(chunks)

    def _repack(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        # Ejemplo: agrupar por source_file y concatenar por grupos
        # (Mantenerlo simple y determinista)
        by_source: Dict[str, List[ProcessedChunk]] = {}
        for c in chunks:
            src = c.source_file or "unknown"
            by_source.setdefault(src, []).append(c)

        repacked: List[ProcessedChunk] = []
        for src, group in by_source.items():
            # deja los chunks del mismo doc juntos
            repacked.extend(group)

        return repacked


class RerankingDecorator(RetrievalDecorator):
    """
    Post-retrieval: rerank con un scorer externo (cross-encoder o LLM scorer).
    scorer(query, chunk_text)->float
    """

    def __init__(self, inner: RetrievalStrategy, scorer, top_k: int):
        super().__init__(inner)
        self.scorer = scorer
        self.top_k = top_k

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        # Pedimos un pool más grande y luego reducimos
        pool = self.inner.retrieve_context(
            query=query, k=max(k, self.top_k), filters=filters)
        scored = [(self.scorer(query, c.content), c) for c in pool]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

# ---- Factory para Streamlit (COMPAT) ----

# ---- Adapter para compatibilidad con Streamlit actual (usa .retrieve) ----


class _StreamlitRetrieveAdapter:
    """
    Streamlit legacy espera .retrieve(query, top_k, filters).
    Tu arquitectura nueva usa .retrieve_context(query, k, filters).
    Este adaptador evita romper la app mientras migras.
    """

    def __init__(self, strategy: RetrievalStrategy):
        self._strategy = strategy

    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None):
        return self._strategy.retrieve_context(query=query, k=top_k, filters=filters)


def get_strategy_from_selection(
    selection,
    vector_service,
    api_key: str,
    use_reranker: bool = False,
    use_rrf: bool = True,
):
    """
    Factory REAL basada en tus clases existentes.
    Retorna un objeto con .retrieve(...) para compatibilidad con app_streamlit.py.
    (Internamente usa retrieve_context + Decorators).
    """

    # Estrategia base real: HybridSearchStrategy (ya la tienes en src/hybrid_search_strategy.py)
    from src.hybrid_search_strategy import HybridSearchStrategy

    # HyDE generator (si no tienes uno, degradamos a passthrough)
    def _hyde_passthrough(q: str) -> str:
        # Si aún no implementas el generador real, no inventamos: devolvemos query.
        return q

    # 1) Base
    strategy: RetrievalStrategy = HybridSearchStrategy(store=vector_service)

    # 2) Pre-retrieval decorators según selección
    # Self-Query lo gestiona RAGService en tu app actual, así que aquí no lo activamos por defecto.
    # (Si quieres moverlo totalmente a Decorator, luego cambiamos RAGService.)
    if "HyDE" in (selection or []):
        strategy = HyDEDecorator(strategy, hyde_generator=_hyde_passthrough)

    # Step-back / Decomposition: tú NO los tienes implementados como Decorators aquí.
    # Para no inventar código, los dejamos como "no-op" (no hacen nada).
    # Si luego me pasas tus implementaciones reales, los conectamos.

    # 3) Post-retrieval: repacking lo controla tu RAGService actual.
    # Si quieres que sea decorador puro, puedes envolver con ContextRepackerDecorator aquí.

    # 4) Reranking/RRF: tú no tienes implementaciones reales en este archivo.
    # No las activamos (evita imports rotos). Cuando tengas esas clases, lo reactivamos.

    # Adaptador para que Streamlit siga usando .retrieve(...)
    return _StreamlitRetrieveAdapter(strategy)
