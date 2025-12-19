# retrieval_strategy.py
from __future__ import annotations
import re
import json

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from openai import OpenAI
from src.models import ProcessedChunk


# --- CONSTANTES DE PROMPTS (Exposed for UI) ---
PROMPT_HYDE = "Redacta un breve documento hipotético que responda idealmente a la consulta del usuario, como si la información ya estuviera disponible. Devuelve SOLO el texto."
PROMPT_STEP_BACK = "Da un paso atrás de la consulta específica y formula una única pregunta general y de alto nivel que ayude a resolver el problema subyacente. Devuelve SOLO la pregunta."
PROMPT_DECOMPOSITION = "Descompón la consulta del usuario en almenos 3 sub-preguntas claras y ordenadas que permitan resolverla paso a paso. Devuelve SOLO las sub-preguntas. Example: [\"q1\", \"q2\"]"


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


def reciprocal_rank_fusion(results_lists: List[List[ProcessedChunk]], k_const: int = 60) -> List[ProcessedChunk]:
    """
    Combina múltiples listas de resultados usando RRF.
    Score = sum(1 / (k_const + rank + 1))
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, ProcessedChunk] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            if doc.chunk_id not in doc_map:
                doc_map[doc.chunk_id] = doc
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + (1.0 / (k_const + rank + 1))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    final_docs = []
    for sid in sorted_ids:
        doc_map[sid].metadata["rrf_score"] = scores[sid]
        final_docs.append(doc_map[sid])
    return final_docs


def generate_hyde_doc(client: OpenAI, query: str, model: str = "gpt-3.5-turbo") -> str:
    """Genera un documento hipotético para HyDE (lógica expuesta)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROMPT_HYDE},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            seed=42  # <--- Determinismo
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return query


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

    def __init__(self, inner: RetrievalStrategy, hyde_generator, use_rrf: bool = False):
        super().__init__(inner)
        self.hyde_generator = hyde_generator  # callable(query)->str
        self.use_rrf = use_rrf

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        hypothetical_doc = self.hyde_generator(query)
        docs_hyde = self.inner.retrieve_context(query=hypothetical_doc, k=k, filters=filters)

        if self.use_rrf:
            # Si RRF está activo, fusionamos HyDE + Query Original
            docs_orig = self.inner.retrieve_context(query=query, k=k, filters=filters)
            return reciprocal_rank_fusion([docs_orig, docs_hyde])[:k]

        return docs_hyde


class StepBackDecorator(RetrievalDecorator):
    """
    Genera una pregunta más abstracta (Step-back), busca para ambas y fusiona resultados.
    """

    def __init__(self, inner: RetrievalStrategy, client: OpenAI, model: str = "gpt-3.5-turbo", use_rrf: bool = False):
        super().__init__(inner)
        self.client = client
        self.model = model
        self.use_rrf = use_rrf

    def generate_step_back(self, query: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_STEP_BACK},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                seed=42  # <--- Determinismo
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return query

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        step_back_q = self.generate_step_back(query)
        
        # Recuperar original y step-back
        docs_orig = self.inner.retrieve_context(query, k, filters)
        if step_back_q and step_back_q != query:
            docs_back = self.inner.retrieve_context(step_back_q, k, filters)
        else:
            docs_back = []

        if self.use_rrf:
            return reciprocal_rank_fusion([docs_orig, docs_back])[:k]

        # Fusión simple (deduplicación manteniendo orden de relevancia original)
        seen = {d.chunk_id for d in docs_orig}
        merged = list(docs_orig)
        for d in docs_back:
            if d.chunk_id not in seen:
                merged.append(d)
                seen.add(d.chunk_id)
        
        return merged[:k]


class DecompositionDecorator(RetrievalDecorator):
    """
    Descompone la query en sub-preguntas, busca para cada una y fusiona.
    """

    def __init__(self, inner: RetrievalStrategy, client: OpenAI, model: str = "gpt-3.5-turbo", use_rrf: bool = False):
        super().__init__(inner)
        self.client = client
        self.model = model
        self.use_rrf = use_rrf

    def decompose(self, query: str) -> List[str]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PROMPT_DECOMPOSITION},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                seed=42  # <--- Determinismo
            )
            txt = resp.choices[0].message.content.strip()
            # Limpieza básica de markdown json
            if "```" in txt:
                txt = txt.split("```")[1].replace("json", "").strip()
            return json.loads(txt)
        except Exception:
            return [query]

    def retrieve_context(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[ProcessedChunk]:
        sub_queries = self.decompose(query)
        if not isinstance(sub_queries, list) or not sub_queries:
            sub_queries = [query]

        if self.use_rrf:
            results_lists = [self.inner.retrieve_context(sq, k, filters) for sq in sub_queries]
            return reciprocal_rank_fusion(results_lists)[:k]

        all_docs = []
        seen = set()
        
        # Buscamos para cada sub-query (fusión simple)
        for sq in sub_queries:
            docs = self.inner.retrieve_context(sq, k, filters)
            for d in docs:
                if d.chunk_id not in seen:
                    seen.add(d.chunk_id)
                    all_docs.append(d)
        
        return all_docs[:k]


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
        
        results = []
        for score, c in scored[:k]:
            c.metadata["rerank_score"] = score
            results.append(c)
        return results

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

    # Cliente OpenAI para las estrategias avanzadas
    client = OpenAI(api_key=api_key)
    model_name = "gpt-3.5-turbo"  # Modelo rápido para retrieval logic

    # 1) Base
    strategy: RetrievalStrategy = HybridSearchStrategy(store=vector_service)

    # 2) Pre-retrieval decorators según selección
    
    # Decomposition
    if "Decomposition" in (selection or []):
        strategy = DecompositionDecorator(strategy, client, model=model_name, use_rrf=use_rrf)

    # Step-back
    if "Step-back" in (selection or []):
        strategy = StepBackDecorator(strategy, client, model=model_name, use_rrf=use_rrf)

    # HyDE
    if "HyDE" in (selection or []):
        def _hyde_generator(q: str) -> str:
            return generate_hyde_doc(client, q, model_name)
        strategy = HyDEDecorator(strategy, hyde_generator=_hyde_generator, use_rrf=use_rrf)

    # 3) Post-retrieval: Reranking
    if use_reranker:
        def _openai_scorer(query: str, text: str) -> float:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict relevance judge. Rate the relevance of the text to the query on a scale from 0.0 (irrelevant) to 1.0 (perfect answer). If the text does not contain the answer, score it below 0.5. Be critical. Return ONLY the float value."},
                        {"role": "user", "content": f"Query: {query}\n\nText: {text[:800]}"}
                    ],
                    max_tokens=10,
                    temperature=0,
                    seed=42  # <--- Determinismo
                )
                val = resp.choices[0].message.content.strip()
                match = re.search(r"(\d+(\.\d+)?)", val)
                return float(match.group(1)) if match else 0.0
            except Exception:
                return 0.0
        
        # top_k=20 para traer candidatos extra antes de rerankear
        strategy = RerankingDecorator(strategy, scorer=_openai_scorer, top_k=20)

    # Adaptador para que Streamlit siga usando .retrieve(...)
    return _StreamlitRetrieveAdapter(strategy)
