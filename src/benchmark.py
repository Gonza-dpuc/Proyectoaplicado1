from pathlib import Path
import json
from statistics import mean
import os
import re
from dotenv import load_dotenv

from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService
from src.retrieval_strategy import get_strategy_from_selection

load_dotenv()

EVAL_PATH = Path("data/processed/eval_set.json")

DEFAULT_EVAL_SET = [
    {"query": "m/z 449.107 actividad antidiabética",
        "relevant_doc_ids": ["myricetin_paper_1"]},
    {"query": "inhibición de agregación plaquetaria myricetin",
        "relevant_doc_ids": ["platelet_aggregation_study"]},
]


def load_eval_set(path: Path = EVAL_PATH):
    if not path.exists():
        print(
            f"[WARN] No se encontró {path}, usando DEFAULT_EVAL_SET en benchmark.py")
        return DEFAULT_EVAL_SET

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    eval_set = []
    for item in raw:
        query = item.get("query") or item.get("question")
        if not query:
            continue

        if "relevant_doc_ids" in item:
            rel_ids = item["relevant_doc_ids"]
        elif "source_chunk_id" in item:
            rel_ids = [item["source_chunk_id"]]
        else:
            rel_ids = [item["relevant_doc_id"]
                       ] if "relevant_doc_id" in item else []

        eval_set.append(
            {
                "query": query,
                "relevant_doc_ids": rel_ids,
                "ground_truth": item.get("ground_truth"),
                "source_file": item.get("source_file"),
            }
        )

    return eval_set


def extract_filters_from_query(query: str) -> dict:
    filters = {}
    mz_match = re.search(
        r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", query, re.IGNORECASE)
    rt_match = re.search(
        r"\b(rt|retention time)\b\s*[:=]?\s*(\d+\.?\d*)", query, re.IGNORECASE)

    if mz_match:
        try:
            mz_val = float(mz_match.group(1))
            tol = 0.05
            filters["mz"] = {"gte": mz_val - tol, "lte": mz_val + tol}
        except ValueError:
            pass

    if rt_match:
        try:
            rt_val = float(rt_match.group(1))
            tol = 0.2
            filters["rt"] = {"gte": rt_val - tol, "lte": rt_val + tol}
        except ValueError:
            pass

    return filters


def _build_vector_service(collection_name: str) -> VectorStoreService:
    url = os.getenv("QDRANT_URL")
    if url:
        url = url.strip()

    qdrant = QdrantImpl(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=collection_name,
    )
    return VectorStoreService(db_impl=qdrant)


def _build_retriever(
    collection_name: str,
    strategies_selection: list,
    use_reranker: bool,
    use_rrf: bool,
):
    vector_service = _build_vector_service(collection_name)
    return get_strategy_from_selection(
        selection=strategies_selection,
        vector_service=vector_service,
        api_key=os.getenv("OPENAI_API_KEY"),
        use_reranker=use_reranker,
        use_rrf=use_rrf,
    )


def eval_metrics_at_k(
    k: int = 5,
    eval_set=None,
    collection_name: str = "bioactives_hito1",
    strategies_selection: list = None,
    use_self_query: bool = False,
    use_reranker: bool = False,
    use_rrf: bool = True,
):
    """
    Evalúa retrieval con UNA interfaz: retriever.retrieve_context(query, k, filters)

    - Precision@k (macro)
    - Recall@k (macro)
    """
    if eval_set is None:
        eval_set = load_eval_set()

    # Baseline: sin estrategias => simple
    if not strategies_selection:
        strategies_selection = ["Simple"]

    retriever = _build_retriever(
        collection_name=collection_name,
        strategies_selection=strategies_selection,
        use_reranker=use_reranker,
        use_rrf=use_rrf,
    )

    per_query_results = []

    for item in eval_set:
        query = item["query"]
        target_source = item.get("source_file")

        # “Modo justo” recomendado: comparar por nombre de archivo cuando exista
        if target_source:
            relevant_set = {os.path.basename(target_source)}
        else:
            relevant_set = set(item["relevant_doc_ids"])

        filters = extract_filters_from_query(query) if use_self_query else None

        # ✅ Interfaz única
        docs = retriever.retrieve_context(query=query, k=k, filters=filters)

        retrieved_set = set()
        for d in docs:
            if target_source:
                if d.source_file:
                    retrieved_set.add(os.path.basename(d.source_file))
            else:
                if d.chunk_id:
                    retrieved_set.add(d.chunk_id)

        hits = len(relevant_set & retrieved_set)
        num_relevant = len(relevant_set)
        num_retrieved = len(retrieved_set)

        precision = hits / num_retrieved if num_retrieved > 0 else 0.0
        recall = hits / num_relevant if num_relevant > 0 else 0.0

        per_query_results.append(
            {
                "query": query,
                "relevant_doc_ids": list(relevant_set),
                "retrieved_doc_ids": list(retrieved_set),
                "num_relevant": num_relevant,
                "num_retrieved": num_retrieved,
                "num_hits": hits,
                "precision": precision,
                "recall": recall,
            }
        )

    macro_precision = mean(r["precision"] for r in per_query_results)
    macro_recall = mean(r["recall"] for r in per_query_results)

    return {
        "precision_at_k": macro_precision,
        "recall_at_k": macro_recall,
        "per_query": per_query_results,
    }
