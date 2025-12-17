from pathlib import Path
import json
from statistics import mean
import os
from src.retriever import BioactivesRetriever
from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService
from src.retrieval_strategy import get_strategy_from_selection
from dotenv import load_dotenv
import re

load_dotenv()

# Ruta por defecto del conjunto de evaluación externo
EVAL_PATH = Path("data/processed/eval_set.json")

# Eval set embebido como fallback, por compatibilidad
DEFAULT_EVAL_SET = [
    {
        "query": "m/z 449.107 actividad antidiabética",
        "relevant_doc_ids": ["myricetin_paper_1"],
    },
    {
        "query": "inhibición de agregación plaquetaria myricetin",
        "relevant_doc_ids": ["platelet_aggregation_study"],
    },
]


def load_eval_set(path: Path = EVAL_PATH):
    """
    Carga el conjunto de evaluación desde JSON.
    Formato esperado (lista de objetos):
    [
        {
            "query": "texto de la consulta",
            "relevant_doc_ids": ["id_doc1", "id_doc2", ...]
        },
        ...
    ]
    """
    if not path.exists():
        # Fallback al eval set embebido
        print(
            f"[WARN] No se encontró {path}, usando DEFAULT_EVAL_SET en benchmark.py"
        )
        return DEFAULT_EVAL_SET

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    eval_set = []
    for item in raw:
        # Adaptador: El dataset de Gemini usa "question", el legacy usa "query"
        query = item.get("query") or item.get("question")
        if not query:
            continue

        if "relevant_doc_ids" in item:
            rel_ids = item["relevant_doc_ids"]
        elif "source_chunk_id" in item:
            # Compatibilidad con dataset sintético de Gemini
            rel_ids = [item["source_chunk_id"]]
        else:
            # Compatibilidad con formato antiguo
            rel_ids = [item["relevant_doc_id"]] if "relevant_doc_id" in item else []

        eval_set.append({
            "query": query,
            "relevant_doc_ids": rel_ids,
            "ground_truth": item.get("ground_truth"),
            "source_file": item.get("source_file"), # <--- Agregamos el nombre del archivo esperado
        })

    return eval_set


def extract_filters_from_query(query: str) -> dict:
    """Detecta intenciones de búsqueda estructurada (m/z) en lenguaje natural (Self-Query)."""
    filters = {}
    # Regex para capturar m/z con tolerancia
    mz_match = re.search(r"(?:m/z|mz|mass)\s*[:=]?\s*(\d+\.?\d*)", query, re.IGNORECASE)
    
    if mz_match:
        try:
            mz_val = float(mz_match.group(1))
            tolerance = 0.05 
            filters["mz"] = {"gte": mz_val - tolerance, "lte": mz_val + tolerance}
        except ValueError:
            pass
    return filters

def eval_metrics_at_k(k: int = 5, eval_set=None, collection_name: str = "bioactives_prod", use_advanced_strategies: bool = False, strategies_selection: list = None, use_self_query: bool = False, use_reranker: bool = False, use_rrf: bool = True):
    """
    Calcula métricas de recuperación sobre el eval_set:

    - Recall@k (macro-average)
    - Detalle por query (para depuración o visualización en Streamlit)

    Devuelve un diccionario:
    {
        "precision_at_k": float,
        "recall_at_k": float,
        "per_query": [
            {
                "query": str,
                "relevant_doc_ids": [str, ...],
                "retrieved_doc_ids": [str, ...],
                "num_relevant": int,
                "num_retrieved": int,
                "num_hits": int,
                "precision": float,
                "recall": float,
            },
            ...
        ]
    }
    """
    if eval_set is None:
        eval_set = load_eval_set()

    # Determinar si usamos modo avanzado (si hay flag o lista de estrategias)
    is_advanced = use_advanced_strategies or (strategies_selection is not None and len(strategies_selection) > 0)

    # Configurar el Retriever (Simple vs Avanzado)
    if is_advanced:
        # Usamos el stack completo del Hito 2 (HyDE, etc.)
        url = os.getenv("QDRANT_URL")
        if url: url = url.strip()
        
        qdrant = QdrantImpl(
            collection_name=collection_name,
            url=url,
            api_key=os.getenv("QDRANT_API_KEY")
        )
        vector_service = VectorStoreService(db_impl=qdrant)
        
        # Si no se especifica lista, usamos HyDE por defecto
        selection = strategies_selection if strategies_selection is not None else ["HyDE"]
        
        retriever = get_strategy_from_selection(
            selection=selection, 
            vector_service=vector_service, 
            api_key=os.getenv("OPENAI_API_KEY"),
            use_reranker=use_reranker,
            use_rrf=use_rrf
        )
    else:
        # Usamos búsqueda simple (Hito 1 o Baseline)
        retriever = BioactivesRetriever(collection_name=collection_name)

    per_query_results = []

    for item in eval_set:
        query = item["query"]
        
        # Lógica de Comparación: Documento (Nombre) vs Chunk (ID)
        target_source = item.get("source_file")
        
        if target_source:
            # Modo Justo: Comparamos nombres de archivo (para Hito 1 vs Hito 2)
            relevant_set = {os.path.basename(target_source)}
        else:
            # Modo Estricto: Comparamos IDs de chunks (Legacy)
            relevant_set = set(item["relevant_doc_ids"])

        # Lógica Self-Query (Filtros)
        filters = None
        if use_self_query:
            filters = extract_filters_from_query(query)

        # Ejecutar búsqueda usando la clase BioactivesRetriever
        if is_advanced:
            # Las estrategias usan el método .retrieve()
            docs = retriever.retrieve(query, top_k=k, filters=filters)
        else:
            # BioactivesRetriever usa .search()
            docs = retriever.search(query, limit=k)
        
        retrieved_set = set()
        for d in docs:
            if target_source:
                # Extraer nombre de archivo
                val = d.source_file if hasattr(d, "source_file") else d.get("source_file")
                if val: retrieved_set.add(os.path.basename(val))
            else:
                # Extraer ID
                val = d.chunk_id if hasattr(d, "chunk_id") else d.get("chunk_id")
                if val: retrieved_set.add(val)

        hits = len(relevant_set & retrieved_set)
        num_relevant = len(relevant_set)
        num_retrieved = len(retrieved_set) # Ojo: esto cuenta documentos únicos recuperados

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


# Funciones de conveniencia para mantener compatibilidad con la app
def eval_precision_at_k(k: int = 5) -> float:
    """
    Devuelve Precision@k promedio sobre el conjunto de evaluación.
    Mantiene la misma firma que la versión original.
    """
    metrics = eval_metrics_at_k(k=k)
    return metrics["precision_at_k"]


def eval_recall_at_k(k: int = 5) -> float:
    """
    Devuelve Recall@k promedio sobre el conjunto de evaluación.
    """
    metrics = eval_metrics_at_k(k=k)
    return metrics["recall_at_k"]


def eval_details_at_k(k: int = 5):
    """
    Devuelve el detalle por query, útil para visualizar en Streamlit.
    """
    metrics = eval_metrics_at_k(k=k)
    return metrics["per_query"]


if __name__ == "__main__":
    k_test = 5
    metrics = eval_metrics_at_k(k=k_test)
    print(f"Precision@{k_test}: {metrics['precision_at_k']:.3f}")
    print(f"Recall@{k_test}:    {metrics['recall_at_k']:.3f}")
    print("Ejemplo de detalle por query:")
    for row in metrics["per_query"]:
        print(
            f"- Query: {row['query']!r} | hits={row['num_hits']} "
            f"| P={row['precision']:.2f} | R={row['recall']:.2f}"
        )
