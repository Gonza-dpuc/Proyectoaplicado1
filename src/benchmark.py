from pathlib import Path
import json
from statistics import mean

from retriever import retrieve

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
        query = item["query"]
        if "relevant_doc_ids" in item:
            rel_ids = item["relevant_doc_ids"]
        else:
            # Compatibilidad con formato antiguo
            rel_ids = [item["relevant_doc_id"]]

        eval_set.append(
            {
                "query": query,
                "relevant_doc_ids": rel_ids,
            }
        )
    return eval_set


def eval_metrics_at_k(k: int = 5, eval_set=None):
    """
    Calcula métricas de recuperación sobre el eval_set:

    - Precision@k (macro-average)
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

    per_query_results = []

    for item in eval_set:
        query = item["query"]
        relevant_doc_ids = set(item["relevant_doc_ids"])

        docs = retrieve(query, k=k)
        retrieved_doc_ids = [d["metadata"]["doc_id"] for d in docs]

        retrieved_set = set(retrieved_doc_ids)
        hits = len(relevant_doc_ids & retrieved_set)

        num_relevant = len(relevant_doc_ids)
        num_retrieved = len(retrieved_doc_ids)

        precision = hits / num_retrieved if num_retrieved > 0 else 0.0
        recall = hits / num_relevant if num_relevant > 0 else 0.0

        per_query_results.append(
            {
                "query": query,
                "relevant_doc_ids": list(relevant_doc_ids),
                "retrieved_doc_ids": retrieved_doc_ids,
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
