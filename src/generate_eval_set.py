import json
from pathlib import Path
import os
from dotenv import load_dotenv

from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService
from src.retrieval_strategy import get_strategy_from_selection

load_dotenv()

OUTPUT_PATH = Path("data/processed/eval_set.json")

QUERIES = [
    "anthocyanin biosynthesis",
    "actividad antidiabética",
    "flavonoid pathway regulation",
    "platelet aggregation inhibition",
    "anti-inflammatory plant compounds",
    "antioxidant capacity phenolics",
    "metabolomic profiling fruit ripening",
    "biosynthesis of flavonols",
    "antimicrobial activity natural extracts",
    "polyphenol chemical structure",
    "bioactive compounds grape skin",
    "LC-MS metabolite identification",
    "enzyme inhibition plant secondary metabolites",
    "nutraceutical potential berries",
    "postharvest physiological changes metabolites",
    "phenolic content quantification methods",
    "plant defense metabolites",
    "biosynthetic pathway transcription factors",
    "antihypertensive plant metabolites",
    "antidiabetic mechanism polyphenols"
]


def _build_vector_service(collection_name: str) -> VectorStoreService:
    url = os.getenv("QDRANT_URL")
    if url:
        url = url.strip()
    qdrant = QdrantImpl(url=url, api_key=os.getenv(
        "QDRANT_API_KEY"), collection_name=collection_name)
    return VectorStoreService(db_impl=qdrant)


def generate_eval_set(k_retrieve=10, top_relevant=2, collection_name="bioactives_hito1"):
    eval_entries = []

    vector_service = _build_vector_service(collection_name)

    # Baseline: Simple
    retriever = get_strategy_from_selection(
        selection=["Simple"],
        vector_service=vector_service,
        api_key=os.getenv("OPENAI_API_KEY"),
        use_reranker=False,
        use_rrf=False,
    )

    for query in QUERIES:
        print(f"Procesando query: {query}")

        results = retriever.retrieve_context(
            query=query, k=k_retrieve, filters=None)
        if not results:
            print(f"  [WARN] Sin resultados para '{query}', saltando...")
            continue

        # Seleccionar source_file únicos (más estable que chunk_id)
        sources = []
        for r in results:
            src = os.path.basename(r.source_file) if r.source_file else None
            if src and src not in sources:
                sources.append(src)

        relevant = sources[:top_relevant]

        eval_entries.append(
            {
                "query": query,
                # en benchmark se interpretará como “modo estricto” si falta source_file
                "relevant_doc_ids": relevant,
                "source_file": relevant[0] if relevant else None
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=4, ensure_ascii=False)

    print(
        f"\nEval set generado con {len(eval_entries)} queries en {OUTPUT_PATH}")


EVAL_PATH = Path("data/eval/eval_set.json")


def load_eval_set():
    """
    Carga el eval set desde disco.
    Retorna una lista de dicts con:
      - query
      - source_file (o relevant_doc_ids)
    """
    if not EVAL_PATH.exists():
        return []

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    generate_eval_set()
