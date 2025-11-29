import json
from pathlib import Path
from src.retriever import retrieve

OUTPUT_PATH = Path("data/processed/eval_set.json")

# 20 QUERIES REPRESENTATIVAS (se pueden ajustar)
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

def generate_eval_set(k_retrieve=10, top_relevant=2):
    eval_entries = []

    for query in QUERIES:
        print(f"Procesando query: {query}")

        results = retrieve(query, k=k_retrieve)

        if not results:
            print(f"  [WARN] Sin resultados para '{query}', saltando...")
            continue

        # Seleccionar doc_id únicos
        doc_ids = []
        for r in results:
            doc_id = r["metadata"]["doc_id"]
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

        relevant = doc_ids[:top_relevant]  # los top-2 documentos

        eval_entries.append({
            "query": query,
            "relevant_doc_ids": relevant,
        })

    # Guardar JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_entries, f, indent=4, ensure_ascii=False)

    print(f"\nEval set generado con {len(eval_entries)} queries en {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_eval_set()
