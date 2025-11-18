from retriever import retrieve

EVAL_SET = [
    {
        "query": "m/z 449.107 actividad antidiabética",
        "relevant_doc_id": "myricetin_paper_1"
    },
    {
        "query": "inhibición de agregación plaquetaria myricetin",
        "relevant_doc_id": "platelet_aggregation_study"
    },
]


def eval_precision_at_k(k=5):
    hits = 0
    total = len(EVAL_SET)
    for item in EVAL_SET:
        docs = retrieve(item["query"], k=k)
        retrieved_ids = [d["metadata"]["doc_id"] for d in docs]
        if item["relevant_doc_id"] in retrieved_ids:
            hits += 1
    return hits / total
