from src.retrieval_strategy import get_strategy_from_selection
from src.rag_service import RAGService
from src.vector_store_service import VectorStoreService
from src.qdrant_impl import QdrantImpl
import streamlit as st
from typing import Dict, Any, List, Optional
from statistics import mean
from pathlib import Path
import json
import os
from dotenv import load_dotenv
load_dotenv()


# =========================
# Config general
# =========================
st.set_page_config(
    page_title="Bioactives RAG",
    page_icon="🧪",
    layout="wide",
)

QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip() or None
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()


# =========================
# Helpers: eval set
# =========================
EVAL_PATH_CANDIDATES = [
    Path("data/processed/eval_set.json"),
    Path("data/eval/eval_set.json"),
    Path("eval_set.json"),
]


def load_eval_set() -> List[Dict[str, Any]]:
    """
    Carga eval_set.json con estructura tipo:
      {"query": "...", "relevant_doc_ids": ["id1","id2", ...], ...}
    """
    path = next((p for p in EVAL_PATH_CANDIDATES if p.exists()), None)
    if path is None:
        return []

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    eval_set = []
    for item in raw:
        q = item.get("query") or item.get("question")
        if not q:
            continue
        rel = item.get("relevant_doc_ids") or []
        if isinstance(rel, str):
            rel = [rel]
        eval_set.append({"query": q, "relevant_doc_ids": rel})
    return eval_set


# =========================
# Helpers: services
# =========================
@st.cache_resource
def get_services(collection_name: str):
    """
    Retorna VectorStoreService para una colección dada.
    """
    db = QdrantImpl(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
    )
    svc = VectorStoreService(db_impl=db)
    return svc


# =========================
# Métricas
# =========================
def reciprocal_rank(relevant_set: set, ranked_list: list) -> float:
    for i, item in enumerate(ranked_list, start=1):
        if item in relevant_set:
            return 1.0 / i
    return 0.0


def run_retrieval_eval_plus(
    collection_name: str,
    strategies: list,
    top_k: int,
    use_self_query: bool,
    use_reranker: bool,
    use_rrf: bool,
):
    """
    Evalúa retrieval comparando por doc_id vs relevant_doc_ids del eval_set.json.
    Retorna macro:
      - Precision@k
      - Recall@k
      - HitRate@k
      - MRR@k
    """
    eval_set = load_eval_set()
    if not eval_set:
        return None

    svc = get_services(collection_name)

    retriever = get_strategy_from_selection(
        selection=strategies,
        vector_service=svc,
        api_key=OPENAI_API_KEY,
        use_reranker=use_reranker,
        use_rrf=use_rrf,
    )

    rows = []

    # Nota: self-query lo aplica tu RAGService en chat; en benchmark lo dejamos OFF/ON según flag,
    # pero como tu retriever actual es adapter con retrieve(...), aquí no parseamos filtros.
    # (Si quieres aplicar self-query en benchmark, lo incorporamos después.)
    for item in eval_set:
        query = item["query"]
        relevant_set = set(item.get("relevant_doc_ids", []))

        docs = retriever.retrieve(query=query, top_k=top_k, filters=None)

        ranked_doc_ids = []
        for d in docs:
            # docs normalmente son ProcessedChunk
            doc_id = getattr(d, "doc_id", None)
            if doc_id:
                ranked_doc_ids.append(str(doc_id))

        retrieved_set = set(ranked_doc_ids)
        hits = len(relevant_set & retrieved_set)

        precision = hits / len(retrieved_set) if retrieved_set else 0.0
        recall = hits / len(relevant_set) if relevant_set else 0.0
        hitrate = 1.0 if hits > 0 else 0.0
        rr = reciprocal_rank(relevant_set, ranked_doc_ids)

        rows.append(
            {
                "query": query,
                "precision": precision,
                "recall": recall,
                "hitrate": hitrate,
                "rr": rr,
            }
        )

    return {
        "precision_at_k": mean(r["precision"] for r in rows),
        "recall_at_k": mean(r["recall"] for r in rows),
        "hitrate_at_k": mean(r["hitrate"] for r in rows),
        "mrr_at_k": mean(r["rr"] for r in rows),
        "per_query": rows,
    }


# =========================
# UI: páginas
# =========================
def page_chat():
    st.title("💬 Asistente Bio-Actives (RAG)")
    st.caption(
        "Chat normal o comparación Hito 1 vs Hito 2 (respuesta + retrieval).")

    if not OPENAI_API_KEY:
        st.warning(
            "⚠️ Falta OPENAI_API_KEY en tu .env. El chat no podrá generar respuestas.")
        return

    col_left, col_right = st.columns([1, 2])

    with col_left:
        mode = st.radio(
            "Modo",
            ["Normal", "Comparar Hito 1 vs Hito 2"],
            index=0
        )

        if mode == "Normal":
            selected_collection = st.selectbox(
                "Colección",
                ["bioactives_hito1", "bioactives_hito2"],
                index=1
            )
        else:
            st.info(
                "Comparación: Hito 1 usa bioactives_hito1 (Simple). Hito 2 usa bioactives_hito2 (configurable).")

        selected_strategies = st.multiselect(
            "Estrategias (solo afectan Hito 2 en modo comparación)",
            ["Simple", "HyDE", "Step-back", "Decomposition"],
            default=["Simple"] if mode == "Normal" else ["HyDE", "Step-back"]
        )

        top_k = st.slider("Top K", min_value=1, max_value=15, value=5, step=1)

        use_self_query = st.checkbox(
            "Self-Query (filtros mz/rt desde la query)", value=True)
        use_repacking = st.checkbox(
            "Repacking (post-procesar contexto)", value=True)

        use_reranker = st.checkbox("Reranking", value=False)
        use_rrf = st.checkbox("RRF Fusion", value=True)

        response_mode = st.radio(
            "Modo respuesta",
            ["Respuesta Aumentada (RAG)", "Solo Retrieval (mostrar chunks)"],
            index=0
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with col_right:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "compare" in msg:
                    with st.expander("🆚 Comparación"):
                        st.markdown(msg["compare"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📚 Fuentes"):
                        for s in msg["sources"]:
                            st.write(f"- {s}")

        prompt = st.chat_input(
            "Ej: ¿Qué efectos tiene el maqui en la inflamación?")
        if not prompt:
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        generate_response = (response_mode == "Respuesta Aumentada (RAG)")

        # -------------------------
        # MODO NORMAL
        # -------------------------
        if mode == "Normal":
            svc = get_services(selected_collection)

            with st.chat_message("assistant"):
                with st.spinner("Procesando..."):
                    try:
                        strategy = get_strategy_from_selection(
                            selection=selected_strategies,
                            vector_service=svc,
                            api_key=OPENAI_API_KEY,
                            use_reranker=use_reranker,
                            use_rrf=use_rrf
                        )

                        rag = RAGService(strategy, OPENAI_API_KEY)

                        result = rag.answer_question(
                            prompt,
                            top_k=top_k,
                            use_self_query=use_self_query,
                            use_repacking=use_repacking,
                            generate_response=generate_response
                        )

                        answer = result.get("answer", "")
                        sources = result.get("sources", [])
                        context_used = result.get("context_used", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 Fuentes"):
                                for s in sources:
                                    st.write(f"- {s}")

                        if not generate_response:
                            with st.expander("🔎 Contexto recuperado (debug)"):
                                for i, d in enumerate(context_used[:top_k], start=1):
                                    src = getattr(d, "source_file", "unknown")
                                    content = getattr(d, "content", "")
                                    st.write(f"**[{i}] {src}**")
                                    st.write(
                                        content[:900] + ("..." if len(content) > 900 else ""))

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer,
                                "sources": sources}
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            return

        # -------------------------
        # MODO COMPARACIÓN
        # -------------------------
        svc_h1 = get_services("bioactives_hito1")
        svc_h2 = get_services("bioactives_hito2")

        with st.chat_message("assistant"):
            with st.spinner("Comparando Hito 1 vs Hito 2..."):
                try:
                    # Hito 1 (baseline fijo)
                    strategy_h1 = get_strategy_from_selection(
                        selection=["Simple"],
                        vector_service=svc_h1,
                        api_key=OPENAI_API_KEY,
                        use_reranker=False,
                        use_rrf=False
                    )
                    rag_h1 = RAGService(strategy_h1, OPENAI_API_KEY)
                    res_h1 = rag_h1.answer_question(
                        prompt,
                        top_k=top_k,
                        use_self_query=False,
                        use_repacking=False,
                        generate_response=generate_response
                    )

                    # Hito 2 (configurable)
                    strategy_h2 = get_strategy_from_selection(
                        selection=selected_strategies or ["Simple"],
                        vector_service=svc_h2,
                        api_key=OPENAI_API_KEY,
                        use_reranker=use_reranker,
                        use_rrf=use_rrf
                    )
                    rag_h2 = RAGService(strategy_h2, OPENAI_API_KEY)
                    res_h2 = rag_h2.answer_question(
                        prompt,
                        top_k=top_k,
                        use_self_query=use_self_query,
                        use_repacking=use_repacking,
                        generate_response=generate_response
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Hito 1 (bioactives_hito1)")
                        st.markdown(res_h1.get("answer", ""))
                        if res_h1.get("sources"):
                            with st.expander("📚 Fuentes H1"):
                                for s in res_h1["sources"]:
                                    st.write(f"- {s}")
                        if not generate_response:
                            with st.expander("🔎 Contexto H1"):
                                for i, d in enumerate(res_h1.get("context_used", [])[:top_k], start=1):
                                    src = getattr(d, "source_file", "unknown")
                                    content = getattr(d, "content", "")
                                    st.write(f"**[{i}] {src}**")
                                    st.write(
                                        content[:900] + ("..." if len(content) > 900 else ""))

                    with c2:
                        st.subheader("Hito 2 (bioactives_hito2)")
                        st.markdown(res_h2.get("answer", ""))
                        if res_h2.get("sources"):
                            with st.expander("📚 Fuentes H2"):
                                for s in res_h2["sources"]:
                                    st.write(f"- {s}")
                        if not generate_response:
                            with st.expander("🔎 Contexto H2"):
                                for i, d in enumerate(res_h2.get("context_used", [])[:top_k], start=1):
                                    src = getattr(d, "source_file", "unknown")
                                    content = getattr(d, "content", "")
                                    st.write(f"**[{i}] {src}**")
                                    st.write(
                                        content[:900] + ("..." if len(content) > 900 else ""))

                    compare_md = (
                        f"**H1 sources:** {len(res_h1.get('sources', []))}  \n"
                        f"**H2 sources:** {len(res_h2.get('sources', []))}"
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "🆚 Comparación completada (ver columnas).",
                            "compare": compare_md,
                            "sources": (res_h2.get("sources") or []),
                        }
                    )

                except Exception as e:
                    st.error(f"❌ Error en comparación: {e}")


def page_benchmark_basic():
    st.title("📊 Benchmark Básico (Hito 1 vs Hito 2)")
    st.caption(
        "Calcula Precision@k y Recall@k (macro) usando eval_set.json por doc_id.")

    eval_set = load_eval_set()
    if not eval_set:
        st.warning(
            "No se encontró eval_set.json. Se busca en: data/processed/, data/eval/ o raíz.")
        return

    top_k = st.slider("Top K", 1, 20, 5, 1)

    st.subheader("Configuración Hito 2 (mejorado)")
    selected_strategies = st.multiselect(
        "Estrategias Hito 2",
        ["Simple", "HyDE", "Step-back", "Decomposition"],
        default=["HyDE", "Step-back"]
    )
    use_reranker_bench = st.checkbox("Reranking (Hito 2)", value=False)
    use_rrf_bench = st.checkbox("RRF (Hito 2)", value=True)

    if st.button("▶️ Ejecutar Benchmark Básico"):
        with st.spinner("Evaluando..."):
            m_h1 = run_retrieval_eval_plus(
                collection_name="bioactives_hito1",
                strategies=["Simple"],
                top_k=top_k,
                use_self_query=False,
                use_reranker=False,
                use_rrf=False,
            )
            m_h2 = run_retrieval_eval_plus(
                collection_name="bioactives_hito2",
                strategies=selected_strategies or ["Simple"],
                top_k=top_k,
                use_self_query=False,
                use_reranker=use_reranker_bench,
                use_rrf=use_rrf_bench,
            )

        if m_h1 is None or m_h2 is None:
            st.error(
                "No se pudo ejecutar el benchmark. Revisa conexión a Qdrant y eval_set.")
            return

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Hito 1 Precision@k", f"{m_h1['precision_at_k']:.3f}")
            st.metric("Hito 1 Recall@k", f"{m_h1['recall_at_k']:.3f}")
        with c2:
            st.metric("Hito 2 Precision@k", f"{m_h2['precision_at_k']:.3f}")
            st.metric("Hito 2 Recall@k", f"{m_h2['recall_at_k']:.3f}")

        st.subheader("Detalle por query (Hito 1)")
        st.dataframe(m_h1["per_query"])

        st.subheader("Detalle por query (Hito 2)")
        st.dataframe(m_h2["per_query"])


def page_metrics_compare():
    st.title("📈 Comparación de Métricas (Académica)")
    st.caption(
        "Recomendado para defensa: Precision/Recall + HitRate + MRR (macro), comparando doc_id.")

    eval_set = load_eval_set()
    if not eval_set:
        st.warning(
            "No se encontró eval_set.json. Se busca en: data/processed/, data/eval/ o raíz.")
        return

    top_k = st.slider("Top K", 1, 20, 5, 1)

    st.subheader("Configuración Hito 2 (mejorado)")
    selected_strategies = st.multiselect(
        "Estrategias Hito 2",
        ["Simple", "HyDE", "Step-back", "Decomposition"],
        default=["HyDE", "Step-back"]
    )
    use_reranker_bench = st.checkbox("Reranking (Hito 2)", value=False)
    use_rrf_bench = st.checkbox("RRF (Hito 2)", value=True)

    if st.button("▶️ Ejecutar comparación de métricas"):
        with st.spinner("Calculando métricas..."):
            m_h1 = run_retrieval_eval_plus(
                collection_name="bioactives_hito1",
                strategies=["Simple"],
                top_k=top_k,
                use_self_query=False,
                use_reranker=False,
                use_rrf=False,
            )

            m_h2 = run_retrieval_eval_plus(
                collection_name="bioactives_hito2",
                strategies=selected_strategies or ["Simple"],
                top_k=top_k,
                use_self_query=False,
                use_reranker=use_reranker_bench,
                use_rrf=use_rrf_bench,
            )

        if not m_h1 or not m_h2:
            st.error(
                "No se pudo ejecutar la evaluación (revisa conexión a Qdrant y eval_set).")
            return

        summary = [
            {"metric": "Precision@k",
                "hito1": m_h1["precision_at_k"], "hito2": m_h2["precision_at_k"]},
            {"metric": "Recall@k",
                "hito1": m_h1["recall_at_k"],    "hito2": m_h2["recall_at_k"]},
            {"metric": "HitRate@k",
                "hito1": m_h1["hitrate_at_k"],   "hito2": m_h2["hitrate_at_k"]},
            {"metric": "MRR@k",
                "hito1": m_h1["mrr_at_k"],       "hito2": m_h2["mrr_at_k"]},
        ]
        for row in summary:
            row["delta(h2-h1)"] = row["hito2"] - row["hito1"]

        st.subheader("Resumen (macro) y mejora relativa")
        st.dataframe(summary)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Hito 1 Precision@k", f"{m_h1['precision_at_k']:.3f}")
            st.metric("Hito 1 Recall@k", f"{m_h1['recall_at_k']:.3f}")
            st.metric("Hito 1 HitRate@k", f"{m_h1['hitrate_at_k']:.3f}")
            st.metric("Hito 1 MRR@k", f"{m_h1['mrr_at_k']:.3f}")
        with c2:
            st.metric("Hito 2 Precision@k", f"{m_h2['precision_at_k']:.3f}")
            st.metric("Hito 2 Recall@k", f"{m_h2['recall_at_k']:.3f}")
            st.metric("Hito 2 HitRate@k", f"{m_h2['hitrate_at_k']:.3f}")
            st.metric("Hito 2 MRR@k", f"{m_h2['mrr_at_k']:.3f}")

        st.subheader("Detalle por query (Hito 1)")
        st.dataframe(m_h1["per_query"])

        st.subheader("Detalle por query (Hito 2)")
        st.dataframe(m_h2["per_query"])


def page_about():
    st.title("ℹ️ Información del sistema")
    st.markdown(
        """
**Colecciones:**
- `bioactives_hito1`: baseline (Hito 1)
- `bioactives_hito2`: mejorado (Hito 2)

**Eval set:**
- Se espera `eval_set.json` con `query` y `relevant_doc_ids`.
- La evaluación se hace por **doc_id** (más robusto que nombre de archivo).

**Comparación:**
- Chat comparativo (H1 vs H2)
- Métricas retrieval: Precision/Recall/HitRate/MRR
"""
    )


# =========================
# Navegación
# =========================
st.sidebar.title("🧭 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["Chat", "Benchmark Básico", "Comparación Métricas", "Acerca de"],
    index=0
)

if page == "Chat":
    page_chat()
elif page == "Benchmark Básico":
    page_benchmark_basic()
elif page == "Comparación Métricas":
    page_metrics_compare()
else:
    page_about()
