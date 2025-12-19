from src.retrieval_strategy import (
    get_strategy_from_selection, 
    RetrievalStrategy, 
    generate_hyde_doc, 
    StepBackDecorator, 
    DecompositionDecorator,
    PROMPT_HYDE,
    PROMPT_STEP_BACK,
    PROMPT_DECOMPOSITION
)
from src.rag_service import RAGService
from src.vector_store_service import VectorStoreService
from src.qdrant_impl import QdrantImpl
from src.ingestion.embedders import OpenAIAdapter
from src.ingestion.factory import DocumentLoaderFactory
from qdrant_client import models
from openai import OpenAI
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

if not QDRANT_URL:
    st.error("⚠️ Error de Configuración: La variable `QDRANT_URL` no está definida en el archivo .env. La conexión a la base de datos fallará.")

def show_qdrant_error(e: Exception):
    """Muestra un mensaje de error amigable para problemas de conexión con Qdrant."""
    err_msg = str(e)
    # Detectar errores de DNS, conexión rechazada o timeouts
    if "getaddrinfo failed" in err_msg or "No se pudo conectar a Qdrant" in err_msg or "Connection refused" in err_msg:
        st.error(
            f"❌ **Error de Conexión con Qdrant**\n\n"
            f"No se pudo conectar a: `{QDRANT_URL}`\n\n"
            "**Posibles soluciones:**\n"
            "1. **Puerto:** Intenta agregar `:6333` al final de la URL en el `.env` (ej. `...qdrant.io:6333`).\n"
            "2. **Cluster:** Verifica en Qdrant Cloud que el cluster esté 'Healthy'.\n"
            "3. **Red:** Revisa tu conexión a internet o VPN.\n\n"
            f"**Detalle técnico:** `{err_msg}`"
        )
    else:
        st.error(f"❌ Error: {e}")


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
def get_services(collection_name: str, url: str, api_key: Optional[str]):
    """
    Retorna VectorStoreService para una colección dada.
    """
    db = QdrantImpl(
        url=url,
        api_key=api_key,
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

    svc = get_services(collection_name, QDRANT_URL, QDRANT_API_KEY)

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
        # FIX: Asegurar que los IDs sean strings para comparar correctamente
        relevant_set = {str(x) for x in item.get("relevant_doc_ids", [])}

        docs = retriever.retrieve(query=query, top_k=top_k, filters=None)

        ranked_doc_ids = []
        for d in docs:
            # docs normalmente son ProcessedChunk
            doc_id = getattr(d, "doc_id", None)
            if doc_id is not None:
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
            svc = get_services(selected_collection, QDRANT_URL, QDRANT_API_KEY)

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
                        show_qdrant_error(e)
            return

        # -------------------------
        # MODO COMPARACIÓN
        # -------------------------
        svc_h1 = get_services("bioactives_hito1", QDRANT_URL, QDRANT_API_KEY)
        svc_h2 = get_services("bioactives_hito2", QDRANT_URL, QDRANT_API_KEY)

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
                        generate_response=False
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
                        st.info("Baseline: Solo Retrieval (Chunks)")
                        for i, d in enumerate(res_h1.get("context_used", [])[:top_k], start=1):
                            src = getattr(d, "source_file", "unknown")
                            content = getattr(d, "content", "")
                            with st.expander(f"[{i}] {src}"):
                                st.markdown(content)

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
                    show_qdrant_error(e)


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
            try:
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
            except Exception as e:
                show_qdrant_error(e)
                return

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


def page_ingest():
    st.title("📥 Ingesta de Documentos (Prod)")
    st.markdown("---")

    DATA_DIR = Path("data/prod_raw")

    # Permitir cambiar el nombre de la colección (por defecto Hito 2)
    prod_collection_name = st.text_input("Colección destino", value="bioactives_hito2")

    st.info(
        f"**Carpeta de origen:** `{DATA_DIR.absolute()}`"
    )

    if not DATA_DIR.exists():
        st.error(f"❌ No existe el directorio: `{DATA_DIR}`")
        return

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    count_files = len(pdf_files)
    
    c1, c2 = st.columns(2)
    c1.metric("Documentos encontrados", count_files)
    c2.metric("Estado", "Listo" if count_files > 0 else "Sin archivos")

    if count_files == 0:
        st.warning("Agrega archivos PDF en la carpeta `data/prod_raw` para continuar.")
        return

    if st.button("▶️ Correr Ingesta"):
        st.write("Iniciando proceso...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Inicializar servicios
            openai_adapter = OpenAIAdapter()
            db_impl = QdrantImpl(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                collection_name=prod_collection_name
            )
            vector_service = VectorStoreService(db_impl=db_impl)
            client = db_impl.client

            batch_size = 5
            processed_count = 0
            skipped_count = 0

            for i in range(0, count_files, batch_size):
                batch = pdf_files[i : i + batch_size]
                batch_chunks = []
                
                status_text.text(f"Procesando lote {i//batch_size + 1}...")

                for file_path in batch:
                    filename = file_path.name
                    
                    # Verificar duplicados
                    try:
                        count_res = client.count(
                            collection_name=prod_collection_name,
                            count_filter=models.Filter(
                                must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=filename))]
                            )
                        )
                        if count_res.count > 0:
                            skipped_count += 1
                            continue
                    except Exception:
                        pass # Colección no existe aún

                    # Cargar y procesar
                    loader = DocumentLoaderFactory.get_loader(str(file_path))
                    chunks = loader.load_and_chunk(str(file_path))
                    for c in chunks:
                        c.source_file = filename
                    batch_chunks.extend(chunks)

                if batch_chunks:
                    vectors = openai_adapter.embed_chunks(batch_chunks)
                    for idx, c in enumerate(batch_chunks):
                        c.dense_vector = vectors[idx]
                    vector_service.index_chunks(batch_chunks)
                    processed_count += len(batch)

                progress_bar.progress(min((i + batch_size) / count_files, 1.0))

            progress_bar.progress(1.0)
            st.success(f"✅ Proceso finalizado. Nuevos: {processed_count}, Saltados (duplicados): {skipped_count}")

        except Exception as e:
            show_qdrant_error(e)


class DummyStrategy(RetrievalStrategy):
    """Estrategia vacía para instanciar decoradores en modo debug."""
    def retrieve_context(self, query, k=5, filters=None):
        return []


def page_retrieval_debug():
    st.title("🔍Recuperación")
    st.caption("Visualiza cómo las estrategias transforman tu query antes de buscar.")

    if not OPENAI_API_KEY:
        st.warning("⚠️ Se requiere OPENAI_API_KEY para generar las transformaciones.")
        return

    query = st.text_input("Query de prueba", value="¿Qué efectos tiene el maqui en la inflamación?")
    
    # Selección de estrategia
    strategy_type = st.selectbox("Estrategia a analizar", ["Simple (Direct)", "HyDE", "Step-back", "Decomposition"])

    if st.button("Generar Transformación"):
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with st.spinner(f"Ejecutando lógica de {strategy_type}..."):
            try:
                if strategy_type == "Simple (Direct)":
                    st.info("ℹ️ La estrategia **Simple** no transforma la query. Se envía directamente al buscador vectorial.")

                elif strategy_type == "HyDE":
                    res = generate_hyde_doc(client, query)
                    with st.expander("🤖 Ver Prompt del Agente"):
                        st.code(PROMPT_HYDE, language="text")
                    st.subheader("📄 Documento Hipotético (HyDE)")
                    st.info(res)
                    st.caption("El sistema buscará documentos similares a este texto generado, no a tu pregunta original.")
                    
                elif strategy_type == "Step-back":
                    # Instanciamos el decorador con una estrategia dummy solo para acceder al método
                    decorator = StepBackDecorator(DummyStrategy(), client)
                    res = decorator.generate_step_back(query)
                    with st.expander("🤖 Ver Prompt del Agente"):
                        st.code(PROMPT_STEP_BACK, language="text")
                    st.subheader("🔙 Pregunta Step-back (Abstracta)")
                    st.info(res)
                    st.caption("El sistema buscará tanto tu pregunta original como esta pregunta abstracta.")
                    
                elif strategy_type == "Decomposition":
                    decorator = DecompositionDecorator(DummyStrategy(), client)
                    res = decorator.decompose(query)
                    with st.expander("🤖 Ver Prompt del Agente"):
                        st.code(PROMPT_DECOMPOSITION, language="text")
                    st.subheader("🧩 Sub-preguntas (Decomposition)")
                    st.write(res)
                    st.caption("El sistema realizará una búsqueda independiente para cada una de estas sub-preguntas.")

            except Exception as e:
                st.error(f"Error generando transformación: {e}")

    st.markdown("---")
    st.subheader("🕵️ Simulación de Retrieval (Top 5)")
    st.caption("Ejecuta la búsqueda real usando la estrategia seleccionada arriba + opciones extra.")

    with st.expander("ℹ️ ¿Qué significan los puntajes?"):
        st.markdown("""
        **Rerank Score (0.0 - 1.0):** 
        Es un puntaje de relevancia semántica asignado por un LLM (o Cross-Encoder) que evalúa específicamente qué tan bien responde el texto a la pregunta. Un valor cercano a 1.0 indica alta relevancia.
        
        **RRF Score (Reciprocal Rank Fusion):**
        Es un puntaje calculado al combinar múltiples listas de resultados (ej. búsqueda original + búsqueda HyDE). 
        Fórmula: `sum(1 / (k + rank_i))`. Premia a los documentos que aparecen consistentemente en los primeros lugares de diferentes estrategias de búsqueda.
        """)

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_collection = st.selectbox("Colección", ["bioactives_hito2", "bioactives_hito1"], index=0, key="sim_col")
    with col_sim2:
        sim_use_reranker = st.checkbox("Activar Reranker", value=True, key="sim_rerank")
        sim_use_rrf = st.checkbox("Activar RRF", value=True, key="sim_rrf")

    if st.button("🔍 Buscar y Rankear"):
        svc = get_services(sim_collection, QDRANT_URL, QDRANT_API_KEY)
        
        # Usamos la estrategia seleccionada en el selectbox de arriba
        selection = []
        if strategy_type and strategy_type != "Simple (Direct)":
            selection = [strategy_type]
        
        with st.spinner("Recuperando documentos..."):
            try:
                strategy = get_strategy_from_selection(
                    selection=selection,
                    vector_service=svc,
                    api_key=OPENAI_API_KEY,
                    use_reranker=sim_use_reranker,
                    use_rrf=sim_use_rrf
                )
                
                results = strategy.retrieve(query, top_k=5)
                
                if not results:
                    st.warning("No se encontraron resultados.")
                else:
                    for i, doc in enumerate(results, 1):
                        scores_info = []
                        if "rerank_score" in doc.metadata:
                            scores_info.append(f"Rerank: {doc.metadata['rerank_score']:.4f}")
                        if "rrf_score" in doc.metadata:
                            scores_info.append(f"RRF: {doc.metadata['rrf_score']:.4f}")
                        
                        score_label = f" ({' | '.join(scores_info)})" if scores_info else ""
                        
                        with st.expander(f"#{i} {doc.source_file}{score_label}"):
                            st.markdown(f"**Content:** {doc.content}")
                            st.json(doc.metadata)
                            
            except Exception as e:
                show_qdrant_error(e)


def page_metrics_compare():
    st.title("📈 Comparación de Métricas (Académica)")
    st.caption(
        "Precision/Recall + HitRate + MRR (macro), comparando doc_id.")

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
            try:
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
            except Exception as e:
                show_qdrant_error(e)
                return

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


def page_generation_debug():
    st.title("🧠 Generación")
    st.caption("Inspecciona el contexto recuperado, el prompt construido y la respuesta final.")

    if not OPENAI_API_KEY:
        st.warning("⚠️ Se requiere OPENAI_API_KEY.")
        return

    col1, col2 = st.columns(2)
    with col1:
        collection = st.selectbox("Colección", ["bioactives_hito2", "bioactives_hito1"], index=0, key="gen_col")
    with col2:
        top_k = st.number_input("Top K Contexto", min_value=1, max_value=20, value=5, key="gen_k")

    query = st.text_area("Query", value="¿Qué efectos tiene el maqui en la inflamación?")

    with st.expander("Configuración de Retrieval"):
        strategies = st.multiselect("Estrategias", ["Simple", "HyDE", "Step-back", "Decomposition"], default=["Simple"], key="gen_strat")
        use_reranker = st.checkbox("Reranking", value=True, key="gen_rerank")
        use_rrf = st.checkbox("RRF", value=True, key="gen_rrf")

    if st.button("🚀 Ejecutar Generación"):
        svc = get_services(collection, QDRANT_URL, QDRANT_API_KEY)
        strategy = get_strategy_from_selection(
            selection=strategies,
            vector_service=svc,
            api_key=OPENAI_API_KEY,
            use_reranker=use_reranker,
            use_rrf=use_rrf
        )
        rag = RAGService(strategy, OPENAI_API_KEY)

        with st.spinner("Generando..."):
            try:
                result = rag.answer_question(
                    query, 
                    top_k=top_k, 
                    use_self_query=True, 
                    use_repacking=True, 
                    generate_response=True
                )
                
                st.markdown("### 1. Contexto Recuperado (Input)")
                context_used = result.get("context_used", [])
                if context_used:
                    for i, doc in enumerate(context_used, 1):
                        with st.expander(f"Chunk {i}: {doc.source_file}"):
                            st.markdown(f"**Content:** {doc.content}")
                            st.json(doc.metadata)
                else:
                    st.warning("No se recuperó contexto.")

                st.markdown("### 2. Prompt Construido")
                prompt_text = result.get("generated_prompt", "No disponible")
                st.code(prompt_text, language="text")

                st.markdown("### 3. Respuesta Final")
                st.success(result.get("answer"))
                
            except Exception as e:
                show_qdrant_error(e)


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
    ["Chat", "Benchmark Básico", "Comparación Métricas", "Ingesta", "Recuperación", "Generación", "Acerca de"],
    index=0
)

if page == "Chat":
    page_chat()
elif page == "Benchmark Básico":
    page_benchmark_basic()
elif page == "Comparación Métricas":
    page_metrics_compare()
elif page == "Ingesta":
    page_ingest()
elif page == "Recuperación":
    page_retrieval_debug()
elif page == "Generación":
    page_generation_debug()
else:
    page_about()
