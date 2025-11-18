import streamlit as st
from retriever import retrieve
from benchmark import eval_precision_at_k, eval_recall_at_k

st.set_page_config(page_title="BioActives RAG Baseline", layout="wide")

st.title("BioActives RAG – Baseline (Grupo 5)")

# Sidebar: métricas de rendimiento
st.sidebar.header("Métricas de rendimiento (Baseline)")

k_eval = st.sidebar.slider("k para métricas @k", 1, 20, 5)

# Calculamos siempre las métricas para el k actual
with st.sidebar.expander("Resultados actuales de evaluación", expanded=True):
    try:
        p_at_k = eval_precision_at_k(k_eval)
        r_at_k = eval_recall_at_k(k_eval)
        st.write(f"**Precision@{k_eval}:** {p_at_k:.2f}")
        st.write(f"**Recall@{k_eval}:** {r_at_k:.2f}")
        st.caption(
            "Precision@k: de los documentos recuperados, cuántos eran relevantes.\n\n"
            "Recall@k: de los documentos relevantes que existen, cuántos se recuperaron."
        )
    except Exception as e:
        st.error(f"No se pudieron calcular las métricas: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    Esta app evalúa solo el **retrieval** del sistema RAG 
    (no la parte generativa).  
    Usa el conjunto de evaluación preparado en `data/processed/eval_set.json`.
    """
)

# Cuerpo principal de la app
st.markdown("### Consulta")

query = st.text_input(
    "Ingresa una query sobre features/bioactividad:",
    help=(
        "Puedes combinar m/z, RT, nombre del compuesto, bioactividad, etc. "
        "Ejemplo: 'm/z 449.107 RT 8.2 min myricetin actividad antidiabética'."
    ),
)

top_k = st.slider("Número de documentos (chunks) a recuperar (k)", 1, 20, 8)

# Ayuda / ejemplos de queries
with st.expander("Ver ejemplos de consultas útiles"):
    st.markdown(
        """
        Aquí tienes algunos ejemplos de queries para este dominio:

        - `m/z 449.107 actividad antidiabética`
        - `myricetin inhibición agregación plaquetaria`
        - `antocianinas de maqui perfil y bioactividad`
        - `polifenoles de té verde efectos cardiovasculares`

        La idea es describir una **feature** o un **compuesto** y la  
        **bioactividad de interés**, tal como lo haría un analista metabolómico.
        """
    )


if st.button("Buscar"):
    if not query.strip():
        st.warning("Escribe una query primero.")
    else:
        with st.spinner("Buscando en el índice vectorial..."):
            results = retrieve(query, k=top_k)

        st.markdown(f"### Resultados de la búsqueda")
        st.write(f"Se recuperaron **{len(results)}** chunks:")

        if not results:
            st.info("No se encontraron resultados para esta consulta.")
        else:
            for i, r in enumerate(results, start=1):
                meta = r.get("metadata", {})
                doc_id = meta.get("doc_id", "N/A")
                source = meta.get("source", "N/A")
                doc_type = meta.get("doc_type", "desconocido")
                chunk_index = meta.get("chunk_index", None)
                total_chunks = meta.get("total_chunks", None)

                header = f"Resultado {i} – doc_id={doc_id} (dist={r['distance']:.3f})"

                with st.expander(header):
                    st.write(f"**Fuente:** {source}")
                    st.write(f"**Tipo de documento (doc_type):** `{doc_type}`")

                    if chunk_index is not None and total_chunks is not None:
                        try:
                            # Mostramos indices 1-based para el usuario
                            st.write(
                                f"**Chunk:** {int(chunk_index) + 1} / {int(total_chunks)}"
                            )
                        except Exception:
                            # Si no son enteros, los mostramos tal cual
                            st.write(
                                f"**Chunk:** {chunk_index} / {total_chunks}")

                    st.markdown("---")
                    st.write(r.get("text", ""))
