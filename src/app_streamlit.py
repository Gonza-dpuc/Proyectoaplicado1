import streamlit as st
from retriever import retrieve
from benchmark import eval_precision_at_k

st.set_page_config(page_title="BioActives RAG Baseline", layout="wide")

st.title("BioActives RAG – Baseline (Grupo 5)")

st.sidebar.header("Métricas de rendimiento (Baseline)")
k_eval = st.sidebar.slider("k para Precision@k", 1, 10, 5)
if st.sidebar.button("Recalcular métricas"):
    p_at_k = eval_precision_at_k(k_eval)
    st.sidebar.write(f"Precision@{k_eval}: **{p_at_k:.2f}**")

st.markdown("### Consulta")
query = st.text_input("Ingresa una query sobre features/bioactividad:")

top_k = st.slider("Número de documentos a recuperar (k)", 1, 10, 5)

if st.button("Buscar"):
    if not query.strip():
        st.warning("Escribe una query primero.")
    else:
        results = retrieve(query, k=top_k)
        st.write(f"Se recuperaron {len(results)} chunks:")
        for i, r in enumerate(results, start=1):
            with st.expander(f"Resultado {i} – doc_id={r['metadata']['doc_id']} (dist={r['distance']:.3f})"):
                st.write(f"**Fuente:** {r['metadata']['source']}")
                st.write(r["text"])
