import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import plotly.graph_objects as go
from pathlib import Path
# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTACIONES BACKEND ---
from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService
from src.rag_service import RAGService
from src.retrieval_strategy import get_strategy_from_selection
from src.benchmark import eval_metrics_at_k, load_eval_set # Importar benchmark
from src.retriever import BioactivesRetriever

# --- CONFIGURACIÓN DE PÁGINA (GLOBAL) ---
st.set_page_config(
    page_title="BioActive RAG Dashboard",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
if QDRANT_URL: QDRANT_URL = QDRANT_URL.strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================================================
# 🛠️ FUNCIONES AUXILIARES (CACHE)
# =========================================================

@st.cache_resource
def get_services(collection_name="bioactives_prod"):
    """Inicializa la conexión a BD una sola vez"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("❌ Faltan credenciales en .env")
        return None, None

    try:
        db = QdrantImpl(collection_name=collection_name, url=QDRANT_URL, api_key=QDRANT_API_KEY)
        svc = VectorStoreService(db_impl=db)
        # Retornamos también el cliente crudo de Qdrant para estadísticas
        return svc, db.client 
    except Exception as e:
        st.error(f"❌ Error conectando a Qdrant: {e}")
        return None, None

# =========================================================
# 📄 PÁGINA 1: CONSULTAS (CHAT)
# =========================================================
def page_chat():
    st.title("💬 Asistente Bio-Activo")
    st.markdown("Interactúa con la base de conocimiento científica.")

    # Configuración de Colección
    st.sidebar.subheader("🗄️ Base de Datos")
    selected_collection = st.sidebar.selectbox(
        "Colección:",
        ["bioactives_prod", "bioactives_hito1"],
        index=0,
        help="Selecciona la versión de la base de datos a consultar."
    )
    
    # Inicializar servicios con la colección seleccionada
    vector_service, _ = get_services(selected_collection)

    # Sidebar específico del chat
    st.sidebar.subheader("⚙️ Estrategias de Búsqueda")
    selected_strategies = st.sidebar.multiselect(
        "Algoritmos:", ["HyDE", "Step-Back", "Decomposition"], default=["HyDE"]
    )
    
    use_self_query = st.sidebar.checkbox("Self-Query (Filtros m/z)", value=True)
    use_repacking = st.sidebar.checkbox("Repacking (Agrupar Contexto)", value=True)
    use_reranker = st.sidebar.checkbox("Reranking (Cross-Encoder)", value=False, help="Reordena resultados usando un modelo neuronal.")
    
    # RRF Logic
    rrf_disabled = len(selected_strategies) <= 1
    use_rrf = st.sidebar.checkbox("Reranking (RRF)", value=True, disabled=rrf_disabled, help="Fusiona resultados de múltiples estrategias.")
    
    st.sidebar.divider()
    st.sidebar.subheader("👁️ Modo de Visualización")
    response_mode = st.sidebar.radio("Salida:", ["Respuesta Aumentada (RAG)", "Solo Documentos (k)"], index=0)
    
    top_k = st.sidebar.slider("Docs a recuperar:", 1, 10, 5)

    # Historial
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Fuentes"):
                    for src in msg["sources"]:
                        st.write(f"- {src}")

    if prompt := st.chat_input("Ej: ¿Qué efectos tiene el maqui en la inflamación?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    strategy = get_strategy_from_selection(selected_strategies, vector_service, OPENAI_API_KEY, use_reranker=use_reranker, use_rrf=use_rrf)
                    rag = RAGService(strategy, OPENAI_API_KEY)
                    
                    # Guardamos el último RAG service en sesión para analizarlo en otras páginas
                    st.session_state['last_rag_interaction'] = {
                        'query': prompt,
                        'strategies': selected_strategies,
                        'top_k': top_k
                    }

                    generate_response = (response_mode == "Respuesta Aumentada (RAG)")
                    result = rag.answer_question(
                        prompt, 
                        top_k=top_k, 
                        use_self_query=use_self_query, 
                        use_repacking=use_repacking,
                        generate_response=generate_response
                    )
                    
                    if generate_response:
                        st.markdown(result["answer"])
                    else:
                        st.info(f"🔍 Se recuperaron **{len(result['context_used'])}** fragmentos relevantes:")
                        for i, doc in enumerate(result['context_used']):
                            # Manejo seguro de dict vs objeto
                            content = doc.get('content', '') if isinstance(doc, dict) else getattr(doc, 'content', '')
                            src = doc.get('source_file', '') if isinstance(doc, dict) else getattr(doc, 'source_file', '')
                            score = doc.get('score', 0) if isinstance(doc, dict) else getattr(doc, 'score', 0)
                            
                            with st.expander(f"#{i+1} {os.path.basename(src)} (Score: {score:.4f})"):
                                st.markdown(content)
                    
                    # Guardamos contexto para la página de "Generación"
                    st.session_state['last_context'] = result['context_used']
                    st.session_state['last_answer'] = result['answer'] if result['answer'] else "Modo Solo Documentos (Sin generación)"

                    if result["sources"]:
                        with st.expander("📚 Fuentes Utilizadas"):
                            for src in result["sources"]:
                                st.caption(f"📄 {src}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"] if generate_response else f"🔍 *Mostrando {len(result['context_used'])} documentos recuperados (sin generación).* Checkea los expanders arriba.",
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# =========================================================
# 📄 PÁGINA 2: PROCESO DE INGESTA (DASHBOARD)
# =========================================================
def page_ingest_view():
    st.title("📥 Panel de Ingesta de Datos")
    
    # Inicializar servicios (Default Prod)
    vector_service, qdrant_client = get_services("bioactives_prod")

    # Usamos Tabs para organizar la información
    tab1, tab2, tab3 = st.tabs(["🏗️ Arquitectura", "🆚 Evolución (Hito 1 vs 2)", "🔍 Auditoría DB"])
    
    with tab1:
        st.markdown("### Pipeline de Procesamiento ETL")
        st.markdown("Transformación de documentos científicos en vectores de conocimiento:")

        # Diagrama Mermaid
        st.markdown("""
        ```mermaid
        graph LR
            A[📂 PDFs Crudos] -->|Parsing Inteligente| B(Detección de Estructura)
            B -->|Conversión| C[Formatos Markdown & Tablas]
            C -->|all-MiniLM-L6-v2| D{Chunking Semántico}
            D -->|Agrupación| E[Fragmentos Contextuales]
            E -->|text-embedding-3-small| F(Vectorización)
            F -->|Upsert| G[(Qdrant Cloud)]
            
            style C fill:#ff9,stroke:#333,stroke-width:2px
            style D fill:#f9f,stroke:#333,stroke-width:2px
            style G fill:#bbf,stroke:#333,stroke-width:2px
        ```
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.info("**1. Markdown & Tablas**\nPreservamos la estructura tabular, evitando la 'sopa de letras'.")
        c2.warning("**2. Chunking Semántico**\nCortamos el texto por cambios de tema, no por caracteres.")
        c3.success("**3. Embedding SOTA**\nModelos de última generación de OpenAI (1536 dim).")

    with tab2:
        st.header("🆚 Comparativo de Evolución")
        st.markdown("Visualiza la diferencia en la calidad de datos entre nuestra entrega anterior y la actual.")
        
        col_old, col_new = st.columns(2)
        
        with col_old:
            st.error("❌ Hito 1: Extracción Básica")
            st.caption("Tecnología: `PyPDF2` / `RecursiveCharacterTextSplitter`")
            st.markdown("El texto se extraía línea por línea, perdiendo la noción de filas y columnas en las tablas. La IA no podía interpretar los datos numéricos.")
            
            # Simulación de PDF mal leído
            mock_bad_text = """
            Tabla 1. Contenido de Polifenoles
            Muestra IC50 (mg/L) Antocianinas
            Maqui 12.5 450
            Murta 15.2 120
            Arándano 22.1 300
            Fuente: Estudio 2023
            La actividad antioxidante fue medida...
            """
            st.text_area("Resultado Hito 1 (Texto Plano):", mock_bad_text, height=200, disabled=True)
            st.markdown("Resultados RAG: *'El maqui tiene 12.5 antocianinas'* (Alucinación por mala lectura).")

        with col_new:
            st.success("✅ Hito 2: Parsing Estructurado")
            st.caption("Tecnología: `Docling/LlamaParse` + `SemanticChunker`")
            st.markdown("Convertimos el documento a **Markdown**. Las tablas se reconstruyen perfectamente, permitiendo razonamiento numérico preciso.")
            
            # Simulación de Markdown correcto
            mock_good_text = """
            ### Tabla 1. Contenido de Polifenoles y Actividad Antioxidante
            
            | Muestra   | IC50 (mg/L) | Antocianinas (mg/100g) |
            |-----------|-------------|------------------------|
            | **Maqui** | 12.5        | 450                    |
            | Murta     | 15.2        | 120                    |
            | Arándano  | 22.1        | 300                    |
            
            > *Fuente: Estudio comparativo 2023*
            
            La actividad antioxidante fue medida utilizando el método DPPH...
            """
            st.text_area("Resultado Hito 2 (Markdown):", mock_good_text, height=200, disabled=True)
            st.markdown("Resultados RAG: *'El Maqui presenta el menor IC50 (12.5), indicando mayor potencia...'*")

    with tab3:
        st.header("🔍 Auditoría en Tiempo Real")
        
        if not qdrant_client:
            st.error("Sin conexión a DB.")
            return

        try:
            info = qdrant_client.get_collection("bioactives_prod")
            st.metric("Total Vectores (Chunks)", f"{info.points_count:,}")
        except:
            st.warning("No se pudo conectar.")

        if st.button("🔄 Cargar Muestra Real"):
            try:
                records, _ = qdrant_client.scroll("bioactives_prod", limit=3, with_payload=True)
                for rec in records:
                    pl = rec.payload
                    with st.expander(f"📄 {os.path.basename(pl.get('source_file','Doc'))}"):
                        st.code(pl.get('content')[:500] + "...", language="markdown")
            except Exception as e:
                st.error(f"Error: {e}")

# =========================================================
# 📄 PÁGINA 3: PROCESO DE RETRIEVAL (X-RAY)
# =========================================================
def page_retrieval_view():
    st.title("🔎 Rayos X: Retrieval Process")
    st.markdown("Analiza cómo las estrategias encuentran información antes de responder.")
    
    # Inicializar servicios (Default Prod)
    vector_service, _ = get_services("bioactives_prod")

    query = st.text_input("Query de prueba:", value=st.session_state.get('last_rag_interaction', {}).get('query', ""))
    
    col1, col2 = st.columns(2)
    with col1:
        strategy_name = st.selectbox("Estrategia a auditar:", ["HyDE", "Step-Back", "Decomposition"])
    with col2:
        top_k = st.slider("Top K:", 1, 10, 3)

    if st.button("🧪 Ejecutar Análisis de Búsqueda"):
        with st.status("Ejecutando pipeline de recuperación...", expanded=True) as status:
            
            # 1. Visualizar paso intermedio (Alucinación o Abstracción)
            st.write(f"**1. Pre-procesamiento ({strategy_name}):**")
            
            mock_rag = get_strategy_from_selection([strategy_name], vector_service, OPENAI_API_KEY)
            
            # Truco para mostrar los internos (accedemos a métodos privados para demo)
            if strategy_name == "HyDE":
                hypo = mock_rag.strategies[0]._generate_hypothetical_doc(query) if hasattr(mock_rag, 'strategies') else mock_rag._generate_hypothetical_doc(query)
                st.info(f"👻 **Documento Hipotético Generado:**\n\n_{hypo}_")
            
            elif strategy_name == "Step-Back":
                step = mock_rag.strategies[0]._generate_step_back(query) if hasattr(mock_rag, 'strategies') else mock_rag._generate_step_back(query)
                st.info(f"🔙 **Pregunta Step-Back:**\n\n_{step}_")

            elif strategy_name == "Decomposition":
                 subq = mock_rag.strategies[0]._decompose_query(query) if hasattr(mock_rag, 'strategies') else mock_rag._decompose_query(query)
                 st.info(f"🧩 **Sub-preguntas:** {subq}")

            # 2. Ejecutar búsqueda real
            st.write("**2. Búsqueda Vectorial (Qdrant):**")
            results = mock_rag.retrieve(query, top_k=top_k)
            status.update(label="Búsqueda completada", state="complete")

        st.subheader(f"📄 Resultados Encontrados ({len(results)})")
        
        for i, res in enumerate(results):
            # Normalizar
            if isinstance(res, dict):
                content = res.get('content', '')
                src = res.get('source_file', '')
            else:
                content = getattr(res, 'content', '')
                src = getattr(res, 'source_file', '')
                
            with st.expander(f"Resultado #{i+1} - {os.path.basename(src)}"):
                st.markdown(f"**Score Similitud:** (Implícito en ranking)") 
                st.text(content)

# =========================================================
# 📄 PÁGINA 4: PROCESO DE GENERACIÓN (PROMPT)
# =========================================================
def page_generation_view():
    st.title("🧠 Generación & Prompt Engineering")
    st.markdown("Inspecciona cómo se construye el prompt final para GPT-3.5/4.")

    if 'last_context' not in st.session_state:
        st.warning("⚠️ Primero realiza una consulta en la página de 'Consultas' para ver datos aquí.")
        return

    st.subheader("1. Contexto Recuperado (Inyectado)")
    docs = st.session_state['last_context']
    
    context_text = ""
    for i, doc in enumerate(docs):
        if isinstance(doc, dict):
            src = os.path.basename(doc.get('source_file', ''))
            txt = doc.get('content', '')
        else:
            src = os.path.basename(getattr(doc, 'source_file', ''))
            txt = getattr(doc, 'content', '')
        context_text += f"\n--- FUENTE {i+1}: {src} ---\n{txt}\n"
    
    st.text_area("Contexto Crudo:", value=context_text, height=200, disabled=True)

    st.subheader("2. Prompt del Sistema (Instrucciones)")
    system_prompt = (
        "Eres un asistente de investigación experto en fitoquímica y compuestos bioactivos. "
        "Responde a la pregunta basándote EXCLUSIVAMENTE en el contexto proporcionado..."
    )
    st.info(system_prompt)

    st.subheader("3. Prompt del Usuario Final")
    last_query = st.session_state.get('last_rag_interaction', {}).get('query', 'N/A')
    final_prompt = f"Contexto:\n{context_text}\n\nPregunta: {last_query}"
    
    with st.expander("Ver Prompt Completo enviado a OpenAI"):
        st.code(final_prompt)

    st.subheader("4. Respuesta Generada")
    st.success(st.session_state.get('last_answer', ''))
# =========================================================
# 📄 PÁGINA 5: MÉTRICAS & BENCHMARKING
# =========================================================
def page_benchmark_view():
    st.title("📊 Métricas de Desempeño (Benchmark)")
    st.markdown("Comparativa técnica entre el **Modelo Base (Hito 1)** y el **Modelo Avanzado (Hito 2 - Actual)**.")

    # --- CONFIGURACIÓN DE ESTRATEGIAS ---
    st.subheader("⚙️ Configuración del Modelo Avanzado")
    
    col_strat, col_opts = st.columns(2)
    
    with col_strat:
        selected_strategies = st.multiselect(
            "1. Preprocesamiento de Queries:",
            ["HyDE", "Step-Back", "Decomposition"],
            default=["HyDE"]
        )
        
    with col_opts:
        st.write("2. Componentes Adicionales:")
        use_self_query_bench = st.checkbox("Self-Query (Filtros m/z)", value=True, help="Extrae filtros químicos de la pregunta.")
        use_reranker_bench = st.checkbox("Reranking (Cross-Encoder)", value=False, help="Activa reordenamiento neuronal.")
        
        # Lógica para checkbox de RRF
        rrf_disabled = len(selected_strategies) <= 1
        use_rrf_bench = st.checkbox(
            "Reranking (RRF)", 
            value=True, 
            disabled=rrf_disabled,
            help="Fusiona resultados de múltiples estrategias (Reciprocal Rank Fusion). Si se desactiva, usa unión simple."
        )

    # --- EJECUCIÓN REAL ---
    if st.button("🚀 Ejecutar Benchmark en Tiempo Real (Dataset Gemini)"):
        with st.spinner("Evaluando colecciones... esto puede tardar unos segundos."):
            try:
                dataset_path = Path("data/synthetic_dataset_gemini.json")
                if not dataset_path.exists():
                    st.error("No se encontró data/synthetic_dataset_gemini.json")
                    return
                
                eval_set = load_eval_set(dataset_path)
                
                # Evaluar Hito 1
                m_h1 = eval_metrics_at_k(k=5, eval_set=eval_set, collection_name="bioactives_hito1", use_advanced_strategies=False)
                # Evaluar Prod
                m_prod = eval_metrics_at_k(
                    k=5, 
                    eval_set=eval_set, 
                    collection_name="bioactives_prod", 
                    use_advanced_strategies=True,
                    strategies_selection=selected_strategies,
                    use_self_query=use_self_query_bench,
                    use_reranker=use_reranker_bench,
                    use_rrf=use_rrf_bench
                )
                
                st.session_state['benchmark_results'] = {
                    "Métrica": ["Recall@5 (Recuperación)", "Precision@5"],
                    "Hito 1 (Básico)": [m_h1['recall_at_k'], m_h1['precision_at_k']],
                    "Hito 2 (Avanzado)": [m_prod['recall_at_k'], m_prod['precision_at_k']]
                }
                st.success("Evaluación completada.")
            except Exception as e:
                st.error(f"Error durante el benchmark: {e}")

    # Usar datos reales si existen, sino mostrar placeholder
    if 'benchmark_results' in st.session_state:
        df_metrics = pd.DataFrame(st.session_state['benchmark_results'])
        
        st.divider()
        
        st.subheader("Resultados de la Ejecución")
        # Transformamos datos para Plotly
        df_melted = df_metrics.melt(id_vars="Métrica", var_name="Modelo", value_name="Score")
        
        fig_bar = px.bar(
            df_melted, 
            x="Métrica", 
            y="Score", 
            color="Modelo", 
            barmode="group",
            text_auto=".2f",
            color_discrete_map={"Hito 1 (Básico)": "#EF553B", "Hito 2 (Avanzado)": "#00CC96"},
            title="Comparativa de Rendimiento (Recall & Precision)"
        )
        fig_bar.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)
        
        with st.expander("📝 Nota sobre la Metodología"):
            st.caption("""
            * **Hito 1:** Utilizaba chunking por caracteres y búsqueda simple.
            * **Hito 2:** Incorpora Parsing de Markdown, Chunking Semántico y Búsqueda Híbrida (Ensemble).
            * **Métricas:** Calculadas en tiempo real sobre el dataset sintético de Gemini.
            """)
    else:
        st.info("👈 Ejecuta el benchmark para ver los resultados reales.")

    # --- SECCIÓN DE ANÁLISIS DETALLADO ---
    st.divider()
    st.header("🔬 Análisis Detallado por Pregunta")
    st.markdown("Selecciona una pregunta del dataset para comparar visualmente qué recupera cada modelo.")

    dataset_path = Path("data/synthetic_dataset_gemini.json")
    if not dataset_path.exists():
        st.warning("⚠️ No se encontró data/synthetic_dataset_gemini.json")
    else:
        # Cargar dataset
        eval_set = load_eval_set(dataset_path)
        
        # Crear opciones para el selectbox (Index + Inicio de pregunta)
        options = {f"{i+1}. {item['query'][:80]}...": item for i, item in enumerate(eval_set)}
        selected_label = st.selectbox("Selecciona una pregunta:", list(options.keys()))
        
        if selected_label:
            item = options[selected_label]
            query = item['query']
            ground_truth_ids = item.get('relevant_doc_ids', [])
            
            st.info(f"**Pregunta:** {query}")
            if st.checkbox("Ver Ground Truth (Respuesta Esperada)"):
                st.write(item.get('ground_truth', 'No disponible'))
                st.caption(f"ID Esperado: {ground_truth_ids}")
                st.caption(f"Documento Esperado: {item.get('source_file', 'No especificado')}")

            if st.button("🔍 Comparar Hito 1 vs Hito 2"):
                col1, col2 = st.columns(2)
                
                # Función helper para mostrar resultados
                def show_results(col, title, collection, is_prod=False):
                    with col:
                        st.subheader(title)
                        try:
                            retriever = BioactivesRetriever(collection_name=collection)
                            results = retriever.search(query, limit=3)
                            
                            for i, res in enumerate(results):
                                # Verificar match solo si es prod (Hito 1 tiene IDs distintos)
                                is_match = is_prod and (res.get('chunk_id') in ground_truth_ids)
                                icon = "✅" if is_match else "📄"
                                
                                with st.container(border=True):
                                    st.markdown(f"**{i+1}. {res.get('source_file')}** {icon}")
                                    st.caption(f"Score (Similitud): {res.get('score', 0):.4f} (Mayor es mejor ⬆️)")
                                    with st.expander("Ver contenido completo"):
                                        st.markdown(res.get('content', ''))
                        except Exception as e:
                            st.error(f"Error: {e}")

                show_results(col1, "1️⃣ Hito 1 (Básico)", "bioactives_hito1", is_prod=False)
                show_results(col2, "2️⃣ Hito 2 (Prod)", "bioactives_prod", is_prod=True)

# =========================================================
# 🧭 NAVEGACIÓN PRINCIPAL
# =========================================================
def main():
    st.sidebar.title("🧭 Navegación")
    
    # 1. Agregamos la nueva opción a la lista
    page = st.sidebar.radio(
        "Ir a:",
        [
            "🤖 Consultas", 
            "📥 Ingesta (DB)", 
            "🔍 Retrieval (Debug)", 
            "🧠 Generación (Prompt)",
            "📊 Métricas (Benchmark)" # <--- NUEVA PÁGINA
        ]
    )

    st.sidebar.markdown("---")

    # 2. Conectamos la opción con la función
    if page == "🤖 Consultas":
        page_chat()
    elif page == "📥 Ingesta (DB)":
        page_ingest_view()
    elif page == "🔍 Retrieval (Debug)":
        page_retrieval_view()
    elif page == "🧠 Generación (Prompt)":
        page_generation_view()
    elif page == "📊 Métricas (Benchmark)": # <--- NUEVA CONEXIÓN
        page_benchmark_view()

if __name__ == "__main__":
    main()
