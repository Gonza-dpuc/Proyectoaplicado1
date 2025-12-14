import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE RUTAS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTACIONES BACKEND ---
from src.qdrant_impl import QdrantImpl
from src.vector_store_service import VectorStoreService
from src.rag_service import RAGService
from src.retrieval_strategy import get_strategy_from_selection

# --- CONFIGURACIÓN DE PÁGINA (GLOBAL) ---
st.set_page_config(
    page_title="BioActive RAG Dashboard",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "bioactives_prod" 

# =========================================================
# 🛠️ FUNCIONES AUXILIARES (CACHE)
# =========================================================

@st.cache_resource
def get_services():
    """Inicializa la conexión a BD una sola vez"""
    if not QDRANT_URL or not QDRANT_API_KEY:
        st.error("❌ Faltan credenciales en .env")
        return None, None

    try:
        db = QdrantImpl(collection_name=COLLECTION_NAME, url=QDRANT_URL, api_key=QDRANT_API_KEY)
        svc = VectorStoreService(db_impl=db)
        # Retornamos también el cliente crudo de Qdrant para estadísticas
        return svc, db.client 
    except Exception as e:
        st.error(f"❌ Error conectando a Qdrant: {e}")
        return None, None

vector_service, qdrant_client = get_services()

# =========================================================
# 📄 PÁGINA 1: CONSULTAS (CHAT)
# =========================================================
def page_chat():
    st.title("💬 Asistente Bio-Activo")
    st.markdown("Interactúa con la base de conocimiento científica.")

    # Sidebar específico del chat
    st.sidebar.subheader("Estrategias de Búsqueda")
    selected_strategies = st.sidebar.multiselect(
        "Algoritmos:", ["HyDE", "Step-Back", "Decomposition"], default=["HyDE"]
    )
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
        if not selected_strategies:
            st.warning("Selecciona una estrategia.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                try:
                    strategy = get_strategy_from_selection(selected_strategies, vector_service, OPENAI_API_KEY)
                    rag = RAGService(strategy, OPENAI_API_KEY)
                    
                    # Guardamos el último RAG service en sesión para analizarlo en otras páginas
                    st.session_state['last_rag_interaction'] = {
                        'query': prompt,
                        'strategies': selected_strategies,
                        'top_k': top_k
                    }

                    result = rag.answer_question(prompt, top_k=top_k)
                    
                    st.markdown(result["answer"])
                    
                    # Guardamos contexto para la página de "Generación"
                    st.session_state['last_context'] = result['context_used']
                    st.session_state['last_answer'] = result['answer']

                    if result["sources"]:
                        with st.expander("📚 Fuentes Utilizadas"):
                            for src in result["sources"]:
                                st.caption(f"📄 {src}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

# =========================================================
# 📄 PÁGINA 2: PROCESO DE INGESTA (DASHBOARD)
# =========================================================
def page_ingest_view():
    st.title("📥 Panel de Ingesta de Datos")
    
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
            info = qdrant_client.get_collection(COLLECTION_NAME)
            st.metric("Total Vectores (Chunks)", f"{info.points_count:,}")
        except:
            st.warning("No se pudo conectar.")

        if st.button("🔄 Cargar Muestra Real"):
            try:
                records, _ = qdrant_client.scroll(COLLECTION_NAME, limit=3, with_payload=True)
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
# 🧭 NAVEGACIÓN PRINCIPAL
# =========================================================
def main():
    st.sidebar.title("🧭 Navegación")
    
    page = st.sidebar.radio(
        "Ir a:",
        ["🤖 Consultas", "📥 Ingesta (DB)", "🔍 Retrieval (Debug)", "🧠 Generación (Prompt)"]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"DB: `{COLLECTION_NAME}`")

    if page == "🤖 Consultas":
        page_chat()
    elif page == "📥 Ingesta (DB)":
        page_ingest_view()
    elif page == "🔍 Retrieval (Debug)":
        page_retrieval_view()
    elif page == "🧠 Generación (Prompt)":
        page_generation_view()

if __name__ == "__main__":
    main()