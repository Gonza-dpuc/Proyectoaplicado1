import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

import streamlit as st
from dotenv import load_dotenv

# --- IMPORTACIONES DE LA NUEVA ARQUITECTURA (HITO 2) ---
# Asegúrate de que los archivos creados en el paso anterior estén en src/
from src.qdrant_impl import QdrantImpl        # Implementación Concreta (Bridge)
from src.vector_store_service import VectorStoreService # Abstracción (Bridge)
from src.hybrid_search_strategy import HybridSearchStrategy # Estrategia (Strategy)

# Mantenemos esto para la carga inicial de datos si es necesario
from src.initialize import initialize_st 
# Nota: src.benchmark deberá actualizarse para recibir la estrategia, 
# por ahora lo comentamos o lo adaptamos.
# from src.benchmark import eval_precision_at_k 

load_dotenv()

# ============================================================
#   1. CONFIGURACIÓN E INICIALIZACIÓN (SETUP)
# ============================================================

st.set_page_config(page_title="BioActives RAG - Advanced", layout="wide")
st.title("BioActives RAG – Arquitectura Avanzada (Bridge + Strategy)")

# Variable de estado para controlar si el pipeline ya se instanció
if "rag_strategy" not in st.session_state:
    st.session_state.rag_strategy = None

def setup_rag_pipeline():
    """
    Configura el ensamblaje de clases (Composición Root).
    Aquí es donde ocurre la 'Magia' de los patrones.
    """
    try:
        # 1. BRIDGE: Seleccionamos la Tecnología
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY") # <--- Agregar esto
        collection_name = os.getenv("QDRANT_COLLECTION", "bioactives_v1")
        
        # Pasar la API Key al constructor
        db_impl = QdrantImpl(
            collection_name=collection_name, 
            url=qdrant_url, 
            api_key=qdrant_api_key # <--- Agregar esto
        )
        
        # 2. BRIDGE: Inicializamos el Servicio (Abstracción)
        rag_service = VectorStoreService(db_impl=db_impl)
        
        # 3. STRATEGY: Definimos la estrategia de búsqueda
        # (Aquí podríamos elegir entre SemanticStrategy o HybridStrategy)
        strategy = HybridSearchStrategy(store_service=rag_service)
        
        # Guardamos la estrategia lista para usar en sesión
        st.session_state.rag_strategy = strategy
        st.toast("Pipeline RAG inicializado con éxito (Bridge + Strategy)", icon="✅")
        
    except Exception as e:
        st.error(f"Error inicializando el pipeline: {e}")

# ============================================================
#   2. CONTROL DE PRIMERA EJECUCIÓN
# ============================================================

# Inicialización de lógica (Clases)
if st.session_state.rag_strategy is None:
    with st.spinner("Conectando a Qdrant y configurando estrategias..."):
        setup_rag_pipeline()

# Inicialización de datos (ETL) - Solo si es necesario cargar PDFs
if "data_initialized" not in st.session_state:
    st.sidebar.info("Verificando datos...")
    # initialize_st() # Descomenta si tu script de carga de datos debe correr aquí
    st.session_state.data_initialized = True

# ============================================================
#   3. SIDEBAR: CONTROLES
# ============================================================

st.sidebar.header("Configuración del Sistema")

if st.sidebar.button("Re-conectar Pipeline"):
    setup_rag_pipeline()

st.sidebar.markdown("---")
st.sidebar.header("Métricas (Evaluación)")

# NOTA: Para el Hito 2, el evaluador debe recibir la estrategia.
# Esta sección requiere que actualices src.benchmark para usar DIP.
# Por ahora, dejamos un placeholder.
st.sidebar.warning("⚠️ El módulo de métricas se está actualizando para usar DIP.")

# ============================================================
#   4. BÚSQUEDA DE DOCUMENTOS (USANDO STRATEGY)
# ============================================================

st.markdown("### Consulta Bio-Activa")

query = st.text_input(
    "Ingresa una query:",
    value="Tengo una feature con m/z 449.107 en Té Verde",
    help="El sistema usará Hybrid Search Strategy."
)

# Filtros opcionales (Simulación de UI avanzada)
with st.expander("Filtros Avanzados (Metadata)"):
    col1, col2 = st.columns(2)
    with col1:
        min_mz = st.number_input("Min m/z", value=0.0)
    with col2:
        max_mz = st.number_input("Max m/z", value=1000.0)
    
    filters = {}
    if min_mz > 0 or max_mz < 1000:
        filters["mz"] = {"gte": min_mz, "lte": max_mz}

top_k = st.slider("Documentos a recuperar (k)", 1, 20, 5)

if st.button("Buscar"):
    if not query.strip():
        st.warning("Escribe una query primero.")
    else:
        if st.session_state.rag_strategy is None:
            st.error("El pipeline no está inicializado.")
        else:
            with st.spinner("Ejecutando HybridSearchStrategy..."):
                # --- AQUÍ OCURRE EL CAMBIO CLAVE ---
                # Ya no llamamos a retrieve(), llamamos al método de la estrategia
                results = st.session_state.rag_strategy.retrieve_context(
                    query=query, 
                    filters=filters, 
                    k=top_k
                )

            st.markdown(f"### Resultados")
            st.write(f"Se recuperaron **{len(results)}** chunks usando la estrategia híbrida.")

            if not results:
                st.info("No se encontraron resultados.")
            else:
                for i, r in enumerate(results, start=1):
                    # Asumimos que r es un objeto Pydantic ProcessedChunk o un dict
                    # Adaptamos la visualización según tu modelo de datos
                    
                    # Si r es un objeto Pydantic:
                    content = getattr(r, 'text', getattr(r, 'content', str(r)))
                    metadata = getattr(r, 'metadata', {})
                    
                    # Si r es un dict (en caso de que no hayas implementado Pydantic completo aún):
                    if isinstance(r, dict):
                        content = r.get('text', r.get('content', ''))
                        metadata = r.get('metadata', r.get('payload', {}))

                    doc_id = metadata.get("doc_id", "N/A")
                    source = metadata.get("source", "N/A")
                    
                    with st.expander(f"Resultado {i} - {source}"):
                        st.markdown(f"**Contenido:**")
                        st.write(content)
                        st.markdown("**Metadatos:**")
                        st.json(metadata)