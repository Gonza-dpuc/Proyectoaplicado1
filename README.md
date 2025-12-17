# 🧬 Sistema RAG Avanzado para Análisis de Compuestos Bioactivos

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** de última generación, diseñado para procesar, indexar y consultar literatura científica sobre fitoquímica, metabolómica y alimentos funcionales.

El sistema representa una evolución desde un modelo base (**Hito 1**) hacia una arquitectura avanzada (**Hito 2**) que incorpora estrategias de recuperación complejas para mejorar la precisión en consultas técnicas y numéricas.

## 🚀 Características Principales (Hito 2)

* **Ingesta Estructurada:** Procesamiento de PDFs a **Markdown** (preservando tablas y estructura) en lugar de texto plano.
* **Estrategias de Recuperación Avanzada:**
    *   **HyDE (Hypothetical Document Embeddings):** Genera respuestas hipotéticas para mejorar la búsqueda semántica.
    *   **Step-Back Prompting:** Abstrae la pregunta para buscar contexto teórico.
    *   **Decomposition:** Divide preguntas complejas en sub-consultas.
* **Fusión y Reordenamiento:**
    *   **Ensemble Retrieval:** Combina múltiples estrategias usando **Reciprocal Rank Fusion (RRF)**.
    *   **Reranking:** Refina los resultados usando modelos **Cross-Encoder**.
* **Consultas Estructuradas (Self-Query):** Detecta automáticamente filtros químicos (ej. `m/z 449.1`, `RT 8.2 min`) en lenguaje natural y los aplica a la base de datos.
* **Dashboard Integral:** Interfaz en Streamlit para Chat, Auditoría de Ingesta, Debugging de Retrieval y Benchmarking en tiempo real.

## ️ Requisitos del Sistema

* **Python:** 3.10 o superior.
* **Qdrant Cloud:** Cluster activo.
* **OpenAI API Key:** Para embeddings y generación.
* **Google Gemini API Key:** Para la generación de datasets sintéticos de evaluación.

## 📦 Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPO>
    cd Proyectoaplicado1
    ```

2.  **Crear entorno virtual e instalar dependencias:**
    ```bash
    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Mac/Linux: source .venv/bin/activate
    pip install -r requirements.txt
    ```

## ⚙️ Configuración (.env)

Crea un archivo `.env` en la raíz con las siguientes variables:

```env
# LLM & Embeddings
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=... (Para generar datasets de evaluación)

# Vector Database (Qdrant)
QDRANT_API_KEY=...
QDRANT_URL=https://tu-cluster.qdrant.io  # Sin puerto ni espacios
```

## 🏃‍♂️ Uso

### 1. Dashboard Principal (Streamlit)
La forma más fácil de interactuar con el sistema (Chat, Métricas, Ingesta).
```bash
streamlit run app_streamlit.py
```

### 2. Ingesta de Datos (Producción)
Procesa PDFs desde `data/prod_raw/` hacia la colección `bioactives_prod`. Soporta "resume" (no duplica archivos) y extracción de metadatos químicos.
```bash
python scripts/ingest_prod.py
```

### 3. Generación de Dataset de Evaluación
Usa Google Gemini para crear preguntas complejas (QA pairs) basadas en tus documentos para pruebas de estrés.
```bash
python scripts/generate_dataset_gemini.py
```

### 4. Comparación de Modelos (Benchmark)
Ejecuta una evaluación técnica comparando Hito 1 vs Hito 2.
```bash
python scripts/compare_collections.py
```

## 📂 Estructura del Proyecto

```plaintext
Proyectoaplicado1/
├── app_streamlit.py         # Dashboard principal
├── data/
│   ├── prod_raw/            # PDFs para producción
│   └── synthetic_dataset... # Dataset de evaluación generado
├── scripts/
│   ├── ingest_prod.py       # Pipeline de ingesta oficial
│   ├── generate_dataset...  # Generador de QA con Gemini
│   └── compare_collections.py # Script de benchmark CLI
├── src/
│   ├── rag_service.py       # Orquestador RAG (Generation)
│   ├── retrieval_strategy.py # Estrategias (HyDE, RRF, Rerank)
│   ├── vector_store_service.py # Lógica de negocio DB
│   ├── retriever.py         # Cliente de búsqueda para benchmarks
│   └── qdrant_impl.py       # Adaptador Qdrant
└── requirements.txt
```