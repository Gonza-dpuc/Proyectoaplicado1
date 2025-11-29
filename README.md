BioActives RAG - Baseline (Grupo 5)
===================================

Pipeline RAG ligero para recuperar informacion sobre compuestos bioactivos. Usa OpenAI embeddings, guarda el indice en ChromaDB y expone una app en Streamlit para probar la recuperacion y ver metricas de Precision@k y Recall@k.

## Requisitos rapidos
- Python 3.10+ recomendado
- Cuenta de OpenAI con clave para embeddings (`text-embedding-3-small`)
- pip y un entorno virtual (opcional pero sugerido)

## Instalacion
1) Crear y activar entorno virtual (opcional):
```
python -m venv .venv
.venv\Scripts\activate
```
2) Instalar dependencias:
```
pip install -r requirements.txt
```
3) Crear archivo `.env` en la raiz con tu clave:
```
OPENAI_API_KEY=tu_clave
```
4) Colocar los archivos fuente (PDF, CSV, TXT) en `data/raw/`.

## Flujo del pipeline (CLI)
Ejecuta en este orden para construir el indice vectorial:
```
python -m src.ingest            # Lee data/raw y genera data/processed/corpus.parquet
python -m src.chunking          # Parte los documentos en chunks -> data/processed/chunks.parquet
python -m src.build_index       # Crea el indice Chroma en chroma_db_bioactives/
python -m src.generate_eval_set # Arma data/processed/eval_set.json con queries de prueba
```

## App Streamlit
Arranca la UI con:
```
streamlit run app_streamlit.py
```
Que hace la app:
- En la primera carga corre el pipeline completo (si no existe).
- Sidebar con Precision@k y Recall@k calculados sobre `data/processed/eval_set.json`.
- Boton para reinicializar todo el pipeline.
- Busqueda libre sobre el indice (top-k configurable) con metadatos del chunk recuperado.

## Evaluacion rapida
Puedes calcular metricas desde consola:
```
python -m src.benchmark     # Usa eval_set.json o el fallback interno (k=5 por defecto)
```
Para cambiar k, ajusta `eval_metrics_at_k` en `src/benchmark.py` o usa directamente `eval_precision_at_k` y `eval_recall_at_k` en tus scripts (son las mismas funciones que consume la app).

## Estructura principal
- `app_streamlit.py`             : UI Streamlit y orquestacion de pipeline/metricas.
- `src/ingest.py`                : Lee PDFs, CSV y TXT de `data/raw` y los limpia.
- `src/chunking.py`              : Divide en chunks con solapamiento y metadatos.
- `src/build_index.py`           : Genera embeddings OpenAI y crea la coleccion Chroma.
- `src/retriever.py`             : Funciones de consulta contra el indice persistente.
- `src/generate_eval_set.py`     : Crea queries de evaluacion y doc_ids relevantes.
- `src/benchmark.py`             : Precision@k y Recall@k (macro) sobre el eval set.
- `src/debug_chroma.py`          : Lista colecciones almacenadas en `chroma_db_bioactives/`.

## Directorios y artefactos
- `data/raw/`         : Entrada (PDF, CSV, TXT).
- `data/processed/`   : Salida intermedia (`corpus.parquet`, `chunks.parquet`, `eval_set.json`).
- `chroma_db_bioactives/` : Indice vectorial persistente usado por la app y el retriever.

## Notas y tips
- Si cambias el contenido de `data/raw`, vuelve a correr todo el pipeline para regenerar chunks e indice.
- Las metricas dependen del `eval_set.json`; ajusta las queries o doc_ids relevantes segun tu coleccion.
- Si la app no encuentra la coleccion, ejecuta de nuevo `python -m src.build_index` o usa `python -m src.debug_chroma` para revisar las colecciones existentes.
