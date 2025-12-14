# 🧬 Sistema RAG para Análisis de Compuestos Bioactivos

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** diseñado para procesar, indexar y consultar literatura científica relacionada con farmacología y nutracéuticos (específicamente bayas chilenas como Maqui y Murta).

El sistema permite realizar **búsquedas semánticas** sobre PDFs técnicos, recuperando fragmentos relevantes basados en vectores de embeddings.



## 🚀 Características Principales

* **Ingesta Inteligente:** Procesa archivos PDF, extrayendo texto y metadatos.
* **Vectorización:** Utiliza modelos de OpenAI (`text-embedding-3-small`) para generar embeddings de alta calidad.
* **Base de Datos Vectorial:** Integración robusta con **Qdrant Cloud**.
* **Búsqueda Semántica:** Capacidad de encontrar relaciones conceptuales, no solo palabras clave exactas.
* **Filtrado Avanzado:** (En desarrollo) Soporte para filtros por propiedades químicas (ej. masa/carga `m/z`).

## 🧠 Pipeline de Procesamiento (Chunking y Embeddings)

El núcleo de este sistema RAG reside en cómo transformamos documentos científicos densos (PDFs) en información recuperable. Utilizamos un enfoque semántico para preservar el contexto de los hallazgos químicos.



### 1. Estrategia de Chunking (Segmentación)
En lugar de cortar el texto arbitrariamente por número de caracteres, utilizamos una estrategia de **Chunking Semántico**:

* **Detección de Estructura:** El sistema analiza la estructura del PDF (encabezados, párrafos) para intentar mantener secciones lógicas juntas.
* **Split Strategy:** `semantic-local`. Esto agrupa oraciones y párrafos que tienen una fuerte relación temática, evitando cortar una frase o una fórmula química a la mitad.
* **Metadatos:** Cada fragmento conserva su referencia al archivo origen (`source_file`) y, cuando es posible, extrae propiedades específicas como valores de masa/carga (`m/z`) para permitir filtros técnicos futuros.

### 2. Modelo de Embeddings
Una vez fragmentado el texto, convertimos el lenguaje natural en vectores numéricos:

* **Modelo:** `text-embedding-3-small` (OpenAI).
* **Dimensión:** 1536 dimensiones.
* **Justificación:** Este modelo ofrece un equilibrio óptimo entre latencia y precisión semántica, permitiendo capturar matices en descripciones de compuestos bioactivos mejor que modelos anteriores.

### 3. Almacenamiento en Qdrant
Los vectores resultantes se indexan en **Qdrant Cloud** bajo la colección `bioactives_v1`.
* **Vector:** Representación matemática del contenido.
* **Payload (Carga útil):** Contiene el texto original (`content`), nombre del archivo y metadatos extraídos.
* **Índices:** Se ha configurado indexación especial para campos numéricos (como `mz`) para acelerar búsquedas por rango.

## 🛠️ Requisitos del Sistema

* **Python:** 3.10 o superior.
* **Cuenta Qdrant Cloud:** Un cluster activo.
* **OpenAI API Key:** Créditos disponibles para embeddings.

## 📦 Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DEL_REPO>
    cd Proyectoaplicado1
    ```

2.  **Crear entorno virtual:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuración (.env)

⚠️ **IMPORTANTE:** La configuración de la conexión es crítica. Crea un archivo `.env` en la raíz y sigue este formato exacto para evitar errores de DNS (`getaddrinfo failed`).

```env
# API Keys
OPENAI_API_KEY=sk-tu-clave-aqui...
QDRANT_API_KEY=tu-api-key-de-qdrant...

# URL de Qdrant Cloud
# NOTA: No incluir el puerto (:6333) al final. Solo el host y protocolo.
# Incorrecto: [https://xyz.aws.cloud.qdrant.io:6333](https://xyz.aws.cloud.qdrant.io:6333)
# Correcto:   [https://xyz.aws.cloud.qdrant.io](https://xyz.aws.cloud.qdrant.io)
QDRANT_URL=[https://tu-cluster-id.region.aws.cloud.qdrant.io](https://tu-cluster-id.region.aws.cloud.qdrant.io)

# Configuración de la Colección
QDRANT_COLLECTION_NAME=bioactives_v1

#🏃‍♂️ Uso
#1. Ingesta de Datos (Carga de PDFs)

Este script lee los PDFs de data/raw/, genera los embeddings y los sube a Qdrant.
Bash

python scripts/ingest.py

#2. Prueba de Búsqueda

Verifica que el sistema puede recuperar información relevante mediante una consulta de prueba.
Bash

python scripts/test_search.py

#📂 Estructura del Proyecto
Plaintext

Proyectoaplicado1/
├── data/
│   ├── raw/                 # PDFs originales
│   └── processed/           # Datos intermedios (chunks)
├── scripts/
│   ├── ingest.py            # Pipeline de carga y vectorización
│   └── test_search.py       # Script de validación de búsqueda
├── src/
│   ├── qdrant_impl.py       # Lógica de conexión y manejo de Qdrant (Bridge Pattern)
│   ├── vector_store_service.py # Lógica de negocio (Service Layer)
│   └── models.py            # Definiciones de datos (Pydantic)
├── .env                     # Variables de entorno (NO SUBIR A GITHUB)
├── .gitignore               # Archivos ignorados por Git
└── requirements.txt         # Dependencias del proyecto

#🔧 Solución de Problemas Comunes
Error: httpcore.ConnectError: [Errno 11001] getaddrinfo failed

    Causa: La variable QDRANT_URL en el .env tiene el puerto :6333 al final o espacios en blanco invisibles.

    Solución: Edita el .env, borra el puerto y asegúrate de eliminar espacios al final de la línea. El código en src/qdrant_impl.py incluye limpieza automática, pero es mejor tener el .env limpio.

Error: AttributeError: 'QdrantClient' object has no attribute 'search'

    Causa: Estás usando una versión reciente de qdrant-client (v1.10+) que eliminó el método .search().

    Solución: El proyecto ya implementa query_points() que es compatible con las nuevas versiones. Asegúrate de tener las dependencias actualizadas.