import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DIR

# Normalización ligera de la consulta


def normalize_query(text: str) -> str:
    """
    Limpieza muy ligera de la consulta:
    - Convierte a str si viene otro tipo.
    - Elimina saltos de línea y normaliza espacios.
    No toca mayúsculas/minúsculas para no alterar matices.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


# Cliente y colección Chroma
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# aquí usamos embedding_function para que Chroma calcule el embedding de la query.
# Los vectores de los documentos ya fueron generados en build_index.py usando el mismo modelo.
collection = client.get_or_create_collection(
    name="bioactives_chunks",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    ),
)


# Función principal de retrieval
def retrieve(
    query: str,
    k: int = 5,
    doc_type: str | None = None,
    source: str | None = None,
    where: dict | None = None,
):

    normalized_query = normalize_query(query)
    if not normalized_query:
        # Consulta vacía -> no buscamos nada
        return []

    # Construimos el diccionario de filtros
    filters = {}

    # Filtros simples por doc_type y source
    if doc_type is not None:
        filters["doc_type"] = doc_type

    if source is not None:
        # Comparación exacta; en el futuro podrías usar $contains si lo necesitas
        filters["source"] = source

    # Filtros avanzados proporcionados por el usuario
    if where is not None:
        # Permitimos que 'where' sobreescriba claves si coincide
        filters.update(where)

    query_kwargs = {
        "query_texts": [normalized_query],
        "n_results": k,
    }

    # Solo pasamos 'where' si realmente hay filtros
    if filters:
        query_kwargs["where"] = filters

    results = collection.query(**query_kwargs)

    # Si no hay resultados, devolvemos lista vacía
    if not results["ids"] or len(results["ids"][0]) == 0:
        return []

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return docs
