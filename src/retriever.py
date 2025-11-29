import chromadb
from chromadb.utils import embedding_functions
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DIR


# -------------------------------------------------------------------------
# Normalización ligera de la consulta
# -------------------------------------------------------------------------
def normalize_query(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


# -------------------------------------------------------------------------
# Obtener siempre la colección fresca desde disco
# (soluciona el error de Collection [UUID] does not exist)
# -------------------------------------------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name="bioactives_chunks",
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL,
        ),
    )


# -------------------------------------------------------------------------
# Función principal de retrieval
# -------------------------------------------------------------------------
def retrieve(
    query: str,
    k: int = 5,
    doc_type: str | None = None,
    source: str | None = None,
    where: dict | None = None,
):

    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    collection = get_collection()   # <==== CRÍTICO

    # Construimos diccionario de filtros
    filters = {}

    if doc_type is not None:
        filters["doc_type"] = doc_type

    if source is not None:
        filters["source"] = source

    if where is not None:
        filters.update(where)

    query_kwargs = {
        "query_texts": [normalized_query],
        "n_results": k,
    }

    if filters:
        query_kwargs["where"] = filters

    # Ejecutar consulta
    results = collection.query(**query_kwargs)

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
