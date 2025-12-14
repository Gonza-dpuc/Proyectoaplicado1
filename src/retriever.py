from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, QDRANT_URL, QDRANT_API_KEY
from src.schemas import SearchResult, ChunkMetadata


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
# Obtener cliente de Qdrant
# -------------------------------------------------------------------------
def get_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# -------------------------------------------------------------------------
# Función principal de retrieval
# -------------------------------------------------------------------------
def retrieve(
    query: str,
    k: int = 5,
    doc_type: str | None = None,
    source: str | None = None,
    where: dict | None = None,
) -> list[SearchResult]:

    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    # Generar embedding de la consulta
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    embedding_response = openai_client.embeddings.create(
        input=normalized_query,
        model=EMBEDDING_MODEL
    )
    query_vector = embedding_response.data[0].embedding

    client = get_client()

    # Construimos filtros para Qdrant
    must_conditions = []

    if doc_type is not None:
        must_conditions.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))

    if source is not None:
        must_conditions.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))

    if where is not None:
        for key, value in where.items():
            must_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    # Ejecutar consulta
    results = client.search(
        collection_name="bioactives_chunks",
        query_vector=query_vector,
        query_filter=query_filter,
        limit=k
    )

    docs = []
    for hit in results:
        # Validamos que lo que viene de la DB cumpla con nuestro esquema
        meta = ChunkMetadata(**hit.payload)
        
        docs.append(SearchResult(
            chunk_id=str(hit.id),
            text=hit.payload.get("text", ""),
            metadata=meta,
            distance=hit.score
        ))
    return docs
