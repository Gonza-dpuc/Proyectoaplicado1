from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from config import OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_DIR

DATA_PROCESSED = Path("data/processed")


def build_chroma_index():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        "bioactives_chunks",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        )
    )

    df = pd.read_parquet(DATA_PROCESSED / "chunks.parquet")
    collection.add(
        ids=df["chunk_id"].tolist(),
        documents=df["text"].tolist(),
        metadatas=[{
            "doc_id": d,
            "source": s
        } for d, s in zip(df["doc_id"], df["source"])]
    )
    print(f"Index construido con {len(df)} chunks.")


if __name__ == "__main__":
    build_chroma_index()
