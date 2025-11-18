import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")


def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def create_chunks():
    df = pd.read_parquet(DATA_PROCESSED / "corpus.parquet")
    rows = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["text"])
        for i, ch in enumerate(chunks):
            rows.append({
                "doc_id": row["id"],
                "chunk_id": f'{row["id"]}_chunk_{i}',
                "source": row["source"],
                "text": ch
            })
    chunks_df = pd.DataFrame(rows)
    chunks_df.to_parquet(DATA_PROCESSED / "chunks.parquet", index=False)
    print(f"Generados {len(chunks_df)} chunks.")
    return chunks_df


if __name__ == "__main__":
    create_chunks()
