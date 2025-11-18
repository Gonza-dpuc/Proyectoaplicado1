import pandas as pd
from pathlib import Path

DATA_PROCESSED = Path("data/processed")

# Parámetros por defecto para el chunking
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 200
MIN_CHUNK_SIZE = 200  # tamaño mínimo aceptable para el último chunk


def chunk_text(text: str,
               chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP,
               min_chunk_size: int = MIN_CHUNK_SIZE):
    # Normalización básica de espacios
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())

    n = len(text)
    if n == 0:
        return [], []

    chunks = []
    starts = []
    start = 0

    while start < n:
        # Si lo que queda es muy pequeño, lo unimos al chunk anterior
        remaining = n - start
        if chunks and remaining < min_chunk_size:
            chunks[-1] = chunks[-1] + " " + text[start:]
            break

        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        starts.append(start)

        # Avanzamos con solapamiento
        start += max(chunk_size - overlap, 1)

    return chunks, starts


def create_chunks(chunk_size: int = DEFAULT_CHUNK_SIZE,
                  overlap: int = DEFAULT_OVERLAP,
                  min_chunk_size: int = MIN_CHUNK_SIZE):
    """
    Lee data/processed/corpus.parquet y genera data/processed/chunks.parquet
    con metadatos adicionales por chunk.
    """
    corpus_path = DATA_PROCESSED / "corpus.parquet"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"No se encontró {corpus_path}. "
            "Primero ejecuta: python src/ingest.py"
        )

    df = pd.read_parquet(corpus_path)

    rows = []
    num_docs = 0

    for _, row in df.iterrows():
        doc_id = row.get("id")
        source = row.get("source", "")
        text = row.get("text", "")

        if not isinstance(text, str) or not text.strip():
            # Saltar documentos sin texto útil
            continue

        num_docs += 1
        chunks, starts = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        )
        total_chunks = len(chunks)

        for i, (ch, start_pos) in enumerate(zip(chunks, starts)):
            rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "source": source,
                    "text": ch,
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "char_start": int(start_pos),
                    "char_end": int(start_pos + len(ch)),
                }
            )

    chunks_df = pd.DataFrame(rows)
    out_path = DATA_PROCESSED / "chunks.parquet"
    chunks_df.to_parquet(out_path, index=False)
    print(
        f"Generados {len(chunks_df)} chunks a partir de {num_docs} documentos. "
        f"Guardado en {out_path}"
    )
    return chunks_df


if __name__ == "__main__":
    create_chunks()
