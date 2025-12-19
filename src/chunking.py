import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

from src.config import (
    PROCESSED_DIR,
    CHUNKS_HITO1_PATH,
    CHUNKS_HITO2_PATH,
)

# Priorizamos chunks.parquet porque en tu proyecto YA existe y su esquema es conocido
CANDIDATE_INPUTS = [
    PROCESSED_DIR / "chunks.parquet",
    PROCESSED_DIR / "corpus.parquet",
]


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stable_doc_id_from_source(source: str) -> str:
    """
    Si no existe doc_id en el parquet, creamos uno estable basado en la ruta del PDF.
    """
    s = (source or "").strip().encode("utf-8", errors="ignore")
    return hashlib.md5(s).hexdigest()


# -------------------------
# HITO 1: baseline (fijo por caracteres)
# -------------------------
def _chunk_text_fixed_chars(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    text = _clean_text(text)
    text_flat = " ".join(text.split())
    if not text_flat:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 6)

    chunks = []
    start = 0
    n = len(text_flat)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text_flat[start:end].strip()
        if chunk_text:
            chunks.append(
                {"content": chunk_text, "char_start": start, "char_end": end})

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


# -------------------------
# HITO 2: mejorado (párrafos + packing)
# -------------------------
def _split_paragraphs(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _pack_paragraphs_into_chunks(
    paragraphs: List[str], target_size: int = 1200, overlap_chars: int = 200
) -> List[Dict]:
    if not paragraphs:
        return []

    chunks: List[Dict] = []
    current = ""
    global_text = "\n\n".join(paragraphs)
    search_pos = 0

    def _find_span(sub: str, start_from: int) -> Tuple[int, int]:
        idx = global_text.find(sub, start_from)
        if idx == -1:
            return start_from, min(start_from + len(sub), len(global_text))
        return idx, idx + len(sub)

    for p in paragraphs:
        candidate = (current + "\n\n" + p).strip() if current else p

        if len(candidate) <= target_size:
            current = candidate
            continue

        if current:
            s, e = _find_span(current, search_pos)
            chunks.append({"content": current, "char_start": s, "char_end": e})
            search_pos = e

            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + p).strip() if tail else p
        else:
            # párrafo enorme -> fallback a fixed chars
            parts = _chunk_text_fixed_chars(
                p, chunk_size=target_size, overlap=overlap_chars)
            chunks.extend(parts)
            current = ""

    if current:
        s, e = _find_span(current, search_pos)
        chunks.append({"content": current, "char_start": s, "char_end": e})

    return chunks


def _chunk_text_hito2(text: str, target_size: int = 1200, overlap_chars: int = 200) -> List[Dict]:
    pars = _split_paragraphs(text)
    if not pars:
        return []
    return _pack_paragraphs_into_chunks(pars, target_size=target_size, overlap_chars=overlap_chars)


# -------------------------
# Input loader + schema detection
# -------------------------
def _load_input_df() -> pd.DataFrame:
    input_path = next((p for p in CANDIDATE_INPUTS if p.exists()), None)
    if input_path is None:
        raise FileNotFoundError(
            "No se encontró data/processed/chunks.parquet ni data/processed/corpus.parquet.\n"
            "Primero ejecuta tu pipeline de procesamiento (ingesta/chunking original)."
        )

    df = pd.read_parquet(input_path)
    print(f"[chunking] Input: {input_path} | filas: {len(df)}")
    print(f"[chunking] Columnas disponibles: {df.columns.tolist()}")
    return df


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    text_col = "text" if "text" in df.columns else (
        "content" if "content" in df.columns else None)
    source_col = "source" if "source" in df.columns else (
        "source_file" if "source_file" in df.columns else None)
    doc_id_col = "doc_id" if "doc_id" in df.columns else None

    if text_col is None or source_col is None:
        raise ValueError(
            f"No se pudo detectar columnas de texto/fuente. Columnas: {df.columns.tolist()}\n"
            f"Se espera 'text' o 'content' y 'source' o 'source_file'."
        )

    return text_col, source_col, doc_id_col


# -------------------------
# Build outputs
# -------------------------
def _build_chunks_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    text_col, source_col, doc_id_col = _detect_columns(df)

    records = []

    # Agrupar por doc (doc_id si existe, si no por source)
    if doc_id_col:
        group_key = doc_id_col
    else:
        group_key = source_col

    grouped = df.groupby(group_key, dropna=False)

    for g_idx, (g_key, g_df) in enumerate(grouped):
        # Definir doc_id y source_file
        if doc_id_col:
            doc_id = str(g_key)
            source_val = str(g_df[source_col].iloc[0]) if len(g_df) else ""
        else:
            source_val = str(g_key)
            doc_id = _stable_doc_id_from_source(source_val)

        # Construir texto completo del doc:
        # Si input ya venía chunked (chunks.parquet), lo concatenamos por chunk_index si existe
        if "chunk_index" in g_df.columns:
            g_df = g_df.sort_values("chunk_index")
        full_text = "\n\n".join([str(x)
                                for x in g_df[text_col].fillna("").tolist()])
        full_text = _clean_text(full_text)

        if mode == "hito1":
            doc_chunks = _chunk_text_fixed_chars(
                full_text, chunk_size=1200, overlap=200)
        elif mode == "hito2":
            doc_chunks = _chunk_text_hito2(
                full_text, target_size=1200, overlap_chars=200)
        else:
            raise ValueError("mode debe ser 'hito1' o 'hito2'")

        total_chunks = len(doc_chunks)
        source_file = source_val

        for i, c in enumerate(doc_chunks):
            records.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{i}",
                    "doc_id": doc_id,
                    "source_file": source_file,  # estandarizamos
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "char_start": c.get("char_start"),
                    "char_end": c.get("char_end"),
                    "content": c["content"],  # estandarizamos
                }
            )

        if (g_idx + 1) % 20 == 0:
            print(
                f"[chunking:{mode}] Procesados {g_idx+1}/{len(grouped)} documentos...")

    return pd.DataFrame(records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_in = _load_input_df()

    print("=== Generando chunks Hito 1 (baseline) ===")
    chunks_h1 = _build_chunks_df(df_in, mode="hito1")
    chunks_h1.to_parquet(CHUNKS_HITO1_PATH, index=False)
    print(f"[OK] chunks_hito1: {len(chunks_h1)} -> {CHUNKS_HITO1_PATH}")

    print("\n=== Generando chunks Hito 2 (mejorado) ===")
    chunks_h2 = _build_chunks_df(df_in, mode="hito2")
    chunks_h2.to_parquet(CHUNKS_HITO2_PATH, index=False)
    print(f"[OK] chunks_hito2: {len(chunks_h2)} -> {CHUNKS_HITO2_PATH}")

    print("\nChunking por hitos completado.")


if __name__ == "__main__":
    main()
