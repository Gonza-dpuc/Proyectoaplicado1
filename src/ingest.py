from pathlib import Path
import pandas as pd
from pypdf import PdfReader

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_text_files():
    docs = []
    for fp in DATA_RAW.rglob("*.txt"):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        docs.append({"id": fp.stem, "source": str(fp), "text": text})
    return docs


def load_csv_files():
    docs = []
    for fp in DATA_RAW.rglob("*.csv"):
        df = pd.read_csv(fp)
        # Convertimos cada fila en un bloque de texto uniendo columnas
        for i, row in df.iterrows():
            row_text = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            docs.append(
                {
                    "id": f"{fp.stem}_{i}",
                    "source": str(fp),
                    "text": row_text,
                }
            )
    return docs


def load_pdf_files():
    docs = []
    for fp in DATA_RAW.rglob("*.pdf"):
        try:
            reader = PdfReader(fp)
            pages_text = []
            for page in reader.pages:
                pages_text.append(page.extract_text() or "")
            full_text = "\n".join(pages_text)
            docs.append({"id": fp.stem, "source": str(fp), "text": full_text})
        except Exception as e:
            print(f"[WARN] No se pudo leer {fp}: {e}")
    return docs


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def build_corpus():
    docs = load_text_files() + load_csv_files() + load_pdf_files()
    for d in docs:
        d["text"] = basic_clean(d["text"])
    return docs


if __name__ == "__main__":
    corpus = build_corpus()
    df = pd.DataFrame(corpus)
    df.to_parquet(DATA_PROCESSED / "corpus.parquet", index=False)
    print(f"Corpus con {len(df)} documentos guardado.")
