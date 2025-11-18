import os
import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def load_text_files():
    docs = []
    for fp in DATA_RAW.glob("*.txt"):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        docs.append({"id": fp.stem, "source": str(fp), "text": text})
    return docs


def load_csv_files():
    docs = []
    for fp in DATA_RAW.glob("*.csv"):
        df = pd.read_csv(fp)
        for i, row in df.iterrows():
            text = " ".join(str(row[c]) for c in df.columns)
            docs.append({
                "id": f"{fp.stem}_{i}",
                "source": str(fp),
                "text": text
            })
    return docs


def basic_clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def build_corpus():
    docs = load_text_files() + load_csv_files()
    for d in docs:
        d["text"] = basic_clean(d["text"])
    return docs


if __name__ == "__main__":
    corpus = build_corpus()
    df = pd.DataFrame(corpus)
    df.to_parquet(DATA_PROCESSED / "corpus.parquet", index=False)
    print(f"Corpus con {len(df)} documentos guardado.")
