
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 80
CHUNK_OVERLAP = 15
TOP_K = 3


def load_documents():
    documents = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "source": path.name,
                "text": text
            })

    if not documents:
        raise ValueError("No .txt documents found in data/raw.")

    return documents


def chunk_text(text):
    words = text.split()
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + CHUNK_SIZE])

        if chunk:
            chunks.append(chunk)

        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def prepare_chunks(documents):
    chunks = []

    for document in documents:
        for part_number, text in enumerate(
            chunk_text(document["text"])
        ):
            chunks.append({
                "id": len(chunks),
                "source": document["source"],
                "part": part_number,
                "text": text
            })

    return chunks


def main():
    documents = load_documents()
    chunks = prepare_chunks(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print("\nBaseline ready. Type 'exit' to stop.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        start = perf_counter()

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_vector = np.asarray(
            query_vector, dtype=np.float32
        )

        k = min(TOP_K, len(chunks))
        scores, ids = index.search(query_vector, k)

        elapsed_ms = (perf_counter() - start) * 1000

        print(f"\nQuery retrieval time: {elapsed_ms:.2f} ms")

        for rank, (score, idx) in enumerate(
            zip(scores[0], ids[0]), start=1
        ):
            if idx < 0:
                continue

            result = chunks[idx]

            print(f"\nRank: {rank}")
            print(f"Source: {result['source']}")
            print(f"Similarity: {score:.4f}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
