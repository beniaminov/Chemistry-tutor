#!/usr/bin/env python
"""Download and locally benchmark multilingual-e5-base on synthetic chemistry text."""

import argparse
import sys
import threading
import time

import psutil
from sentence_transformers import SentenceTransformer

MODEL_ID = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"

PASSAGES = [
    "passage: Атом состоит из положительно заряженного ядра и электронной оболочки.",
    "passage: В молекуле воды H2O атом кислорода связан с двумя атомами водорода.",
    "passage: При реакции нейтрализации кислота реагирует с основанием с образованием соли и воды.",
    "passage: Валентность показывает число химических связей, образуемых атомом элемента.",
]

QUERIES = [
    "query: Из каких частей состоит атом?",
    "query: Как записывается формула воды?",
    "query: Что образуется при реакции кислоты с основанием?",
]


def memory_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024 * 1024)


def sample_memory(process: psutil.Process, stop: threading.Event, peak: list[float]) -> None:
    while not stop.wait(0.1):
        peak[0] = max(peak[0], memory_mb(process))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 1 <= args.max_seq_length <= 512:
        parser.error("--max-seq-length must be between 1 and 512")

    process = psutil.Process()
    print(f"Python process RSS before loading: {memory_mb(process):.0f} MiB")
    peak_rss = [memory_mb(process)]
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=sample_memory,
        args=(process, stop_sampling, peak_rss),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    model = SentenceTransformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
    )
    model.max_seq_length = args.max_seq_length
    print(f"Model loaded in {time.perf_counter() - started:.1f} s")
    print(f"Python process RSS after loading: {memory_mb(process):.0f} MiB")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")

    started = time.perf_counter()
    passage_vectors = model.encode(
        PASSAGES,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    passage_seconds = time.perf_counter() - started

    started = time.perf_counter()
    query_vectors = model.encode(
        QUERIES,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_seconds = time.perf_counter() - started

    if passage_vectors.shape != (len(PASSAGES), 768):
        raise RuntimeError(f"Unexpected passage embedding shape: {passage_vectors.shape}")
    if query_vectors.shape != (len(QUERIES), 768):
        raise RuntimeError(f"Unexpected query embedding shape: {query_vectors.shape}")

    similarities = query_vectors @ passage_vectors.T
    peak_rss[0] = max(peak_rss[0], memory_mb(process))
    stop_sampling.set()
    sampler.join()
    print(f"Encoded passages: {len(PASSAGES)} in {passage_seconds:.2f} s")
    print(f"Encoded queries: {len(QUERIES)} in {query_seconds:.2f} s")
    print(f"Peak observed process RSS: {peak_rss[0]:.0f} MiB")
    print(f"Cosine similarity matrix: {similarities.shape}")
    print("Top result per query (synthetic smoke test):")
    for query, scores in zip(QUERIES, similarities):
        best_index = int(scores.argmax())
        print(f"- {query.removeprefix('query: ')} -> passage {best_index + 1}")


if __name__ == "__main__":
    main()
