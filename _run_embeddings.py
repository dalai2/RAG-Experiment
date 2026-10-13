"""Standalone runner for the embeddings experiment.
Produces results_embeddings.csv with the same schema as results.csv.
"""
import time, itertools, warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb

warnings.filterwarnings("ignore")

BASE_DIR        = Path(__file__).parent
RESULTS_EMB_CSV = BASE_DIR / "results_embeddings.csv"

CHUNK_SIZES      = [128, 256, 512, 1024, 2048]
OVERLAP_RATIOS   = [0.0, 0.10, 0.20]
N_SAMPLES        = 200
TOP_K            = 3
CHARS_PER_TOKEN  = 4
EMB_MODEL_NAME   = "all-MiniLM-L6-v2"
RETRIEVAL_METHOD = f"sentence-transformers/{EMB_MODEL_NAME}"


# ── Text splitter ──────────────────────────────────────────────────────────────
def split_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    chunks, start = [], 0
    while True:
        end   = min(start + chunk_chars, len(text))
        chunk = text[start:end]
        cut   = end  # nominal boundary (before sentence-aware trimming)
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                idx = chunk.rfind(sep)
                if idx > chunk_chars // 2:
                    chunk = chunk[:idx + len(sep)]
                    cut   = start + idx + len(sep)
                    break
        if chunk.strip():
            chunks.append(chunk.strip())
        if end == len(text):
            break                          # reached end of text — stop
        start = max(cut - overlap_chars, start + 1)
    return chunks


# ── Build ChromaDB ─────────────────────────────────────────────────────────────
def build_vectorstore_emb(
    samples: List[Dict], chunk_size: int, overlap_ratio: float,
    embed_model: SentenceTransformer,
) -> tuple:
    chunk_chars   = chunk_size * CHARS_PER_TOKEN
    overlap_chars = int(chunk_chars * overlap_ratio)

    all_chunks, all_ids = [], []
    for i, s in enumerate(samples):
        for j, chunk in enumerate(split_text(s["context"], chunk_chars, overlap_chars)):
            all_chunks.append(chunk)
            all_ids.append(f"s{i}_c{j}")

    embeddings = embed_model.encode(all_chunks, batch_size=64, show_progress_bar=False)

    client = chromadb.EphemeralClient()
    col    = client.create_collection("exp", metadata={"hnsw:space": "cosine"})
    batch  = 5000
    for i in range(0, len(all_chunks), batch):
        col.add(
            documents=all_chunks[i:i+batch],
            embeddings=embeddings[i:i+batch].tolist(),
            ids=all_ids[i:i+batch],
        )
    return col, client, len(all_chunks)


# ── Metrics ────────────────────────────────────────────────────────────────────
def _relevant(chunk: str, gt: str) -> bool:
    return gt.strip().lower() in chunk.lower()

def hit_rate_fn(chunks: List[str], gt: str) -> float:
    return float(any(_relevant(c, gt) for c in chunks))

def mrr_fn(chunks: List[str], gt: str) -> float:
    for rank, c in enumerate(chunks, 1):
        if _relevant(c, gt):
            return 1.0 / rank
    return 0.0


# ── Single-combo runner ────────────────────────────────────────────────────────
def run_combination_emb(
    samples: List[Dict], chunk_size: int, overlap_ratio: float,
    embed_model: SentenceTransformer, top_k: int = TOP_K,
) -> Dict[str, Any]:
    label = f"cs={chunk_size}, ov={int(overlap_ratio*100)}%"
    print(f"\n{'-'*55}\n  {label}\n{'-'*55}")

    print("  [1/2] Building vector store...")
    t0 = time.perf_counter()
    col, client, total_chunks = build_vectorstore_emb(
        samples, chunk_size, overlap_ratio, embed_model
    )
    build_time         = time.perf_counter() - t0
    avg_chunks_per_doc = total_chunks / len(samples)
    print(f"        {total_chunks} chunks in {build_time:.1f}s (avg {avg_chunks_per_doc:.1f}/doc)")

    print(f"  [2/2] Retrieving for {len(samples)} queries...")
    hit_rates, mrr_scores, latencies = [], [], []
    q_embs = embed_model.encode(
        [s["question"] for s in samples], batch_size=64, show_progress_bar=False
    )

    for s, q_emb in tqdm(zip(samples, q_embs), total=len(samples), desc=label, leave=False):
        t_q = time.perf_counter()
        res = col.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        latencies.append(time.perf_counter() - t_q)
        retrieved = res["documents"][0]
        hit_rates.append(hit_rate_fn(retrieved, s["ground_truth"]))
        mrr_scores.append(mrr_fn(retrieved, s["ground_truth"]))

    result = {
        "chunk_size":         chunk_size,
        "overlap_ratio":      overlap_ratio,
        "overlap_tokens":     int(chunk_size * overlap_ratio),
        "hit_rate":           float(np.mean(hit_rates)),
        "mrr":                float(np.mean(mrr_scores)),
        "avg_latency_s":      float(np.mean(latencies)),
        "p95_latency_s":      float(np.percentile(latencies, 95)),
        "avg_chunks_per_doc": avg_chunks_per_doc,
        "total_chunks":       total_chunks,
        "build_time_s":       build_time,
        "n_samples":          len(samples),
        "top_k":              top_k,
        "retrieval_method":   RETRIEVAL_METHOD,
    }
    print(f"  OK hit_rate@{top_k}={result['hit_rate']:.3f}  "
          f"mrr@{top_k}={result['mrr']:.3f}  "
          f"avg_latency={result['avg_latency_s']:.4f}s")

    client.delete_collection("exp")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading {EMB_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMB_MODEL_NAME)
    print(f"Model ready — dim={embed_model.get_sentence_embedding_dimension()}")

    print("\nLoading SQuAD validation split ...")
    ds = load_dataset("rajpurkar/squad", split="validation")
    ds = ds.shuffle(seed=42).select(range(N_SAMPLES))
    squad_samples = [
        {
            "question":     row["question"],
            "ground_truth": row["answers"]["text"][0],
            "context":      row["context"],
        }
        for row in ds
    ]
    print(f"Loaded {len(squad_samples)} samples.")

    # Resume support
    completed_keys = set()
    existing_df    = pd.DataFrame()
    if RESULTS_EMB_CSV.exists():
        existing_df    = pd.read_csv(RESULTS_EMB_CSV)
        completed_keys = set(zip(existing_df["chunk_size"], existing_df["overlap_ratio"]))
        print(f"Resuming: {len(completed_keys)} combos already done.")

    all_results = []
    for chunk_size, overlap_ratio in itertools.product(CHUNK_SIZES, OVERLAP_RATIOS):
        if (chunk_size, overlap_ratio) in completed_keys:
            print(f">> Skipping cs={chunk_size}, ov={overlap_ratio}")
            continue

        result = run_combination_emb(squad_samples, chunk_size, overlap_ratio, embed_model)
        all_results.append(result)

        new_df = pd.DataFrame(all_results)
        if not existing_df.empty:
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        new_df.to_csv(RESULTS_EMB_CSV, index=False)
        print(f"  Checkpointed: {RESULTS_EMB_CSV}")

    print(f"\nDone! {len(CHUNK_SIZES)*len(OVERLAP_RATIOS)} combos complete.")
    print(f"Saved: {RESULTS_EMB_CSV}")
    df = pd.read_csv(RESULTS_EMB_CSV)
    print(df[["chunk_size", "overlap_ratio", "hit_rate", "mrr", "avg_latency_s"]].to_string(index=False))
