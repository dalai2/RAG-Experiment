"""
TF-IDF retrieval experiment — clean re-run.
Uses LangChain's RecursiveCharacterTextSplitter (same as the original notebook)
so that chunk counts with overlap > 0% are correct.
Saves to results_tfidf_clean.csv with the same schema as results.csv.
"""
import time, itertools, warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

BASE_DIR          = Path(__file__).parent
RESULTS_CLEAN_CSV = BASE_DIR / "results_tfidf_clean.csv"

CHUNK_SIZES      = [128, 256, 512, 1024, 2048]
OVERLAP_RATIOS   = [0.0, 0.10, 0.20]
N_SAMPLES        = 200
TOP_K            = 3
CHARS_PER_TOKEN  = 4
RETRIEVAL_METHOD = "TF-IDF bigrams + cosine"


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_squad(n: int = N_SAMPLES, seed: int = 42) -> List[Dict]:
    ds = load_dataset("rajpurkar/squad", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))
    return [
        {
            "question":     row["question"],
            "ground_truth": row["answers"]["text"][0],
            "context":      row["context"],
        }
        for row in ds
    ]


# ── Chunking via LangChain ─────────────────────────────────────────────────────
def build_chunks(
    samples: List[Dict], chunk_chars: int, overlap_chars: int
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for s in samples:
        chunks.extend(splitter.split_text(s["context"]))
    return chunks


# ── Metrics ────────────────────────────────────────────────────────────────────
def _relevant(chunk: str, gt: str) -> bool:
    return gt.strip().lower() in chunk.lower()


# ── Single-combination runner ──────────────────────────────────────────────────
def run_combination(
    samples: List[Dict], chunk_size: int, overlap_ratio: float, top_k: int = TOP_K
) -> Dict[str, Any]:
    chunk_chars   = chunk_size * CHARS_PER_TOKEN
    overlap_chars = int(chunk_chars * overlap_ratio)
    label = f"cs={chunk_size}, ov={int(overlap_ratio*100)}%"
    print(f"\n{'-'*55}\n  {label}\n{'-'*55}")

    # ── Build TF-IDF index ─────────────────────────────────────────────────────
    print("  [1/2] Chunking + fitting TF-IDF (bigrams)...")
    t0 = time.perf_counter()

    all_chunks     = build_chunks(samples, chunk_chars, overlap_chars)
    total_chunks   = len(all_chunks)
    avg_chunks_per_doc = total_chunks / len(samples)

    vectorizer    = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    chunk_matrix  = vectorizer.fit_transform(all_chunks)   # (n_chunks, vocab)
    build_time    = time.perf_counter() - t0

    print(f"        {total_chunks} chunks, vocab={chunk_matrix.shape[1]:,} "
          f"in {build_time:.2f}s  (avg {avg_chunks_per_doc:.2f}/doc)")

    # ── Retrieve & evaluate ────────────────────────────────────────────────────
    print(f"  [2/2] Retrieving for {len(samples)} queries...")
    hit_rates, mrr_scores, latencies = [], [], []

    for s in tqdm(samples, desc=label, leave=False):
        t_q   = time.perf_counter()
        q_vec = vectorizer.transform([s["question"]])
        sims  = cosine_similarity(q_vec, chunk_matrix).flatten()
        top_idx   = sims.argsort()[::-1][:top_k]
        retrieved = [all_chunks[i] for i in top_idx]
        latencies.append(time.perf_counter() - t_q)

        hr = float(any(_relevant(c, s["ground_truth"]) for c in retrieved))
        hit_rates.append(hr)

        rr = 0.0
        for rank, c in enumerate(retrieved, 1):
            if _relevant(c, s["ground_truth"]):
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

    result = {
        "chunk_size":         chunk_size,
        "overlap_ratio":      overlap_ratio,
        "overlap_tokens":     int(chunk_size * overlap_ratio),
        "hit_rate":           round(float(np.mean(hit_rates)), 6),
        "mrr":                round(float(np.mean(mrr_scores)), 6),
        "avg_latency_s":      round(float(np.mean(latencies)), 6),
        "p95_latency_s":      round(float(np.percentile(latencies, 95)), 6),
        "avg_chunks_per_doc": round(avg_chunks_per_doc, 3),
        "total_chunks":       total_chunks,
        "build_time_s":       round(build_time, 4),
        "n_samples":          len(samples),
        "top_k":              top_k,
        "retrieval_method":   RETRIEVAL_METHOD,
    }
    print(f"  OK hit_rate@{top_k}={result['hit_rate']:.3f}  "
          f"mrr@{top_k}={result['mrr']:.3f}  "
          f"avg_latency={result['avg_latency_s']:.4f}s")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading SQuAD validation split...")
    squad_samples = load_squad()
    print(f"Loaded {len(squad_samples)} samples.")

    # Resume support
    completed_keys = set()
    existing_df    = pd.DataFrame()
    if RESULTS_CLEAN_CSV.exists():
        existing_df    = pd.read_csv(RESULTS_CLEAN_CSV)
        completed_keys = set(zip(existing_df["chunk_size"], existing_df["overlap_ratio"]))
        print(f"Resuming: {len(completed_keys)} combos already done.")

    all_results = []
    for chunk_size, overlap_ratio in itertools.product(CHUNK_SIZES, OVERLAP_RATIOS):
        if (chunk_size, overlap_ratio) in completed_keys:
            print(f">> Skipping cs={chunk_size}, ov={overlap_ratio}")
            continue

        result = run_combination(squad_samples, chunk_size, overlap_ratio)
        all_results.append(result)

        new_df = pd.DataFrame(all_results)
        if not existing_df.empty:
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        new_df.to_csv(RESULTS_CLEAN_CSV, index=False)
        print(f"  Checkpointed: {RESULTS_CLEAN_CSV}")

    print(f"\nDone! {len(CHUNK_SIZES) * len(OVERLAP_RATIOS)} combos complete.")
    print(f"Saved: {RESULTS_CLEAN_CSV}\n")

    df = pd.read_csv(RESULTS_CLEAN_CSV)
    print(df[["chunk_size", "overlap_ratio", "total_chunks",
              "avg_chunks_per_doc", "hit_rate", "mrr",
              "avg_latency_s"]].to_string(index=False))
