# RAG Chunk Size Experiment — Retrieval Evaluation

Measures how `chunk_size` and `overlap` affect **retrieval quality** using standard IR metrics. No LLM generation, no API keys, $0 cost.

📊 **Interactive Dashboard:** [https://dalai2.github.io/RAG-Experiment/](https://dalai2.github.io/RAG-Experiment/)

## Metrics

| Metric | Definition |
|---|---|
| **Hit Rate @ K** | Fraction of queries where ≥1 retrieved chunk contains the ground-truth answer |
| **MRR @ K** | Mean Reciprocal Rank — `1/rank` of the first relevant chunk (0 if none in top-K) |
| **Avg latency** | Wall-clock time for embedding + vector search per query |
| **Avg chunks/doc** | Index size indicator across chunk configs |

**Relevance criterion:** a chunk is relevant if it contains the ground-truth answer string (case-insensitive substring). Valid for SQuAD because answers are verbatim spans extracted from the context.

---

## Experiment design

**Grid:** 5 chunk sizes × 3 overlap ratios = **15 combinations**

| Variable | Values |
|---|---|
| Chunk sizes (tokens) | 128, 256, 512, 1024, 2048 |
| Overlap | 0%, 10%, 20% of chunk size |
| Dataset | `rajpurkar/squad` — 200 Q&A pairs (validation split, seed=42) |
| Top-K | 3 |

---

## Stack & exact versions

| Component | Package | Version |
|---|---|---|
| Chunking + retrieval | `langchain` | 0.3.25 |
| LangChain community | `langchain-community` | 0.3.24 |
| LangChain ChromaDB | `langchain-chroma` | 0.2.4 |
| LangChain Ollama | `langchain-ollama` | 0.3.3 |
| Vector store | `chromadb` | 0.6.3 |
| Dataset | `datasets` | 3.6.0 |
| Data | `pandas` | 2.2.3 |
| Visualization | `seaborn` | 0.13.2 |
| Visualization | `matplotlib` | 3.10.3 |
| Progress bar | `tqdm` | 4.67.1 |
| Python | — | 3.11.x |

**Ollama model:** `nomic-embed-text` (latest as of 2025-01)
- Model card: https://ollama.com/library/nomic-embed-text
- Context window: 8192 tokens
- Embedding dimension: 768

---

## Reproducibility setup

### 1. Install Ollama & pull the embedding model

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: https://ollama.com/download

ollama pull nomic-embed-text
ollama serve   # default: http://localhost:11434
```

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install \
    langchain==0.3.25 \
    langchain-community==0.3.24 \
    langchain-chroma==0.2.4 \
    langchain-ollama==0.3.3 \
    chromadb==0.6.3 \
    datasets==3.6.0 \
    pandas==2.2.3 \
    seaborn==0.13.2 \
    matplotlib==3.10.3 \
    tqdm==4.67.1
```

No API keys needed.

### 3. Run

```bash
jupyter lab notebook.ipynb
```

Or headless:

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb
```

> **Estimated runtime:** ~10–20 min (embedding 200 docs × 15 configs is the bottleneck).  
> **Cost:** $0.

---

## Output files

```
rag-chunk-experiment/
├── notebook.ipynb
├── results.csv
├── figures/
│   ├── heatmap_hit_rate.png               # Hit Rate: chunk × overlap
│   ├── heatmap_mrr.png                    # MRR: chunk × overlap
│   ├── latency_vs_chunksize.png           # Line chart with P95 band
│   └── retrieval_quality_vs_chunksize.png # Hit Rate + MRR side-by-side
└── README.md
```

---

## Notes

- Chunk size is specified in **tokens**, approximated as `chunk_size × 4 chars` (matching nomic-embed-text's BPE tokenization).
- All ChromaDB collections are **in-memory** — no disk state between combinations.
- The notebook supports **resuming**: existing rows in `results.csv` are skipped automatically.
- The experiment intentionally isolates the retrieval step. If you want to measure end-to-end generation quality (faithfulness, answer relevancy), a natural extension is to add a local Ollama LLM (e.g. `llama3.2:3b`) and RAGAS evaluation on top of these results.
