# What's the Best Chunk Size for RAG? We Ran the Numbers.

If you've built a RAG (Retrieval-Augmented Generation) pipeline, you've faced the chunking question. Should you split documents into 256-token chunks? 512? 1024? And does it matter if the chunks overlap?

The internet gives you rules of thumb. We wanted actual data. So we ran a controlled experiment — comparing TF-IDF and semantic embeddings across a full grid of chunk sizes and overlaps. Here's what we found and what it means for your pipeline.

---

## The Setup

We used the SQuAD dataset — 200 question/answer pairs grounded in Wikipedia paragraphs. It's a clean, well-understood benchmark: each question has a specific passage that answers it, which makes hit rate a meaningful metric.

We tested five chunk sizes — **128, 256, 512, 1024, and 2048 tokens** — crossed with three overlap levels — **0%, 10%, and 20%** — giving us 15 combinations per retriever. We ran the full grid twice: once with TF-IDF, once with semantic embeddings.

For **TF-IDF**, we used bigrams and cosine similarity — the classic bag-of-words baseline. For **embeddings**, we used `sentence-transformers/all-MiniLM-L6-v2` with ChromaDB as the vector store. Small model, fast inference, broadly used in production.

Our primary metrics were **Hit Rate@3** (is the answer chunk in the top 3 results?) and **MRR@3** (Mean Reciprocal Rank — where in the top 3 does it land?). Hit Rate@3 is the headline number we'll use throughout.

### Chunking the documents

All chunking was done with LangChain's `RecursiveCharacterTextSplitter`, which splits on natural boundaries (paragraphs, sentences, words) before falling back to hard character splits. The `chunk_overlap` parameter controls how many tokens each chunk shares with its neighbor.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # max tokens per chunk
    chunk_overlap=51,     # ~10% overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.split_text(document_text)
```

Swap `chunk_size` and `chunk_overlap` to try different grid points — that's the entire variable across our experiment.

---

## TF-IDF: More Tokens = More Signal, Up to a Point

Here's what the TF-IDF results look like across chunk sizes (averaging over overlap settings, since overlap barely matters here — more on that in a moment):

| Chunk size | Hit Rate@3 |
|------------|-----------|
| 128 tokens | 87.0% |
| 256 tokens | ~91.0–91.5% |
| 512 tokens | ~92.0–92.5% |
| 1024 tokens | ~92.5% |
| 2048 tokens | ~92.5% |

The pattern is clear: **more tokens help, until they don't.** TF-IDF is a term-matching method. It works by finding chunks that share vocabulary with the query. Short chunks (128 tokens) don't have much vocabulary to match against — you get a lot of near-misses. As chunks grow to 512 tokens, you pick up more of the relevant terms and the hit rate climbs.

But once you hit 1024 tokens, you're not adding useful signal anymore. The retriever has already found the terms it needs; the extra context just adds noise. The ceiling sits at **92.5%**, and you reach it comfortably at 512 tokens.

On overlap: it doesn't move the needle for TF-IDF. Whether you use 0% or 20% overlap, the numbers barely budge. That makes sense — TF-IDF cares about term frequency, not about whether a chunk has a few tokens borrowed from its neighbor.

### Building the TF-IDF retriever

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
chunk_matrix = vectorizer.fit_transform(chunks)

def retrieve(query, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, chunk_matrix).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]
```

No external dependencies beyond sklearn. Fast to build, easy to debug — a solid baseline for any retrieval system.

---

## Embeddings: The Same Sweet Spot, Different Reasons

The embedding results look similar on the surface — 512 tokens is still the sweet spot — but the mechanism and the ceiling are both different.

| Chunk size | Hit Rate@3 (best overlap) |
|------------|--------------------------|
| 128 tokens | 93.0% |
| 256 tokens | ~93.0–93.5% |
| 512 tokens | **95.0%** (10% overlap) |
| 1024 tokens | ~94.5% |
| 2048 tokens | ~94.5% |

A few things stand out.

First, **embeddings start higher.** Even at 128 tokens — where TF-IDF only manages 87% — the embedding model hits 93%. Semantic similarity is more forgiving of short chunks because the model compresses meaning into a dense vector rather than relying on exact term overlap. A short chunk can still carry enough semantic signal to match a query.

Second, **512 tokens is the peak, but then it drops back slightly.** This is the embedding model's sweet spot for semantic density. At 128–256 tokens, chunks are a little thin — there's enough signal, but not optimal. At 512 tokens, you get a coherent semantic unit. Push to 1024 or 2048 and you're diluting the representation — one vector now has to capture too many ideas at once, and it gets blurrier.

Third — and this is the interesting one — **overlap actually matters here.** At 512 tokens, adding 10% overlap bumps hit rate by about 0.5–1 percentage point compared to no overlap. The effect is small but consistent. Why? Overlapping chunks reduce the chance that an answer straddles a chunk boundary. For semantic search, where you're looking for meaning rather than keywords, a slightly incomplete semantic unit can cost you a retrieval. Overlap is cheap insurance against that.

### Building the embedding retriever

```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("rag_chunks")

# Index chunks
embeddings = model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(chunks))],
)

def retrieve(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]
```

ChromaDB runs fully in-memory by default, so there's no server to spin up for prototyping. Swap in `chromadb.PersistentClient` when you're ready to persist the index.

---

## Head-to-Head: 2–6 Points Across the Board

Across every chunk size we tested, embeddings outperform TF-IDF:

| Chunk size | TF-IDF | Embeddings | Delta |
|------------|--------|-----------|-------|
| 128 tokens | 87.0% | 93.0% | +6.0pp |
| 256 tokens | ~91.3% | ~93.3% | +2.0pp |
| 512 tokens | ~92.3% | **95.0%** | +2.7pp |
| 1024 tokens | 92.5% | 94.5% | +2.0pp |
| 2048 tokens | 92.5% | 94.5% | +2.0pp |

The gap is largest at the smallest chunk size, where TF-IDF struggles most. It narrows in the middle as TF-IDF catches up, but never closes.

The ceiling tells the story: **TF-IDF tops out at 92.5%; embeddings top out at 95%.** For a system where you're calling an LLM to generate answers, those extra 2.5 points at the top are the ones that matter most — they're the hard cases where the query and the answer passage don't share obvious vocabulary.

One thing that surprised us: **latency was essentially identical.** Both methods run in approximately 1ms per query at this scale. Chunk size had no measurable effect. If you're choosing between these approaches in a prototype, performance is not a factor — at small-to-medium scale, you won't feel the difference.

> 📊 The full results CSVs (`results_tfidf_clean.csv` and `results_embeddings.csv`) with all 15 chunk × overlap combinations and MRR@3 scores are available on HuggingFace. Link in the comments.

---

## Practical Recommendations

Based on what we found, here's what we'd actually do:

**Use embeddings over TF-IDF if you can.** The 2–6pp improvement is real and consistent. `all-MiniLM-L6-v2` is fast and free. There's no good reason to ship a TF-IDF retriever if you have access to a sentence transformer.

**Default to 512 tokens.** This is the sweet spot for both methods. It's not coincidence — 512 tokens is roughly a coherent paragraph or two, which is the natural unit of meaning in most documents.

**Add 10% overlap if you're using embeddings.** It's a cheap win. For TF-IDF, don't bother.

**Don't go above 1024 tokens.** You lose retrieval precision without gaining meaningful context. If your use case needs long context, that's a job for the LLM's context window, not the retriever.

---

## What's Next

This experiment was controlled in ways that real deployments aren't. SQuAD questions are clean, single-hop, and well-matched to their source passages. Real user queries are messier. Real documents are longer, have varying density, and don't come pre-chunked.

A few natural extensions we're planning: testing on longer-form documents (technical docs, legal text) where 512-token chunks may behave differently; comparing `all-MiniLM-L6-v2` against larger embedding models to see how much the model matters at each chunk size; and running the same grid on a multi-hop QA benchmark where the answer requires combining two passages.

If you want to run this yourself, the code is straightforward — a grid search over chunk sizes with a fixed evaluation loop. Start with the three snippets above, wire them into a loop over the chunk sizes, and measure hit rate against your ground-truth passages.
