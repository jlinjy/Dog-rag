# Dog-Breed RAG System

A retrieval-augmented generation (RAG) system that answers natural-language questions about dog breeds, built end-to-end from scratch: corpus ingestion, embedding, retrieval, generation, evaluation, containerised serving, and deployment to Kubernetes.

The focus of this project is **the engineering around the model, not the model itself** — retrieval quality, measurable evaluation, and production-style serving and operations.

---

## Architecture

```
          ┌───────────────────────────────────────────────┐
  query → │  FastAPI service                              │
          │   ├─ retrieval (pgvector similarity search)   │
          │   │     └─ query decomposition for multi-hop  │
          │   └─ generation (LLM via Ollama)              │ → grounded answer
          └───────────────────────────────────────────────┘
                 │                         │
         ┌───────▼────────┐        ┌───────▼────────┐
         │  PostgreSQL /  │        │  Prometheus    │
         │  pgvector      │        │  (monitoring)  │
         └────────────────┘        └────────────────┘

  Corpus pipeline:  103 Wikipedia breed articles  →  chunking  →
                    sentence-transformers embeddings  →  pgvector
```

---

## Key features

- **End-to-end RAG pipeline** — 103 Wikipedia dog-breed articles scraped (`wikipedia-api`), chunked, embedded with `sentence-transformers` (all-MiniLM-L6-v2), and stored in **pgvector**; retrieval + LLM generation (**Ollama**, local) served behind a **FastAPI** application.
- **Custom evaluation harness** — measures **retrieval recall and generation quality separately** across 13 questions spanning 5 question types (factual, multi-hop, disambiguation, conditional, grounding).
- **Query decomposition** for multi-hop comparison questions — detects multiple entities in a query, retrieves per-entity, and merges the context so each entity is represented.
- **LLM-as-judge evaluation** — upgraded from substring fact-matching to semantic judging, with prompt calibration to correct judge bias.
- **Containerised and orchestrated** — Docker Compose for local development; deployed to a **Kubernetes** (kind) cluster with health/readiness probes and resource limits.
- **CI/CD** — GitHub Actions running automated tests (PyTest) and reproducible Docker builds.
- **Monitoring** — Prometheus exposing request latency and per-phase (retrieval vs. generation) timing.

---

## Tech stack

| Layer | Tools |
|---|---|
| Retrieval | pgvector (PostgreSQL), sentence-transformers (all-MiniLM-L6-v2) |
| Generation | Ollama (local LLM) |
| Serving | FastAPI |
| Evaluation | Custom harness (retrieval recall + fact coverage), LLM-as-judge |
| Containerisation | Docker, Docker Compose |
| Orchestration | Kubernetes (kind), kubectl |
| CI/CD | GitHub Actions, PyTest |
| Monitoring | Prometheus |
| Corpus | wikipedia-api |

---

## Results & evaluation

Evaluation is split into two independently-measured stages so failures can be localised:

- **Retrieval recall** — of the documents that *should* be retrieved for a question, how many were. Improved from **83% → 100%** on the breed-specific question set after fixing a vector-index bug (see below).
- **Generation quality** — whether the answer conveys the expected facts, scored first by substring matching and later by an LLM-as-judge.

> Note: the evaluation set is deliberately small (13 questions) and spans 5 question types for coverage; results are treated as directional, and growing the set and validating the judge against human labels are the natural next steps.

---

## Engineering decisions & debugging

- **Approximate-index bug (IVFFlat).** Retrieval was silently returning wrong nearest neighbours on the small (~1k-vector) corpus — the true nearest neighbour was being beaten by a worse approximate match. Diagnosed by comparing cosine distances with the index scan enabled vs. disabled. **Fix:** dropped the ANN index, since exact search is instant and correct at this scale. Retrieval recall went from 83% to 100%.
- **Measuring retrieval and generation separately.** A single end-to-end score can't tell you which layer failed — bad retrieval (the model never saw the right context) vs. bad generation (it had the context but answered wrong). Scoring them independently makes failures debuggable.
- **LLM-as-judge calibration.** The judge initially penalised correct-but-concise answers (verbosity bias), caught by having it output a *reason* alongside each score and reading those reasons. Fixed by constraining the judge prompt to score only on factual coverage.
- **Query decomposition for multi-hop.** A single query vector for a comparison ("compare X and Y") tends to retrieve chunks for only one entity. Decomposing into per-entity sub-queries guarantees both are represented — a more correct fix than simply increasing top-k.
- **Kubernetes debugging.** Exercised real failure modes: `CrashLoopBackOff` from missing environment variables (diagnosed via `kubectl logs --previous`), `OOMKilled` pods from under-set memory limits (diagnosed via `kubectl describe pod`), and rolling-update behaviour keeping the old pod alive under a bad config.
- **Image optimisation.** Default PyTorch pulls multi-GB GPU libraries unnecessary for CPU inference; pinning CPU-only PyTorch shrank the image substantially and fixed CI disk limits.

---

## Running locally

> Prerequisites: Docker, Docker Compose, and [Ollama](https://ollama.com) running with a model pulled (e.g. `ollama pull llama3.2`).

```bash
# 1. start Postgres (pgvector), the app, and Prometheus
docker compose up -d --build

# 2. build the corpus (fetch breed articles)
python scrape_corpus.py

# 3. chunk and embed into pgvector
python embed_corpus.py

# 4. query the API
#    open http://localhost:8000/docs  and call /ask
```

Run the evaluation harness:

```bash
python eval.py
```

Run the tests:

```bash
pytest -v
```

---

## Project structure

```
scrape_corpus.py     # scrape breed articles into a local corpus
embed_corpus.py     # chunk, embed, and load into pgvector
text_utils.py       # pure text-processing functions (chunking, cleaning)
app/rag.py              # retrieval + generation logic, query decomposition
app/main.py             # FastAPI app (/ask, /health, /metrics)
app/eval.py             # evaluation harness (retrieval recall + LLM-as-judge)
eval_questions.json # evaluation set (13 questions, 5 types)
Dockerfile
docker-compose.yml
deployment.yaml                # Kubernetes deployment + service manifests
.github/workflows/  # CI pipeline
```


---

## Possible next steps

- Grow the evaluation set (30–50 questions) and validate the LLM-as-judge against human-labelled examples.
- Add a hybrid router sending structured/factual queries to SQL and open questions to RAG.
- Response streaming to reduce perceived latency; add retrieval **precision** alongside recall.
