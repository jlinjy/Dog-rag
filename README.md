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
- **Custom evaluation harness** — measures **retrieval recall and generation quality separately** across 14 questions (multi-hop, aggregation, disambiguation, conditional, grounding, plus factual controls).
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

Latest run (2026-11-24, `top_k=4`; generation: llama3.2 at temperature 0.1; judge: qwen2.5:7b at temperature 0):

| Metric | Score |
|---|---|
| Retrieval recall (10 questions with expected breeds) | **100%** |
| LLM-as-judge (mean fraction of key facts conveyed, 13 questions) | **0.69** |
| Fact substring match (mean) | 0.61 |
| Answers using outside / "general knowledge" | 0 of 14 |

**How the judge scores.** Each question has a short list of key facts. The judge returns a yes/no verdict per fact, each backed by a short quote from the answer, and the score is computed in code as the fraction of "yes". It cannot invent its own criteria, and "not enough information" answers score 0 on the facts they don't cover. This replaced a single holistic score that echoed the fact list back (it once credited "hypoallergenic" to an answer that never mentioned it) and scored refusals 1.0. Each answer is also checked for phrases like "based on general knowledge" that signal it stepped outside the retrieved context.

**Where it falls short:**
- *Generation, not retrieval, is now the bottleneck.* After query decomposition was changed to search each breed with the rest of the question (e.g. "siberian husky … hot climate" rather than just "siberian husky"), the relevant chunks are retrieved for all comparison questions — but llama3.2 doesn't always use them (Labrador vs. Golden tail differences are retrieved but not mentioned).
- *Key facts check topics, not conclusions.* For "Which is better suited to a hot climate, a Siberian Husky or a Chihuahua?" the model retrieves the right chunk but concludes the Husky — and still scores 0.5–1.0. Catching this needs an expected answer per question.
- *The judge still misses some facts* written out verbatim (e.g. "Toy Poodle" in a list), so the judge and substring scores are best read together.

> Note: the evaluation set is deliberately small (14 questions: 11 across 6 question types plus 3 controls); results are treated as directional, and growing the set and validating the judge against human labels are the natural next steps.

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
# 1. start Postgres (pgvector), the RAG service, and Prometheus
docker compose -f deploy/compose/docker-compose.yml --env-file .env up -d --build

# 2. build the corpus (fetch breed articles) and embed into pgvector
cd services/rag
python -m app.ingest.scrape_corpus
python -m app.ingest.embed_corpus
python -m app.ingest.load_dog_breeds   # dog_breeds.csv -> dog_breeds table (merges duplicate breeds)

# 3. query the API
#    open http://localhost/docs  and call /ask
```

Run the evaluation harness (from the repo root):

```bash
ollama pull qwen2.5:7b                                   # judge model
PYTHONPATH=services/rag python -m evals.retrieval.run    # optional arg: top_k (default 4)
# report written to evals/reports/rag_only_<timestamp>.txt
```

Run the tests:

```bash
cd services/rag && pytest -v
```

---

## Project structure

```
services/
  rag/                         # RAG service (API: /ask, /health, /metrics)
    app/
      api.py                   # FastAPI app
      retrieval.py             # embedder, DB connection, pgvector search
      decomposition.py         # multi-breed query decomposition
      generation.py            # prompt building + LLM generation
      ingest/                  # scrape_corpus, embed_corpus, load_dog_breeds, text_utils
    data/                      # dog_breeds.csv, corpus/ (gitignored)
    scripts/                   # ad-hoc sanity scripts
    tests/
    Dockerfile
  agent/                       # agent service (scaffold, WIP)
evals/
  datasets/                    # retrieval.jsonl, agent.jsonl
  retrieval/                   # retrieval + LLM-as-judge harness
  agent/                       # agent evals (WIP)
  reports/
deploy/
  compose/                     # docker-compose.yml, init.sql, prometheus.yml
  k8s/                         # rag/, agent/, postgres/, kustomization.yaml
.github/workflows/             # ci.yml, eval.yml, cd.yml
docs/adr/                      # architecture decision records
pyproject.toml                 # uv workspace
```


---

## Possible next steps

- Grow the evaluation set (30–50 questions) and validate the LLM-as-judge against human-labelled examples.
- Add a hybrid router sending structured/factual queries to SQL and open questions to RAG.
- Response streaming to reduce perceived latency; add retrieval **precision** alongside recall.
