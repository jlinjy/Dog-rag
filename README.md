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
# 1. start Postgres (pgvector), the RAG service, and Prometheus
docker compose -f deploy/compose/docker-compose.yml --env-file .env up -d --build

# 2. build the corpus (fetch breed articles) and embed into pgvector
cd services/rag
python -m app.ingest.scrape_corpus
python -m app.ingest.embed_corpus

# 3. query the API
#    open http://localhost/docs  and call /ask
```

### Agent service

`services/agent` is a LangChain agent (`create_agent`, Ollama `llama3.2`) that decides per question
which source to use:

- `breed_database` — text-to-SQL over `dog_breeds.csv` (read-only in-memory SQLite) for exact
  attributes and list/rank/count questions.
- `breed_articles` — calls the RAG service's `/ask` for descriptive/"why" questions; its grounded
  answer is returned directly (`return_direct`) rather than re-summarised.

```bash
# with the compose stack up, the agent is on :8080
curl -X POST localhost:8080/chat -H 'content-type: application/json' \
     -d '{"message": "Which breeds come from Germany?"}'
# -> {"answer": "...", "tools_used": ["breed_database"]}

# locally (RAG service on :80, Ollama running)
cd services/agent && uvicorn app.api:app --port 8080

# routing eval (from repo root)
PYTHONPATH=services/agent python -m evals.agent.run
```

Run the evaluation harness (from the repo root):

```bash
PYTHONPATH=services/rag python -m evals.retrieval.run
```

Run the tests:

```bash
cd services/rag && pytest -v
cd services/agent && pytest -v   # fake LLM, no Ollama/RAG needed
```

---

## Running on Kubernetes (kind)

Both services run in a local [kind](https://kind.sigs.k8s.io) cluster, with an
[Envoy Gateway](https://gateway.envoyproxy.io) (Gateway API) as the single entry point.
Only the agent is exposed; the RAG service is internal and reachable only by the agent.

```
localhost:80 ──▶ kind node :30080 ──▶ Envoy Gateway ──▶ dog-agent-svc ──▶ agent pod
                                                                              │ breed_articles
                                                               dograg-web-svc ──▶ RAG pod
Postgres (Docker Compose) and Ollama run on the host; pods reach them via host.docker.internal.
```

> Prerequisites: Docker Desktop, `kind`, `kubectl`, Ollama running with `llama3.2` pulled,
> a `.env` with the DB settings, and the corpus already embedded into Postgres (see above).

**Bring everything up** (from the repo root):

```bash
./deploy/kind/up.sh
```

The script:

1. starts Postgres via Docker Compose and checks Ollama is running,
2. creates the `dograg` cluster from `deploy/kind/cluster.yaml` if it doesn't exist
   (this config maps `127.0.0.1:80` into the cluster),
3. installs Envoy Gateway (Gateway API CRDs + controller),
4. builds `dograg-web:v1` and `dog-agent:v1` and loads them into kind,
5. applies `deploy/k8s` (kustomize) and waits for the rollouts and the gateway.

Re-run it after code changes; it reuses the existing cluster and restarts the deployments.

**Use it:**

- FastAPI docs: <http://localhost/docs>
- Query the agent:

  ```bash
  curl -X POST localhost/chat -H 'content-type: application/json' \
       -d '{"message": "Which breeds come from Germany?"}'
  # -> {"answer": "...", "tools_used": ["breed_database"]}
  ```

**Useful commands:**

```bash
kubectl get pods                                   # both should be 1/1 Running
kubectl logs deploy/dog-agent-deploy               # agent logs (same for dograg-web-deploy)
kubectl logs deploy/dograg-web-deploy --previous   # logs from a crashed container
kubectl describe pod <pod>                         # events, OOMKilled, probe failures
kubectl get gateway,httproute                      # gateway should be Programmed
kubectl port-forward svc/dograg-web-svc 8000:80    # reach the internal RAG API (localhost:8000/docs)
kind delete cluster --name dograg                  # tear down (Postgres data lives in Compose, untouched)
```

**Notes:**

- DB credentials live in the `dograg-db` Kubernetes Secret, which `up.sh` creates from
  `DB_NAME`, `DB_USER` and `DB_PASSWORD` in `.env`; no secret is committed. If you apply
  manually instead of using `up.sh`, create that Secret first. After changing `.env`, re-run
  `up.sh`: it updates the Secret and restarts the pods.
- The gateway's route timeout is 180s: an agent turn (LLM + RAG) can exceed Envoy's 15s default.
- Port 80 is shared with the Compose RAG service; run one stack or the other.
- To use the cluster, kind must have been created from `deploy/kind/cluster.yaml`; a cluster
  created without the port mapping can't serve `localhost`. Delete and re-run `up.sh` to fix.

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
      ingest/                  # scrape_corpus, embed_corpus, text_utils
    data/                      # dog_breeds.csv, corpus/ (gitignored)
    scripts/                   # ad-hoc sanity scripts
    tests/
    Dockerfile
  agent/                       # LangChain agent: routes to SQL (CSV) or RAG (POST /chat)
evals/
  datasets/                    # retrieval.jsonl, agent.jsonl
  retrieval/                   # retrieval + LLM-as-judge harness
  agent/                       # routing eval (accuracy, retries, call-limit hits)
  reports/
deploy/
  compose/                     # docker-compose.yml, init.sql, prometheus.yml
  kind/                        # cluster.yaml (port mapping), up.sh (one-command bring-up)
  k8s/                         # rag/, agent/, gateway/ (Envoy Gateway entry point), kustomization.yaml
.github/workflows/             # ci.yml, eval.yml, cd.yml
docs/adr/                      # architecture decision records
pyproject.toml                 # uv workspace
```


---

## Possible next steps

- Grow the evaluation set (30–50 questions) and validate the LLM-as-judge against human-labelled examples.
- Add a hybrid router sending structured/factual queries to SQL and open questions to RAG.
- Response streaming to reduce perceived latency; add retrieval **precision** alongside recall.
