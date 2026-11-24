"""Retrieval + generation eval for the RAG service.

Run from the repo root:
    PYTHONPATH=services/rag python -m evals.retrieval.run [k]
"""
import json
import os
from datetime import datetime
from pathlib import Path
from app.decomposition import query_decomposition
from app.generation import generate, LLM_MODEL
from ollama import Client
from evals.retrieval.judge import llm_judge

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "retrieval.jsonl"

# ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))

def eval_retrieval(expected_breeds, retrieved_breeds):
    """
    Recall: of the breeds we expected to be retrieved, how many showed up?
    Returns None when the question doesn't specify expected breeds
    (open-ended aggregation), so it can be excluded from averages.
    """
    if not expected_breeds:
        return None
    hits = sum(1 for b in expected_breeds if b in retrieved_breeds)
    return hits / len(expected_breeds)

def eval_facts(key_facts, answer_text):
    """Fraction of expected key facts that appear in the answer."""
    if not key_facts:
        return None
    text = answer_text.lower()
    hits = sum(1 for f in key_facts if f.lower() in text)
    return hits / len(key_facts)

def run_eval(path, k=4):
    with open(path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    results = []

    for q in questions:
        # --- retrieval ---
        chunks = query_decomposition(q["question"], k)
        retrieved = list(dict.fromkeys(b for b, _ in chunks))
        r_score = eval_retrieval(q["expected_breeds"], retrieved)

        # --- generation ---
        answer_text = generate(q["question"], chunks)   # same call/options as the service
        f_score = eval_facts(q["key_facts"], answer_text)

        llm_score, llm_reason = llm_judge(q["question"], answer_text, q["key_facts"])
        #print(llm_score)

        results.append({"id": q["id"], "type": q["type"],
                        "question": q["question"],
                        "answer": answer_text,
                        "key_facts": q["key_facts"],
                        "retrieval": r_score, "facts": f_score,
                        "expected": q["expected_breeds"], "retrieved": retrieved,
                        "llm_eval_score": llm_score,  
                        "llm_eval_reason": llm_reason, 
                        })
        #print(f"[{q["id"]}] {q["type"]} retrieval={r_score}, facts={f_score}")
        # print(f"answer_text: {answer_text}")
        #print(f"Expected breeds: {q["expected_breeds"]}, retrieved: {retrieved}")
    return results

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

def _mean(results, key):
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else None

def write_report(results, k):
    """Save summary + per-question results (including the LLM answer) to a .txt file."""
    summary = {
        "n": len(results),
        "top_k": k,
        "model": LLM_MODEL,
        "retrieval_recall": _mean(results, "retrieval"),
        "fact_substring": _mean(results, "facts"),
        "llm_judge": _mean(results, "llm_eval_score"),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"rag_only_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(json.dumps({"summary": summary, "results": results},
                               indent=2, ensure_ascii=False))
    return path

if __name__ == "__main__":
    #for k in (4, 10):
    #    print(f"\n===== top_k={k} =====")
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    results = run_eval(path=DATASET, k=k)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Report written to {write_report(results, k)}")
