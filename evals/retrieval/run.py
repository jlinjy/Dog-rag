"""Retrieval + generation eval for the RAG service.

Run from the repo root:
    PYTHONPATH=services/rag python -m evals.retrieval.run
"""
import json
import os
from pathlib import Path
from app.decomposition import query_decomposition
from app.generation import build_prompt, LLM_MODEL
from ollama import Client
from evals.retrieval.judge import llm_judge

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "retrieval.jsonl"

ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))

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
        prompt = build_prompt(q["question"], chunks)
        resp = ollama_client.chat(model=LLM_MODEL,
                                messages=[{
                                'role': 'user',
                                'content': prompt,
                                }])
        answer_text = resp["message"]["content"]
        f_score = eval_facts(q["key_facts"], answer_text)

        llm_score, llm_reason = llm_judge(q, answer_text, q["key_facts"])
        #print(llm_score)

        results.append({"id": q["id"], "type": q["type"],
                        "retrieval": r_score, "facts": f_score,
                        "expected": q["expected_breeds"], "retrieved": retrieved,
                        "llm_eval_score": llm_score,  
                        "llm_eval_reason": llm_reason, 
                        })
        #print(f"[{q["id"]}] {q["type"]} retrieval={r_score}, facts={f_score}")
        # print(f"answer_text: {answer_text}")
        #print(f"Expected breeds: {q["expected_breeds"]}, retrieved: {retrieved}")
    return results

if __name__ == "__main__":
    #for k in (4, 10):
    #    print(f"\n===== top_k={k} =====")
    print(run_eval(path=DATASET, k=4))