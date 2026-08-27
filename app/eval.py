import json
import os
from rag import query_decomposition, build_prompt, LLM_MODEL
from ollama import Client

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
    questions = json.load(open(path))
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

        results.append({"id": q["id"], "type": q["type"],
                        "retrieval": r_score, "facts": f_score,
                        "expected": q["expected_breeds"], "retrieved": retrieved
                        })
        print(f"[{q["id"]}] {q["type"]} retrieval={r_score}, facts={f_score}")
        # print(f"answer_text: {answer_text}")
        print(f"Expected breeds: {q["expected_breeds"]}, retrieved: {retrieved}")
    return results

if __name__ == "__main__":
    #for k in (4, 10):
    #    print(f"\n===== top_k={k} =====")
    run_eval(path="eval_questions.json",k=4)