"""Routing eval for the agent service.

Run from the repo root (RAG service must be reachable at RAG_URL):
    PYTHONPATH=services/agent python -m evals.agent.run
"""
import json
import time
from datetime import datetime
from pathlib import Path

from app.agent import build_agent, run
from evals.agent.metrics import extra_calls, first_route, hit_limit, route_correct

EVALS_DIR = Path(__file__).resolve().parents[1]
DATASET = EVALS_DIR / "datasets" / "agent.jsonl"
REPORTS = EVALS_DIR / "reports"


def run_eval(path=DATASET):
    agent = build_agent()
    with open(path) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    results = []
    for q in questions:
        t0 = time.time()
        resp = run(agent, q["question"])
        r = {
            "id": q["id"],
            "question": q["question"],
            "expected_tool": q["expected_tool"],
            "route": first_route(resp.tools_used),
            "route_correct": route_correct(q["expected_tool"], resp.tools_used),
            "tools_used": resp.tools_used,
            "extra_calls": extra_calls(resp.tools_used),
            "hit_limit": hit_limit(resp.answer),
            "latency_s": round(time.time() - t0, 2),
            "answer": resp.answer,
        }
        results.append(r)
        mark = "ok " if r["route_correct"] else "BAD"
        print(f"[{mark}] {q['id']:<4} {str(r['route']):<15} calls={len(resp.tools_used)} "
              f"{'LIMIT ' if r['hit_limit'] else ''}{q['question']}")

    n = len(results)
    summary = {
        "n": n,
        "routing_accuracy": sum(r["route_correct"] for r in results) / n,
        "hit_limit_rate": sum(r["hit_limit"] for r in results) / n,
        "avg_extra_calls": sum(r["extra_calls"] for r in results) / n,
        "avg_latency_s": sum(r["latency_s"] for r in results) / n,
    }
    print(json.dumps(summary, indent=2))

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"agent_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"report: {out}")
    return summary, results


if __name__ == "__main__":
    run_eval()
