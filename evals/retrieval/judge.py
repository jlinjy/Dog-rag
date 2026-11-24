from ollama import Client
import os
import json
import re

ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))
# DB_DSN = os.environ.get("DOG_DB", "host=localhost port=5432 dbname=dograg user=postgres password=postgres")
LLM_MODEL = "qwen2.5:7b"

# Phrases that signal the model stepped outside the retrieved context.
OUTSIDE_KNOWLEDGE = re.compile(
    r"general knowledge|common knowledge|outside (of )?the (context|sources)|"
    r"not (mentioned|stated) in the (context|sources), (but|however)|"
    r"based on my (own )?knowledge",
    re.IGNORECASE,
)


def flags_outside_knowledge(answer):
    """Deterministic check: does the answer admit to using non-context knowledge?"""
    return bool(OUTSIDE_KNOWLEDGE.search(answer))


def llm_judge(question, answer, key_facts):
    """Coverage: which of the listed key facts does the answer convey?

    The judge only returns a yes/no verdict per key fact; the score is computed
    here as the fraction of "yes", so it cannot grade against criteria of its own.
    """
    if not key_facts:
        return None, "no key facts to check"

    checklist = "\n".join(f"{i+1}. {f}" for i, f in enumerate(key_facts))
    prompt = f"""
    You are checking an answer against a fixed checklist of key facts.
    Each key fact is a keyword or word stem (e.g. "energ" means energy/energetic).

    Question: {question}
    Answer: {answer}

    Key facts:
    {checklist}

    For EACH numbered key fact, decide only: does the answer mention or convey
    this fact anywhere? A passing mention or a list item counts, and so does a
    clear synonym or paraphrase.
    - Judge ONLY the listed key facts. Ignore anything else the answer includes or omits.
    - Do not penalise brevity or missing details that are not in the list.
    - First quote the few words (10 or fewer) from the answer that convey the
      fact ("" if none), then set "present": true only if that quote conveys it.
      Keep quotes short and do not include double quote characters in them.
    - If the answer says it lacks information about a fact, that fact is NOT conveyed.

    Respond with ONLY this JSON, one entry per key fact in list order:
    {{"facts": [{{"fact": "<key fact>", "evidence": "<short quote, max 10 words, or empty>", "present": true or false}}, ...]}} """

    resp = ollama_client.chat(model=LLM_MODEL,
                              messages=[{"role": "user", "content": prompt}],
                              options={"temperature": 0})
    return parse_checklist(resp.message.content, key_facts)


def _fact_key(s):
    """Normalise a fact label: drop list numbering ("1. "), quotes, case."""
    return re.sub(r"^\s*\d+[.)]\s*", "", str(s)).strip().strip("\"'").lower()


def parse_checklist(content, key_facts):
    """Match judge entries to key facts by their "fact" label.

    Extra or repeated entries are ignored (first entry per fact wins). If the
    labels don't line up but the entry count equals the number of key facts,
    fall back to list order.
    """
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if not match:
        print(f"[judge: no JSON found] {content[:80]}")
        return None, "no json"
    try:
        entries = json.loads(match.group())["facts"]
        entries = [e for e in entries
                   if isinstance(e, dict) and isinstance(e.get("present"), bool)]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[judge parse failed] {e}")
        return None, "parse failed"

    by_fact = {}
    for e in entries:
        by_fact.setdefault(_fact_key(e.get("fact", "")), e)

    matched = [by_fact.get(_fact_key(f)) for f in key_facts]
    if None in matched and len(entries) == len(key_facts):
        matched = entries   # labels garbled but order/count is usable
    if not entries:
        print("[judge: no verdicts returned]")
        return None, "no verdicts"
    # A fact the judge skipped is treated as not conveyed (in practice it
    # skips facts that are absent from the answer).
    matched = [m if m is not None else {"present": False, "evidence": "", "skipped": True}
               for m in matched]

    reason = "; ".join(f"{f}: {'yes' if m['present'] else 'no'}"
                       + (" (skipped by judge)" if m.get("skipped") else "")
                       + (f" ({m.get('evidence')!r})" if m.get("evidence") else "")
                       for f, m in zip(key_facts, matched))
    return sum(m["present"] for m in matched) / len(key_facts), reason


def grounding_judge(question, answer, chunks):
    """Faithfulness: is every claim in the answer supported by the retrieved context?"""
    context = "\n\n".join(f"[Source {i+1}: {b}] {c}" for i, (b, c) in enumerate(chunks))
    prompt = f"""
    You are checking whether an answer is grounded in the provided context.

    --- CONTEXT ---
    {context}

    --- QUESTION ---
    {question}

    --- ANSWER ---
    {answer}

    List each factual claim in the answer and check whether the CONTEXT supports it.
    Claims drawn from general knowledge count as UNSUPPORTED, even if they are true.
    Saying "the context does not contain this" is not a claim.

    Score = fraction of claims supported by the context (1.0 if all are, 0.0 if none are).

    Respond with ONLY this JSON {{"score": <0.0-1.0>, "reason": "<brief, name any unsupported claims>"}} """

    resp = ollama_client.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return parse_judge(resp.message.content)


def parse_judge(content):
    match = re.search(r'\{.*\}', content, re.DOTALL)   # find the first {...} block
    if not match:
        print(f"[judge: no JSON found] {content[:80]}")
        return None, "no json"
    try:
        parsed = json.loads(match.group())
        return parsed["score"], parsed.get("reason", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[judge parse failed] {e}")
        return None, "parse failed"
