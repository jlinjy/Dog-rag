from ollama import chat, Client
import os
import json
import re

ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))
# DB_DSN = os.environ.get("DOG_DB", "host=localhost port=5432 dbname=dograg user=postgres password=postgres")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

def llm_judge(question, answer, key_facts):
    prompt = f"""
    You are an evaluator. Read the answer below and judge it yourself. 

    Question: {question}
    Answer: {answer}
    Expected facts that answer should convey: {key_facts}

    Does the answer correctly and groundedly address the expected facts? 
    Score based only on whether the answer correctly conveys the expected facts and stays grounded in them. 
    Do not penalise brevity. 
    
    Score 1.0 if all expected facts are present and accurate, 0.0 if none are.

    Respond with ONLY this JSON {{"score": <0,0-1.0>, "reason": "<brief>"}} """

    resp = ollama_client.chat(model=LLM_MODEL, messages=[{"role": "user", "content":prompt}])
    parsed = parse_judge(resp.message.content)
    return parsed

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