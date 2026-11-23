import os
from ollama import Client

from app.decomposition import query_decomposition

ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))
LLM_MODEL = "llama3.2" # gemma3 for multi-modal


def build_prompt(query: str, chunks: list[tuple[str, str]]):
  """Assemble a grounded prompt: numbered context + the question."""
  context = "\n\n".join(
      f"[Source {i+1}: {breed}] \n {content}" 
      for i, (breed, content) in enumerate(chunks)
  )
  return (
      "You are a helpful assistant answering questions about dog breeds. "
      "Answer ONLY using the context below. If the context does not contain "
      "the answer, say you don't have enough information — do not invent facts. "
      "Cite the source breed(s) you used.\n\n"
      f"--- CONTEXT ---\n{context}\n\n"
      f"--- QUESTION ---\n{query}\n\n"
      "Answer:" 
  )


def generate(query, chunks): 
  # New function separated out for timing
  prompt = build_prompt(query, chunks)
  response = ollama_client.chat(model=LLM_MODEL, messages=[
    {
      'role': 'user',
      'content': prompt,
    },
  ])
  return response['message']['content']


def answer(query: str):
  chunks = query_decomposition(query)
  prompt = build_prompt(query, chunks)

  response = ollama_client.chat(model=LLM_MODEL, messages=[
    {
      'role': 'user',
      'content': prompt,
    },
  ])
  reply = response['message']['content']
  sources = ", ".join(dict.fromkeys(breed for breed, _ in chunks)) # unique, ordered
  return {"reply": reply, "sources": sources}
  #return f"{reply} \n\n [retrieved from {sources}]"

if __name__ == "__main__":
  while True:
    q = input("\nQuestion (or 'quit'): ").strip()
    if q.lower() in ("quit", "exit", "q", ""):
        break
    print(answer(q))
