import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from ollama import chat, Client
from ollama import ChatResponse

load_dotenv()    
    
ollama_client = Client(host=os.environ.get("OLLAMA-HOST", "http://localhost:11434"))
# DB_DSN = os.environ.get("DOG_DB", "host=localhost port=5432 dbname=dograg user=postgres password=postgres")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2" # gemma3 for multi-modal
TOP_K = 4

embedder = SentenceTransformer(EMBED_MODEL)
#conn = psycopg2.connect(DB_DSN)
conn = psycopg2.connect(
        host= os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )
register_vector(conn)

def retrieve(query: str, k: int = TOP_K):
  """ Return top k most similar chunks as query"""
  qv = embedder.encode(query)
  cur = conn.cursor()
  cur.execute("SELECT breed, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
              (qv, k),
              )
  return cur.fetchall()


def query_decomposition(query: str, k=TOP_K, k_per_breed = 3):
  """ Detect which breed names appear in the question, make one sub-query per breed """
  cur = conn.cursor()
  cur.execute("SELECT DISTINCT breed FROM chunks")
  breeds = [row[0] for row in cur.fetchall()]

  # Get breeds mentioned in query
  q = query.lower()
  breeds_in_q = [b.replace("_", " ") for b in breeds if b.replace("_", " ") in q]

  if len(breeds_in_q) < 2:
    # do normal retrieval
    return retrieve(query, k)

  all_chunks = []
  for b in breeds_in_q:
    chunk = retrieve(b, k_per_breed)
    all_chunks.extend(chunk)

  return all_chunks


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