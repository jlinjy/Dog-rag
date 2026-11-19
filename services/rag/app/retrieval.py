import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

# DB_DSN = os.environ.get("DOG_DB", "host=localhost port=5432 dbname=dograg user=postgres password=postgres")
EMBED_MODEL = "all-MiniLM-L6-v2"
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
