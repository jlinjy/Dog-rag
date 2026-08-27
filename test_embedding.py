import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import os

# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "dograg"
# DB_USER = "postgres"
# DB_PASSWORD = "postgres"

model = SentenceTransformer("all-MiniLM-L6-v2")

# postgres connection - need this because postgres in docker, not local
conn = psycopg2.connect(
        host= os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"]
    )
register_vector(conn)
cur = conn.cursor()

query = "compare pomeranian and maltese"
qv = model.encode(query)
cur.execute(
    "SELECT breed, content FROM chunks ORDER BY embedding <=> %s LIMIT 3;", (qv,)
)
for breed, content in cur.fetchall():
    print(f"\n=== {breed} === \n {content[:300]}...")