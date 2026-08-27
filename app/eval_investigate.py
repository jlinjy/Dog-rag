# from rag import retrieve
from rag import embedder, conn   # reuse your existing model + connection
from pgvector.psycopg2 import register_vector

# for breed, content in retrieve("pomeranian", k=4):
#     print(breed, "-", content[:80])

register_vector(conn)
cur = conn.cursor()

# Disable index
cur.execute("SET enable_indexscan = off;")
cur.execute("SET enable_bitmapscan = off;")

# then re-run the same SELECT
query = "What size is a pomeranian?"
qv = embedder.encode(query)          # THIS is "your query vector" — the embedding of the text

# show the 10 nearest chunks WITH their distances and breeds
cur.execute(
    "SELECT breed, embedding <=> %s AS distance, LEFT(content, 60) "
    "FROM chunks ORDER BY distance LIMIT 10;",
    (qv,)
)
for breed, dist, snippet in cur.fetchall():
    print(f"{dist:.4f}  {breed:<20} {snippet}")