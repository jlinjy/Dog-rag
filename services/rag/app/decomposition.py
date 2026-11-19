from app.retrieval import conn, retrieve, TOP_K


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
