from app.retrieval import conn, retrieve, TOP_K


def query_decomposition(query: str, k=TOP_K, k_per_breed = 3):
  """ Detect which breed names appear in the question, make one sub-query per breed.

  Each sub-query is "<breed> <rest of the question>" (other breed names removed),
  searched only within that breed's chunks, so it finds the breed's chunks that
  are relevant to the question rather than just its intro.
  k_per_breed is a minimum; with a larger k each breed gets k // n_breeds chunks.
  """
  cur = conn.cursor()
  cur.execute("SELECT DISTINCT breed FROM chunks")
  breeds = [row[0] for row in cur.fetchall()]

  # Get breeds mentioned in query
  q = query.lower()
  breeds_in_q = [b for b in breeds if b.replace("_", " ") in q]

  if len(breeds_in_q) < 2:
    # do normal retrieval
    return retrieve(query, k)

  # The question without any breed names, e.g. "which is better suited to a hot climate, a or a?"
  rest = q
  for b in sorted(breeds_in_q, key=len, reverse=True):
    rest = rest.replace(b.replace("_", " "), " ")
  rest = " ".join(rest.split())

  per_breed = max(k_per_breed, k // len(breeds_in_q))
  all_chunks = []
  for b in breeds_in_q:
    sub_query = f"{b.replace('_', ' ')} {rest}"
    all_chunks.extend(retrieve(sub_query, per_breed, breed=b))

  return all_chunks
