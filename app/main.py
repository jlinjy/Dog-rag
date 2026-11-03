from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
import logging
import time
from rag import answer, retrieve, generate

app = FastAPI()

logger = logging.getLogger("uvicorn.error")
RETRIEVAL_TIME = Histogram("rag_retrieval_seconds", "Time spent on retrieval")
GENERATION_TIME = Histogram("rag_generation_seconds", "Time spent on generation")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def read_root():
    return {"status": "200"}

@app.post("/ask")
def ask(user_query: str):
    t0 = time.time()
    chunks = retrieve(user_query)
    RETRIEVAL_TIME.observe(time.time()-t0)

    t1 = time.time()
    rag_reply = generate(user_query, chunks)
    GENERATION_TIME.observe(time.time() - t1)

    sources = list(dict.fromkeys(b for b, _ in chunks))
    #rag_reply = answer(user_query)
    #logger.info(rag_reply)
    #return {"reply": rag_reply['reply'], "sources": rag_reply['sources']}
    return {"reply": rag_reply, "sources": sources}


Instrumentator().instrument(app).expose(app)