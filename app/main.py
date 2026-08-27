from fastapi import FastAPI
import logging
from rag import answer


app = FastAPI()

logger = logging.getLogger("uvicorn.error")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def read_root():
    return {"status": "200"}

@app.post("/ask")
def ask(user_query: str):
    rag_reply = answer(user_query)
    logger.info(rag_reply)
    return {"reply": rag_reply['reply'], "sources": rag_reply['sources']}
