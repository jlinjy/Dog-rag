from fastapi import FastAPI

from app.agent import build_agent, run
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title="dog-agent")
agent = build_agent()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # sync def → FastAPI runs it in a threadpool, so the blocking agent call is fine
    return run(agent, req.message)
