from fastapi.testclient import TestClient

from app import api
from app.schemas import ChatResponse


def test_chat_endpoint(monkeypatch):
    monkeypatch.setattr(api, "run", lambda agent, q: ChatResponse(answer=f"echo {q}", tools_used=["breed_database"]))
    client = TestClient(api.app)
    r = client.post("/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert r.json() == {"answer": "echo hi", "tools_used": ["breed_database"]}
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.get("/health").json() == {"status": "ok"}
