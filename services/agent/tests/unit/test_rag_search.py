import httpx
import pytest

from app import config
from app.tools import rag_search


@pytest.fixture
def mock_http(monkeypatch):
    """Route httpx.post through a MockTransport; returns the list of captured requests."""
    calls = []

    def install(handler):
        def fake_post(url, **kw):
            with httpx.Client(transport=httpx.MockTransport(handler)) as c:
                calls.append(kw)
                return c.post(url, **kw)
        monkeypatch.setattr(rag_search.httpx, "post", fake_post)
        return calls
    return install


def test_formats_reply_and_sources(mock_http, monkeypatch):
    monkeypatch.setattr(config, "RAG_URL", "http://rag")
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"reply": "Bred for herding.", "sources": ["collie", "kelpie"]})

    mock_http(handler)
    out = rag_search.breed_articles.invoke({"question": "what were collies for?"})
    assert seen["url"].startswith("http://rag/ask?user_query=what")
    assert out == "Bred for herding.\n\n[article sources: collie, kelpie]"


def test_http_error_is_returned_as_text(mock_http):
    mock_http(lambda req: httpx.Response(503))
    out = rag_search.ask_rag("anything")
    assert out.startswith("Error: breed article search is unavailable")


def test_connection_error_is_returned_as_text(mock_http):
    def handler(req):
        raise httpx.ConnectError("refused")
    mock_http(handler)
    assert "ConnectError" in rag_search.ask_rag("anything")


def test_breed_articles_returns_direct():
    assert rag_search.breed_articles.return_direct is True
