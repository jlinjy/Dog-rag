"""HTTP client for the RAG service (services/rag, POST /ask)."""
import httpx
from langchain_core.tools import tool

from app import config


def ask_rag(question: str) -> str:
    try:
        resp = httpx.post(
            f"{config.RAG_URL}/ask",
            params={"user_query": question},
            timeout=config.RAG_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        return f"Error: breed article search is unavailable ({type(e).__name__}: {e})."
    sources = ", ".join(data.get("sources", [])) or "none"
    return f"{data.get('reply', '')}\n\n[article sources: {sources}]"


# return_direct: the RAG service already returns a grounded answer, so hand it
# straight to the user instead of letting the agent LLM re-summarise (and distort) it.
@tool("breed_articles", return_direct=True)
def breed_articles(question: str) -> str:
    """Search encyclopedia articles about dog breeds and get a grounded summary.

    Use for descriptive or explanatory questions that need prose rather than a
    single number: history and origins, what a breed was bred for, working
    roles, temperament and behaviour explained, suitability for families,
    care, grooming, training, and "why" questions.

    Input: the question in natural language, naming the breed(s) involved.
    """
    return ask_rag(question)
