import os
from pathlib import Path

# --- LLM ---
LLM_MODEL = os.environ.get("AGENT_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --- agent loop ---
# Each model call is one "step" (decide → tool → decide → answer is 2 calls).
MAX_MODEL_CALLS = int(os.environ.get("AGENT_MAX_MODEL_CALLS", "6"))

# --- RAG service ---
RAG_URL = os.environ.get("RAG_URL", "http://localhost:80")
RAG_TIMEOUT_S = float(os.environ.get("RAG_TIMEOUT_S", "120"))

# --- structured data ---
# In Docker BREEDS_CSV is set; locally fall back to the copy in the rag service.
BREEDS_CSV = Path(
    os.environ.get("BREEDS_CSV")
    or Path(__file__).resolve().parents[3] / "services/rag/data/dog_breeds.csv"
)
SQL_MAX_ROWS = 50
