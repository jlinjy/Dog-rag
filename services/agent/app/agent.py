"""create_agent wiring: model + tools + middleware."""
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from app import config
from app.prompts import SYSTEM_PROMPT
from app.schemas import ChatResponse
from app.tools.breed_facts import breed_database
from app.tools.rag_search import breed_articles

TOOLS = [breed_database, breed_articles]


def build_model() -> BaseChatModel:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_HOST, temperature=0)


def build_agent(model: BaseChatModel | None = None):
    return create_agent(
        model or build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # Hard stop so a confused model can't loop forever.
            ModelCallLimitMiddleware(run_limit=config.MAX_MODEL_CALLS, exit_behavior="end"),
        ],
    )


def run(agent, question: str) -> ChatResponse:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = result["messages"]
    tools_used = [tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    # Normally the run ends on the model's final AIMessage. A return_direct tool
    # (breed_articles) ends it on the ToolMessage instead, whose content is the answer.
    final = messages[-1]
    answer = final.content if isinstance(final, (AIMessage, ToolMessage)) else ""
    return ChatResponse(answer=answer, tools_used=tools_used)
