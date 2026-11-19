"""Agent wiring tests with a scripted fake LLM (no Ollama needed).

The fake model decides the route; these tests check that the graph executes
that decision correctly: right tool runs, results flow back, return_direct
ends the turn, and the call limit stops a looping model.
"""
from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import config
from app.agent import build_agent, run
from app.tools import rag_search


class ToolFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(tool, **args):
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"call_{tool}"}])


def agent_with(*replies, loop=False):
    # Fresh copies: LangGraph replaces (not appends) messages that share an id.
    msgs = (m.model_copy() for m in (cycle(replies) if loop else replies))
    return build_agent(ToolFakeModel(messages=msgs))


def test_sql_route_then_final_answer():
    agent = agent_with(
        call("breed_database", sql="SELECT name, lifespan_min_yrs, lifespan_max_yrs FROM breeds WHERE name = 'Beagle'"),
        AIMessage(content="Beagles live 12-15 years."),
    )
    resp = run(agent, "How long does a beagle live?")
    assert resp.tools_used == ["breed_database"]
    assert resp.answer == "Beagles live 12-15 years."


def test_sql_result_reaches_the_model():
    agent = agent_with(
        call("breed_database", sql="SELECT name FROM breeds ORDER BY height_max_in DESC LIMIT 1"),
        AIMessage(content="done"),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "tallest?"}]})
    tool_msg = result["messages"][2]
    assert "Irish Wolfhound" in tool_msg.content


def test_sql_error_then_retry():
    agent = agent_with(
        call("breed_database", sql="SELECT nope FROM breeds"),
        call("breed_database", sql="SELECT COUNT(*) FROM breeds WHERE country = 'England'"),
        AIMessage(content="21 breeds."),
    )
    resp = run(agent, "How many breeds come from England?")
    assert resp.tools_used == ["breed_database", "breed_database"]
    assert resp.answer == "21 breeds."


def test_articles_route_returns_rag_answer_directly(monkeypatch):
    monkeypatch.setattr(rag_search, "ask_rag", lambda q: f"RAG says: {q}")
    # Only one scripted model reply: if return_direct didn't end the turn,
    # the agent would ask the model again and the fake would run out.
    agent = agent_with(call("breed_articles", question="Why police dogs?"))
    resp = run(agent, "Why are German Shepherds police dogs?")
    assert resp.tools_used == ["breed_articles"]
    assert resp.answer == "RAG says: Why police dogs?"


def test_no_tool_call_answer_passes_through():
    agent = agent_with(AIMessage(content="Hello!"))
    resp = run(agent, "hi")
    assert resp.tools_used == []
    assert resp.answer == "Hello!"


def test_looping_model_is_stopped_by_call_limit():
    agent = agent_with(call("breed_database", sql="SELECT nope FROM breeds"), loop=True)
    resp = run(agent, "loop forever")
    assert len(resp.tools_used) == config.MAX_MODEL_CALLS
    assert resp.answer.startswith("Model call limits exceeded")
