"""Per-question metrics for the routing agent."""

LIMIT_MSG = "Model call limits exceeded"


def first_route(tools_used: list[str]) -> str | None:
    """The agent's routing decision = the first tool it called."""
    return tools_used[0] if tools_used else None


def route_correct(expected_tool: str, tools_used: list[str]) -> bool:
    return first_route(tools_used) == expected_tool


def extra_calls(tools_used: list[str]) -> int:
    """Tool calls beyond the first (retries after SQL errors, or a second source)."""
    return max(len(tools_used) - 1, 0)


def hit_limit(answer: str) -> bool:
    return answer.startswith(LIMIT_MSG)
