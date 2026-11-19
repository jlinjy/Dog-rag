from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]  # route taken, in call order, e.g. ["breed_database"]
