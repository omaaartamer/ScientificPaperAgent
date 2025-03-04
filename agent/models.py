from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent during the paper research process."""
    requires_research: bool = False
    num_feedback_requests: int = 0
    is_good_answer: bool = False
    messages: Annotated[Sequence[BaseMessage], add_messages]

class SearchPapersInput(BaseModel):
    """Input object to search papers with the CORE API."""
    query: str = Field(description="The query to search for on the selected archive.")
    max_papers: int = Field(
        default=1,
        description="The maximum number of papers to return. It's default to 1, but you can increase it up to 10 in case you need to perform a more comprehensive search.",
        ge=1,
        le=10
    )
    model_config = ConfigDict(
        json_schema_extra={
            "properties": {
                "query": {"type": "string"},
                "max_papers": {"type": "integer", "minimum": 1, "maximum": 10}
            }
        }
    )

class DecisionMakingOutput(BaseModel):
    requires_research: bool = Field(default=False)
    answer: Optional[str] = None
    
class JudgeOutput(BaseModel):
    is_good_answer: bool = Field(default=False)
    feedback: Optional[str] = None
