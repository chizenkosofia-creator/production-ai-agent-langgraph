"""Agent state definition.

Architecture decision: the graph state is deliberately small.

* ``messages`` uses LangGraph's ``add_messages`` reducer, so every node simply
  *returns* new messages and the reducer appends them (and de-duplicates by
  message id). The full list is persisted by the Postgres checkpointer keyed
  by ``thread_id``, which is what gives the agent durable multi-turn memory.
* ``iterations`` counts LLM calls within the *current* user turn. It has no
  reducer (last write wins) and the API resets it to 0 on every new request,
  which lets ``should_continue`` enforce a per-turn budget even though the
  rest of the state persists across turns.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State carried through the LangGraph workflow"""

    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
