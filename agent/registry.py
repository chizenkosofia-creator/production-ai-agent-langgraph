"""The complete tool list handed to the LLM and to the ToolNode."""

from langchain_core.tools import BaseTool

from agent.ticket_tools import TICKET_TOOLS
from agent.tools import analyze_data, calculator, crm_lookup, search_knowledge_base

ALL_TOOLS: list[BaseTool] = [
    crm_lookup,
    calculator,
    analyze_data,
    search_knowledge_base,
    *TICKET_TOOLS,
]
