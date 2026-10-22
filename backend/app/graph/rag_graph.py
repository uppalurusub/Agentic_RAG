"""Build the LangGraph from independent agent nodes and tools."""
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.answer_agent import AnswerAgent
from app.agents.retrieval_agent import RetrievalAgent


def build_rag_graph(llm, tools: list, max_agent_steps: int):
    """Build and compile the RAG graph.

    Retrieval agent -> (tool call? tools -> retrieval agent) -> answer agent -> END.
    """
    retrieval_agent = RetrievalAgent(llm, tools)
    answer_agent = AnswerAgent(llm)

    builder = StateGraph(MessagesState)
    builder.add_node("retrieval_agent", retrieval_agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("answer_agent", answer_agent)

    builder.set_entry_point("retrieval_agent")
    builder.add_conditional_edges(
        "retrieval_agent",
        tools_condition,
        {"tools": "tools", END: "answer_agent"},
    )
    builder.add_edge("tools", "retrieval_agent")
    builder.add_edge("answer_agent", END)

    # max_agent_steps is applied by the invoke caller as a recursion limit.
    return builder.compile()
