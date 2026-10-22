"""Agent node that decides whether to call the knowledge search tool."""
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel


class RetrievalAgent:
    def __init__(self, llm: BaseChatModel, tools: list):
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: dict) -> dict:
        response: AIMessage = self.llm.invoke(state["messages"])
        return {"messages": [response]}
