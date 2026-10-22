"""Evidence-grounded final-answer agent."""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

ANSWER_PROMPT = """You are an evidence-grounded RAG answer agent.
Use only the retrieved passages contained in the conversation to answer knowledge-base questions.
Cite claims as [source: filename, page: N] when page metadata exists, otherwise [source: filename].
If no relevant retrieved evidence is present, state that the indexed knowledge base did not provide sufficient evidence.
Do not invent facts, passages, page numbers, or sources."""


class AnswerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def __call__(self, state: dict) -> dict:
        messages = state["messages"]
        # Keep prior messages so the answer agent can see tool results and the user question.
        response: AIMessage = self.llm.invoke(
            [SystemMessage(content=ANSWER_PROMPT), *messages]
        )
        return {"messages": [response]}
