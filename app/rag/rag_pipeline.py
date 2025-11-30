import logging
from datetime import datetime
from typing import Any, TypedDict

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from .prompts import HISTORY_PROMPTS, PROMPT_TEMPLATES
from .settings import EMBEDDINGS_MODEL, LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: Original user question
        rewritten_question: Context-aware rewritten question (optional)
        documents: Retrieved documents from vectorstore
        answer: Generated answer (optional)
        messages: Full conversation history for persistence and context
    """

    question: str
    rewritten_question: str | None
    documents: list[Any]
    answer: str | None
    messages: list[BaseMessage]


class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline wrapper using LangGraph.

    Wires together an OpenAI Chat model, OpenAI embeddings and a Chroma
    vectorstore to produce history-aware retrieval-augmented answers.

    Constructor:
        model_name: LLM model name (defaults to value from settings).
        temperature: LLM temperature.
        vectorstore: Initialized `Chroma` vector store (required).
        language: Pipeline language, e.g. "en" or "es".

    Main methods:
        generate_answer(question, thread_id) -> dict with keys:
            - answer: generated answer string
            - sources: list of source documents with 'content' and 'metadata'

    Raises:
        ValueError: if `vectorstore` is not provided to the constructor.
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        vectorstore: Chroma = None,
        language: str = "en",
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for the LLM
            language: Language for prompts ("en" or "es")
        """
        if vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.language = language.lower()

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )

        self.graph = self._build_graph()

        logger.info("✅ Initialized RAG pipeline for language: %s", self.language)

    def build_history_prompt(self) -> PromptTemplate:
        """
        Builds a history prompt template for the RAG pipeline.

        Returns:
            PromptTemplate: The history prompt template.
        """
        template = HISTORY_PROMPTS.get(self.language, HISTORY_PROMPTS["en"])
        return PromptTemplate.from_template(template)

    def build_prompt(self) -> PromptTemplate:
        """
        Builds the main prompt template for the RAG pipeline.

        Returns:
            PromptTemplate: The main prompt template.
        """
        template = PROMPT_TEMPLATES.get(self.language, PROMPT_TEMPLATES["en"])
        return PromptTemplate.from_template(template)

    def _rewrite_question(self, state: GraphState) -> dict:
        """
        Node that rewrites the user question with conversation history context.
        """
        question = state["question"]
        messages = state["messages"]

        if not messages:
            return {"rewritten_question": question}
        history_prompt = self.build_history_prompt()
        rewrite_chain = history_prompt | self.llm

        response = rewrite_chain.invoke({"input": question, "chat_history": messages})

        rewritten_question = response.content
        logger.info("Question rewritten: %s", rewritten_question)

        return {"rewritten_question": rewritten_question}

    def _retrieve_documents(self, state: GraphState) -> dict:
        """
        Node that retrieves documents using the rewritten question.
        """
        rewritten_question = state["rewritten_question"]

        documents = self.retriever.invoke(rewritten_question)
        logger.info("Retrieved %d documents", len(documents))

        return {"documents": documents}

    def _final_answer(self, state: GraphState) -> dict:
        """
        Node that generates the final answer using retrieved documents.
        """
        question = state["rewritten_question"]
        documents = state["documents"]
        original_question = state["question"]
        previous_messages = state["messages"]

        context = "\n\n".join([doc.page_content for doc in documents])

        today = datetime.now().strftime("%Y-%m-%d")
        answer_prompt = self.build_prompt().partial(date=today)

        answer_chain = answer_prompt | self.llm
        response = answer_chain.invoke({"input": question, "context": context})

        answer = response.content
        logger.info("Answer generated successfully")

        updated_messages = previous_messages.copy()
        updated_messages.append(HumanMessage(content=original_question))
        updated_messages.append(AIMessage(content=answer))

        return {"answer": answer, "messages": updated_messages}

    def _build_graph(self):
        """
        Builds the LangGraph state graph with separate nodes.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("final_answer", self._final_answer)

        workflow.add_edge(START, "rewrite_question")
        workflow.add_edge("rewrite_question", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "final_answer")

        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def generate_answer(
        self,
        question: str,
        thread_id: str,
    ) -> dict[str, Any]:
        """
        Generate an answer to a question using the RAG pipeline.

        Args:
            question: The question to answer
            thread_id: The conversation ID

        Returns:
            Dict containing the answer, rewritten_question, and source documents
        """
        config = {"configurable": {"thread_id": thread_id}}

        snapshot = self.graph.get_state(config)
        previous_messages = []

        if snapshot and snapshot.values:
            previous_messages = snapshot.values.get("messages", [])

            logger.info(
                "Loaded %d messages from conversation history",
                len(previous_messages),
            )

        result = self.graph.invoke(
            {
                "question": question,
                "rewritten_question": None,
                "documents": [],
                "answer": None,
                "messages": previous_messages,
            },
            config=config,
        )

        answer = result["answer"]
        rewritten_question = result["rewritten_question"]
        documents = result["documents"]

        sources = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in documents
        ]

        return {
            "answer": answer,
            "rewritten_question": rewritten_question,
            "sources": sources,
        }
