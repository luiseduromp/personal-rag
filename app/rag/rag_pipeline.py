import logging
from datetime import datetime
from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .settings import EMBEDDINGS_MODEL, LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline wrapper.

    Wires together an OpenAI Chat model, OpenAI embeddings and a Chroma
    vectorstore to produce history-aware retrieval-augmented answers.

    Constructor:
        model_name: LLM model name (defaults to value from settings).
        temperature: LLM temperature.
        vectorstore: Initialized `Chroma` vector store (required).
        language: Pipeline language, e.g. "en" or "es".

    Main methods:
        build_history_prompt() -> PromptTemplate
        build_prompt() -> PromptTemplate
        generate_answer(question, chat_history) -> dict with keys:
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
        self.rag_chain = self._create_rag_chain()
        logger.info(f"Initialized RAG pipeline for language: {self.language}")

    HISTORY_PROMPTS = {
        "en": """
            Given the chat history and the latest user question, which may reference 
            information from the chat history, rewrite the question so that it can be 
            fully understood on its own without needing the chat history. Follow these
            rules:
            1. If the question already makes sense on its own, return it exactly as is.
            2. Preserve all important details (names, dates, numbers, and context) from 
            the chat history.
            3. Do not add new information, make assumptions, or answer the question.

            Chat History:
            {chat_history}

            Latest Question:
            {input}

            Standalone question:
            """,
        "es": """
            Teniendo en cuenta el historial de chat y la última pregunta del usuario, 
            que puede hacer referencia a información del historial de chat, reescribe 
            la pregunta para que se entienda completamente por sí sola sin necesidad 
            del historial. Sigue las siguientes reglas.
            1. Si la pregunta ya tiene sentido por sí sola, devuélvela tal como está.
            2. Conserva todos los detalles importantes (nombres, fechas, números y 
            contexto) del historial de chat.
            3. No añadas información nueva, hagas suposiciones ni respondas la pregunta.

            Historial de chat:
            {chat_history}

            Ultima pregunta:
            {input}

            Pregunta independiente:
            """,
    }

    def build_history_prompt(self) -> PromptTemplate:
        """
        Builds a history prompt template for the RAG pipeline.

        Returns:
            PromptTemplate: The history prompt template.
        """
        template = self.HISTORY_PROMPTS.get(self.language, self.HISTORY_PROMPTS["en"])
        return PromptTemplate.from_template(template)

    PROMPT_TEMPLATES = {
        "en": """You are acting as my personal assistant and will respond
            **as if you were me**, using first person ("I", "my", etc.).
            Add some personality to your answers using emojis and not so
            formal language.

            Today's date is {date}. Use it to interpret and respond with
            time references.

            You must strictly follow these rules:

            1. Only use the information provided in the context below. 
            2. If the information is **not available or insufficient**, respond with: 
            **"I'm sorry, I do not know."**  
            3. Do **not** make up or hallucinate information.  
            4. If the question is **too private or personal**, respond with: 
            **"Sorry, I can't answer that."**
            5. Do not state as **future** anything dated before today.

            ---  
            Context:  
            {context}  
            ---  
            Question:  
            {input}  
            ---  
            Answer as me:
            """,
        "es": """Estás actuando como mi asistente personal y responderás
            **como si fueras yo**, usando la primera persona ("yo", "mi", etc.).
            Añade personalidad a tus respuestas usando emojis y un lenguaje 
            menos formal.

            La fecha de hoy es {date}. Úsala para interpretar y responder con
            referencias temporales.

            Debes seguir estrictamente estas reglas:

            1. Solo utiliza la información proporcionada en el contexto a continuación.
            2. Si la información **no está disponible o es insuficiente**, responde con:
            **"Lo siento, no lo sé."**
            3. No inventes ni alucines información.
            4. Si la pregunta es **demasiado privada o personal**, responde con:
            **"Lo siento, no puedo responder eso."**
            5. No declares como **futuro** nada con fecha antes de hoy.

            ---
            Contexto:
            {context}
            ---
            Pregunta:
            {input}
            ---
            Responde como si fueras yo:
            """,
    }

    def build_prompt(self) -> PromptTemplate:
        """
        Builds the main prompt template for the RAG pipeline.

        Returns:
            PromptTemplate: The main prompt template.
        """
        template = self.PROMPT_TEMPLATES.get(self.language, self.PROMPT_TEMPLATES["en"])
        return PromptTemplate.from_template(template)

    def _create_rag_chain(self) -> Runnable:
        """
        Creates the context aware RAG chain for document retrieval and answer
        generation.
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )

        history_prompt = self.build_history_prompt()

        today = datetime.now().strftime("%Y-%m-%d")
        answer_prompt = self.build_prompt().partial(date=today)

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm, retriever=retriever, prompt=history_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm, prompt=answer_prompt
        )

        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def generate_answer(
        self,
        question: str,
        chat_history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Generate an answer to a question using the RAG pipeline.

        Args:
            question: The question to answer
            chat_history: The chat history

        Returns:
            Dict containing the answer and source documents
        """
        history_str = ""

        for message in chat_history:
            prefix = "User" if message["role"] == "user" else "Assistant"
            history_str += f"{prefix}: {message['content']}\n"

        result = self.rag_chain.invoke(
            {
                "input": question,
                "chat_history": history_str,
            }
        )

        return {
            "answer": result["answer"].strip(),
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in result.get("context", [])
            ],
        }
