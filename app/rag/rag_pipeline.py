from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .settings import EMBEDDINGS_MODEL, LLM_MODEL, TEMPERATURE


class RAGPipeline:
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

        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.language = language.lower()

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.rag_chain = self._create_rag_chain()

    HISTORY_PROMPTS = {
        "en": """
            Given the chat history and the latest user question, which might 
            reference context in the chat history, formulate a standalone question
            that can be understood without the chat history. Do **not** answer the
            question, just reformulate it if needed, otherwise return it as is.

            Chat History:
            {chat_history}

            Follow-up Question:
            {input}

            Standalone question:
            """,
        "es": """
            Dado el historial de chat y la última pregunta del usuario, que podría
            hacer referencia al contexto en el historial, formula una pregunta 
            independiente que pueda entenderse sin el historial. **No** respondas 
            la pregunta, solo reformúlala si es necesario, de lo contrario devuélvela 
            como está.

            Historial de chat:
            {chat_history}

            Pregunta de seguimiento:
            {input}

            Pregunta independiente:
            """,
    }

    def build_history_prompt(self):
        template = self.HISTORY_PROMPTS.get(self.language, self.HISTORY_PROMPTS["en"])
        return PromptTemplate.from_template(template)

    PROMPT_TEMPLATES = {
        "en": """You are acting as my personal assistant and will respond
            **as if you were me**, using first person ("I", "my", etc.).

            You must strictly follow these rules:

            1. Only use the information provided in the context below. 
            2. If the information is **not available or insufficient**, respond with: 
            **"I'm sorry, I do not know."**  
            3. Do **not** make up or hallucinate information.  
            4. If the question is **too private or personal**, respond with: 
            **"Sorry, I can't answer that."**

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

            Debes seguir estrictamente estas reglas:

            1. Solo utiliza la información proporcionada en el contexto a continuación.
            2. Si la información **no está disponible o es insuficiente**, responde con:
            **"Lo siento, no lo sé."**
            3. No inventes ni alucines información.
            4. Si la pregunta es **demasiado privada o personal**, responde con:
            **"Lo siento, no puedo responder eso."**

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

    def build_prompt(self):
        template = self.PROMPT_TEMPLATES.get(self.language, self.PROMPT_TEMPLATES["en"])
        return PromptTemplate.from_template(template)

    def _create_rag_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        history_prompt = self.build_history_prompt()
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm, retriever=retriever, prompt=history_prompt
        )

        answer_prompt = self.build_prompt()
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

        print(history_str)

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
