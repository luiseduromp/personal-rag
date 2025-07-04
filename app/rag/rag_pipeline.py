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
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: Name of the OpenAI model to use
            temperature: Temperature parameter for the LLM
        """

        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        self.rag_chain = self._create_rag_chain()

    @staticmethod
    def build_history_prompt():
        prompt_template = """
        Given the chat history and the latest user question, which might 
        reference context in the chat history, formulate a standalone question
        that can be understood without the chat history. Do **not** answer the
        question, just reformulate it if needed, otherwise return it as is.

        Chat History:
        {chat_history}

        Follow-up Question:
        {input}

        Standalone question:
        """
        return PromptTemplate.from_template(prompt_template)

    @staticmethod
    def build_prompt():
        prompt_template = """You are acting as my personal assistant and will respond
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
        """

        return PromptTemplate.from_template(prompt_template)

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
