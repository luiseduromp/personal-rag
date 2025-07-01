import os
from pathlib import Path
from typing import Any, Dict

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .settings import LLM_MODEL, TEMPERATURE

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = os.path.join(BASE_DIR, "chroma_db")


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
        self.model_name = model_name
        self.temperature = temperature
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = vectorstore
        self.qa_chain = None

    def _create_qa_chain(self):
        """Create the QA chain with a custom prompt."""

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
        {question}  
        ---  
        Answer as me:
        """

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        self.qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    def get_answer(self, question: str, k: int = 4) -> Dict[str, Any]:
        """
        Get an answer to a question using the RAG pipeline.

        Args:
            question: The question to answer
            k: Number of relevant documents to retrieve

        Returns:
            Dict containing the answer and source documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")

        if self.qa_chain is None:
            self._create_qa_chain()

        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=k)

        # Get the answer
        result = self.qa_chain(
            {"input_documents": docs, "question": question}, return_only_outputs=True
        )

        # Format the response
        response = {
            "answer": result["output_text"].strip(),
            "sources": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ],
        }

        return response
