import hashlib
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Optional

import fitz
import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings

from .settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DATABASE_DIR,
    EMBEDDINGS_MODEL,
)

logger = logging.getLogger(__name__)


class Loader:
    def __init__(self, database_dir: str = DATABASE_DIR, data_dir: str = DATA_DIR):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = Chroma(
            collection_name="luiseduromp_rag",
            persist_directory=database_dir,
            embedding_function=self.embeddings,
        )
        self.database_dir = database_dir
        self.data_dir = data_dir
        self.ids = []

    def _check_duplicates(self, chunks: list[Document]) -> list[Document]:
        """
        Check for duplicate chunks in the vector store.

        Args:
            chunks: List of chunks to check for duplicates

        Returns:
            List of chunks that are not duplicates
        """
        store_chunks = []
        for chunk in chunks:
            content_hash = self._compute_hash(chunk.page_content)
            chunk.metadata["content_hash"] = content_hash

            results = self.vectorstore.similarity_search_by_vector(
                self.embeddings.embed_query(chunk.page_content), k=1
            )

            if results and any(
                r.metadata["content_hash"] == content_hash for r in results
            ):
                logger.info("Skipping duplicate chunk")
                continue

            store_chunks.append(chunk)

        return store_chunks

    def _compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_documents(self) -> Optional[list[Document]]:
        """
        Load all .txt, .md, and .pdf files from the data directory.

        Returns:
            List of loaded documents, or None if directory doesn't exist or is empty
        """
        documents = []

        if not os.path.isdir(self.data_dir) or not os.listdir(self.data_dir):
            logger.warning("Directory does not exist or is empty")
            return None

        files = os.listdir(self.data_dir)
        for file_path in files:
            full_path = os.path.join(self.data_dir, file_path)
            try:
                logger.info("Loading document: %s", file_path)
                if file_path.endswith((".txt", ".md")):
                    loader = TextLoader(full_path, encoding="utf-8")
                    documents.extend(loader.load())
                elif file_path.endswith(".pdf"):
                    loader = PyMuPDFLoader(full_path)
                    documents.extend(loader.load())

            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue

        return documents if documents else None

    def build_vectorstore(self, documents: list[Document]) -> Chroma:
        """
        Build a vector store from the given documents.

        Args:
            documents: List of documents to build the vector store from

        Returns:
            Chroma vector store
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(documents)

        store_chunks = self._check_duplicates(chunks)

        if store_chunks:
            ids = self.vectorstore.add_documents(store_chunks)
            self.ids.extend(ids)
            logger.info("Added new documents to vector store")

    def init_vectorstore(self) -> Chroma:
        """
        Initialize the vector store.
        """
        if os.path.isdir(self.database_dir) and self.ids:
            logger.info("Vector store found, using existing")
            return self.vectorstore

        logger.info("Vector store not found, Building")
        os.makedirs(self.database_dir, exist_ok=True)
        documents = self.load_documents()
        self.build_vectorstore(documents)
        return self.vectorstore

    def load_from_url(self, url: str) -> Document:
        """
        Load a single document (pdf, txt or markdown) from a URL
        and save it to data_dir.

        Args:
            url: URL of the document to load

        Returns:
            Document object
        """
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content_bytes = response.content

        filename = os.path.basename(url).split("?")[0]
        if not filename or filename == "/":
            filename = f"document_{int(time.time())}"

        file_ext = Path(filename).suffix.lower()
        if not file_ext:
            content_type = response.headers.get("Content-Type", "")
            file_ext = mimetypes.guess_extension(content_type) or ".txt"

        if file_ext not in [".txt", ".md", ".pdf"]:
            raise ValueError(f"Unsupported file type: {file_ext}")

        os.makedirs(self.data_dir, exist_ok=True)

        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content_bytes)

        logger.info("Saved a new document in the data directory")

        if file_ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            with fitz.open(file_path) as doc:
                content = "\n\n".join([page.get_text() for page in doc])

        return Document(
            page_content=content, metadata={"source": url, "file_type": file_ext}
        )

    def insert_one(self, document: Document) -> str:
        """
        Inserts a document into the vector store.

        Args:
            document: Document to insert

        Returns:
            ID of the inserted document
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents([document])

        store_chunks = self._check_duplicates(chunks)

        if store_chunks:
            ids = self.vectorstore.add_documents(store_chunks)
            self.ids.extend(ids)
            logger.info("Added new Document to vector store")

    def add_from_url(self, url: str) -> str:
        document = self.load_from_url(url)
        self.insert_one(document)
        return document.metadata["source"]
