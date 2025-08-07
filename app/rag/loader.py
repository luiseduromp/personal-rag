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
from langchain_openai import OpenAIEmbeddings

from .settings import (
    API_URL,
    CDN_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATABASE_DIR,
    DEFAULT_COLLECTION,
    EMBEDDINGS_MODEL,
)

logger = logging.getLogger(__name__)


class Loader:
    def __init__(
        self,
        database_dir: str = DATABASE_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        data_dir: str = API_URL,
    ):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=database_dir,
            embedding_function=self.embeddings,
        )
        self.database_dir = database_dir
        self.data_dir = data_dir
        self.ids = []
        logger.info("Initialized loader")

    def _list_bucket_files(self) -> list[str]:
        """
        List all files in the S3 Bucket
        """
        try:
            response = requests.get(
                f"{self.data_dir}/rag-list-docs",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("files", [])
        except Exception as e:
            logger.error("Error listing bucket files: %s", str(e))
            return []

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

            results = self.vectorstore.get(where={"content_hash": content_hash})

            if results and results.get("documents"):
                logger.info("Skipping duplicate chunk")
                continue

            store_chunks.append(chunk)

        return store_chunks

    def _compute_hash(self, text: str) -> str:
        """
        Compute a SHA-256 hash of the given text.

        Args:
            text: Text to hash

        Returns:
            SHA-256 hash of the text
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def load_from_url(self, url: str) -> Optional[Document]:
        """
        Load a single document (pdf, txt or markdown) from a URL.

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
            logger.error(f"Unsupported file type: {file_ext}")
            return None

        if file_ext in [".txt", ".md"]:
            content = content_bytes.decode("utf-8")
        else:
            with fitz.open(stream=content_bytes, filetype="pdf") as doc:
                content = "\n\n".join([page.get_text() for page in doc])

        return Document(
            page_content=content, metadata={"source": url, "file_type": file_ext}
        )

    def load_documents(self) -> Optional[list[Document]]:
        """
        Load all .txt, .md, and .pdf files from the cloud data directory.

        Returns:
            List of loaded documents, or None if directory doesn't exist or is empty
        """
        documents = []

        list_files = self._list_bucket_files()

        if not list_files:
            logger.warning("No files found in the bucket")
            return None

        for filename in list_files:
            file = f"{CDN_URL}/{filename}"
            logger.info(f"Loading file from: {file}")
            documents.append(self.load_from_url(file))

        return documents if documents else None

    def build_vectorstore(self, documents: list[Document]):
        """
        Build or augment a vector store from the given documents.

        Args:
            documents: List of documents to build the vector store
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

    def add_from_url(self, url: str) -> str:
        document = self.load_from_url(url)
        self.build_vectorstore([document])
        return document.metadata["source"]

    def init_vectorstore(self):
        """
        Initialize the vectorstore at the start of the application.
        """
        documents = self.load_documents()
        if documents:
            logger.info("Building vector store with loaded documents")
            self.build_vectorstore(documents)
        else:
            logger.warning("No documents found to build the vector store")

        return self.vectorstore
