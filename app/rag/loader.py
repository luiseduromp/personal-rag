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
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .settings import (
    API_URL,
    CDN_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DATABASE_DIR,
    DEFAULT_COLLECTION,
    EMBEDDINGS_MODEL,
)

logger = logging.getLogger(__name__)


class Loader:
    def __init__(
        self,
        language: str = "en",
        database_dir: str = DATABASE_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        data_dir: str = DATA_DIR,
        data_url: str = API_URL,
    ):
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=database_dir,
            embedding_function=self.embeddings,
        )
        self.database_dir = database_dir
        self.data_url = data_url
        self.data_dir = data_dir
        self.ids = []
        self.language = language
        logger.info("Initialized loader")

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        self.leaf_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE.get(self.language, 350),
            chunk_overlap=CHUNK_OVERLAP.get(self.language, 50),
            encoding_name="cl100k_base",
            separators=["\n\n", "\n", " ", ""],
        )
        self.generic_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE.get(self.language, 350),
            chunk_overlap=CHUNK_OVERLAP.get(self.language, 50),
            encoding_name="cl100k_base",
            separators=["\n\n", "\n", " ", ""],
        )

    def _list_bucket_files(self) -> list[str]:
        """
        List all files in the S3 Bucket
        """
        try:
            response = requests.get(
                f"{self.data_url}/rag-list-docs",
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
            page_content=content,
            metadata={
                "source": url,
                "file_type": file_ext,
                "filename": filename,
                "lang_hint": self.language,
            },
        )

    def _split_markdown(self, doc: Document) -> list[Document]:
        """
        Split a markdown Document by headings first, then recursively split
        large sections. Adds breadcrumbs and carries source metadata.
        """
        sections = self.header_splitter.split_text(doc.page_content)

        split_docs: list[Document] = []

        for sec in sections:
            path_parts = [
                sec.metadata.get(h) for h in ("h1", "h2", "h3") if sec.metadata.get(h)
            ]
            section_path = " > ".join(path_parts) if path_parts else None

            leaves = self.leaf_splitter.split_text(sec.page_content)

            for leaf in leaves:
                content = f"[{section_path}]\n\n{leaf}" if section_path else leaf
                split_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            **doc.metadata,
                            **sec.metadata,
                            "section_path": section_path,
                        },
                    )
                )

        if not split_docs:
            return [
                Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "breadcrumbs": None,
                        "section_path": None,
                    },
                )
                for chunk in self.leaf_splitter.split_text(doc.page_content)
            ]
        return split_docs

    def _split_generic(self, doc: Document) -> list[Document]:
        """
        Recursive split for txt/pdf or anything non-markdown.
        """
        return [
            Document(
                page_content=chunk,
                metadata={**doc.metadata, "breadcrumbs": None, "section_path": None},
            )
            for chunk in self.generic_splitter.split_text(doc.page_content)
        ]

    def _filter_by_lang(self, files: list[str]) -> list[str]:
        filtered_files = [
            filename
            for filename in files
            if filename.lower().split("/")[-1].startswith(self.language.lower())
        ]

        if not filtered_files:
            logger.warning(f"No files found matching the language: {self.language}")
            return None

        return filtered_files

    def _load_from_s3(self) -> Optional[list[Document]]:
        """
        Load all .txt, .md, and .pdf files from the cloud data directory.

        Returns:
            List of loaded documents, or None if directory doesn't exist or is empty
        """

        list_files = self._list_bucket_files()

        filtered_files = self._filter_by_lang(list_files)

        if not filtered_files:
            logger.warning(f"No files found matching the language: {self.language}")
            return None

        documents: list[Document] = []

        for filename in filtered_files:
            file = f"{CDN_URL}/{filename}"
            logger.info(f"Loading file from URL: {file}")
            doc = self.load_from_url(file)

            if doc:
                documents.append(doc)

        return documents or None

    def _load_from_disk(self) -> Optional[list[Document]]:
        """
        Load all .txt, .md, and .pdf files from the docs directory, filtered
        by language.

        Returns:
            List of loaded documents, or None if directory doesn't exist or
            is empty
        """
        docs_path = Path(self.data_dir)

        if not docs_path.exists() or not docs_path.is_dir():
            logger.warning("The directory does not exist or is not a directory")
            return None

        all_files = [
            str(file_path)
            for file_path in docs_path.rglob("*")
            if file_path.suffix.lower() in [".md", ".txt", ".pdf"]
        ]

        filtered_files = self._filter_by_lang(all_files)

        if not filtered_files:
            logger.warning(f"No files found matching the language: {self.language}")
            return None

        documents: list[Document] = []

        for file_path in filtered_files:
            file_path = Path(file_path)
            logger.info(f"Loading file from disk: {file_path}")

            try:
                if file_path.suffix.lower() in [".txt", ".md"]:
                    content = file_path.read_text(encoding="utf-8")
                else:
                    with fitz.open(file_path) as doc:
                        content = "\n\n".join([page.get_text() for page in doc])

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": file_path.suffix.lower(),
                            "filename": file_path.name,
                            "lang_hint": self.language,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {e}")

        return documents or None

    def load_documents(self) -> Optional[list[Document]]:
        """
        Load all .txt, .md, and .pdf files from the cloud data directory.

        Returns:
            List of loaded documents, or None if directory doesn't exist or is empty
        """

        documents: list[Document] = self._load_from_disk() or []

        s3_docs = self._load_from_s3()
        if s3_docs:
            documents.extend(s3_docs)

        return documents or None

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Dispatch to markdown-aware splitting or generic splitting per doc.
        """
        chunks: list[Document] = []
        for doc in documents:
            ext = (doc.metadata.get("file_type") or "").lower()
            if ext == ".md":
                chunks.extend(self._split_markdown(doc))
            else:
                chunks.extend(self._split_generic(doc))
        return chunks

    def build_vectorstore(self, documents: list[Document]):
        """
        Build or augment a vector store from the given documents.

        Args:
            documents: List of documents to build the vector store
        """
        chunks = self._split_documents(documents)

        store_chunks = self._check_duplicates(chunks)

        if store_chunks:
            ids = self.vectorstore.add_documents(store_chunks)
            self.ids.extend(ids)
            logger.info("Added new documents to vector store")
        else:
            logger.info("No new chunks to add to vector store")

    def add_from_url(self, url: str) -> Optional[str]:
        """
        Add a document from a URL to the vector store.

        Args:
            url: URL of the document to add

        Returns:
            Metadata of the added document
        """
        document = self.load_from_url(url)
        if document:
            self.build_vectorstore([document])
            return document.metadata["source"]
        else:
            logger.error(f"Failed to load document from URL: {url}")
            return None

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
