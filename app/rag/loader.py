import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .settings import DATA_DIRECTORY, DATABASE_DIRECTORY

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, DATA_DIRECTORY)
DATABASE_DIR = os.path.join(BASE_DIR, DATABASE_DIRECTORY)

logger = logging.getLogger(__name__)

def load_documents() -> List[Document]:
    """
    Load all .txt, .md, and .pdf files from the specified directory.
    
    Returns:
        List of loaded documents
    """

    documents = []
    
    # Check if directory exists
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Directory {DATA_DIR} does not exist")
    
    # Process .txt files
    for file_path in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, file_path)
        try:
            if file_path.endswith(('.txt', '.md')):
                loader = TextLoader(full_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file_path.endswith('.pdf'):
                loader = PyMuPDFLoader(full_path)
                documents.extend(loader.load())
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue

    return documents


def build_vectorstore(documents: List[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DATABASE_DIR)
    return vectorstore


def init_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings()
    
    if os.path.isdir(DATABASE_DIR) and os.listdir(DATABASE_DIR):
        logger.info("Vector store exists, Retrieving")
        return Chroma(persist_directory=DATABASE_DIR, embedding_function=embeddings)
    
    logger.info("Vector store not found, Building")
    documents = load_documents()
    vectorstore = build_vectorstore(documents)
    return vectorstore
    

    
    