from unittest.mock import MagicMock, patch

import pytest

from app.rag.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_vectorstore():
    vectorstore = MagicMock()
    retriever = MagicMock()
    vectorstore.as_retriever.return_value = retriever
    return vectorstore


@pytest.fixture
def mock_llm():
    with patch("app.rag.rag_pipeline.ChatOpenAI") as MockLLM:
        llm_instance = MockLLM.return_value
        yield llm_instance


@patch("app.rag.rag_pipeline.OpenAIEmbeddings")
def test_rag_pipeline_initialization(mock_embeddings, mock_vectorstore, mock_llm):
    pipeline = RAGPipeline(vectorstore=mock_vectorstore)
    assert pipeline.graph is not None
