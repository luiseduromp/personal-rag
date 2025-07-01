import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from .models.schemas import GenerateRequest, GenerateResponse, Token
from .rag.loader import init_vectorstore
from .rag.rag_pipeline import RAGPipeline
from .rag.settings import LLM_MODEL, TEMPERATURE
from .utils.auth import authenticate, create_access_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline."""
    vectorstore = init_vectorstore()
    app.state.rag_pipeline = RAGPipeline(
        model_name=LLM_MODEL, temperature=TEMPERATURE, vectorstore=vectorstore
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    """Simple root endpoint."""
    return {"Info": "Personal RAG API for luiseduromp chatbot"}


@app.post("/token", response_model=Token)
async def get_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Retrieve an access token for authentication."""
    if not authenticate(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(form_data.username)
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: Request, body: GenerateRequest) -> Dict[str, Any]:
    """
    Generate an answer to a question using the RAG pipeline.

    Args:
        body: The request containing the question
    Returns:
        Dict containing the answer and source documents
    """
    rag_pipeline: RAGPipeline = request.app.state.rag_pipeline

    try:
        logger.info("Generating RAG answer")

        response = rag_pipeline.get_answer(question=body.question)

        return {
            "status": "success",
            "message": "Answer generated successfully",
            "answer": response["answer"],
            # TODO: Delete after testing
            "sources": [
                {"content": src["content"], "metadata": src["metadata"]}
                for src in response["sources"]
            ],
        }
    except Exception as e:
        logger.error("Failed to generate answer: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}",
        ) from e
