import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from langdetect import detect

from .models.schemas import GenerateRequest, GenerateResponse, Token, UploadRequest
from .rag.loader import Loader
from .rag.rag_pipeline import RAGPipeline
from .utils.auth import authenticate, create_access_token, verify_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline."""
    en_loader = Loader(language="en", collection_name="luiseduromp_rag")
    en_vectorstore = en_loader.init_vectorstore()
    rag_pipeline = RAGPipeline(vectorstore=en_vectorstore, language="en")

    es_loader = Loader(language="es", collection_name="luiseduromp_esp")
    es_vectorstore = es_loader.init_vectorstore()
    esp_pipeline = RAGPipeline(vectorstore=es_vectorstore, language="es")

    app.state.rag_pipeline = rag_pipeline
    app.state.esp_pipeline = esp_pipeline

    logger.info("RAG pipelines initialized")
    yield


app = FastAPI(lifespan=lifespan)

if os.getenv("ENV", "development") == "development":
    load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """
    Simple root endpoint.
    """
    return {"info": "Personal RAG API for luiseduromp chatbot"}


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
async def generate_answer(
    request: Request, body: GenerateRequest, _: Annotated[str, Depends(verify_token)]
) -> Dict[str, Any]:
    """
    Generate an answer to a question using the RAG pipeline.

    Args:
        body: The request containing the question
    Returns:
        Dict containing the answer and source documents
    """
    rag_pipeline = request.app.state.rag_pipeline
    esp_pipeline = request.app.state.esp_pipeline

    try:
        logger.info("Generating RAG answer")

        language = detect(body.question)

        if language == "es":
            pipeline = esp_pipeline
            logger.info("Using Spanish RAG pipeline")
        else:
            pipeline = rag_pipeline
            logger.info("Using English RAG pipeline")

        response = pipeline.generate_answer(
            question=body.question,
            chat_history=body.chat_history,
        )

        return {
            "status": "success",
            "message": "Answer generated successfully",
            "answer": response["answer"],
            "sources": response["sources"],
        }
    except Exception as e:
        logger.error("Failed to generate answer: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}",
        ) from e


@app.post("/upload")
async def upload_file(
    body: UploadRequest, _: Annotated[str, Depends(verify_token)]
) -> Dict[str, Any]:
    loader = Loader()
    loader.add_from_url(body.url)
    return {"status": "success", "message": "Document uploaded successfully"}
