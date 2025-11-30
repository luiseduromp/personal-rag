from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str
    thread_id: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    hashed_password: str


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class UploadRequest(BaseModel):
    url: str


class GenerateRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []
    thread_id: Optional[str] = None


class SourceDocument(BaseModel):
    content: str
    metadata: Optional[dict]


class GenerateResponse(BaseModel):
    status: str
    message: Optional[str]
    answer: str
    sources: Optional[list[SourceDocument]]
    thread_id: Optional[str] = None


class GenerateDebugResponse(GenerateResponse):
    """Extended response model for debugging that includes the rewritten question."""

    rewritten_question: str
