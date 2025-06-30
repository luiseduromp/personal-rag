from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str

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

class GenerateRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content: str
    metadata: Optional[dict]

class GenerateResponse(BaseModel):
    status: str
    message: Optional[str]
    answer: str
    sources: Optional[list[SourceDocument]]
