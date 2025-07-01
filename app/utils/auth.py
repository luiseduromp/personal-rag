import logging
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from dotenv import load_dotenv
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

load_dotenv()

logger = logging.getLogger(__name__)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def authenticate(username: str, password: str) -> bool:
    """Authenticate a user."""
    expected_username = os.getenv("USERNAME")
    hashed_password = os.getenv("HASHED")
    if username != expected_username:
        return False
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(username: str) -> str:
    """Create a JWT access token."""
    secret_key = os.getenv("SECRET_KEY")
    algorithm = os.getenv("ALGORITHM")
    expire = datetime.now(timezone.utc) + timedelta(minutes=30)
    data = {"sub": username, "exp": expire}
    encoded_jwt = jwt.encode(claims=data, key=secret_key, algorithm=algorithm)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode a JWT token."""
    secret_key = os.getenv("SECRET_KEY")
    algorithm = os.getenv("ALGORITHM")
    try:
        logger.info("Decoding token")
        payload = jwt.decode(token=token, key=secret_key, algorithms=algorithm)
        return payload
    except JWTError as e:
        logger.error("Failed to decode token: %s", str(e))
        return None
