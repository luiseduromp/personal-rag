import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import bcrypt
import pytest
from jose import JWTError, jwt

from app.utils.auth import (
    authenticate,
    create_access_token,
    decode_token,
)

# Test data
TEST_USERNAME = "testuser"
TEST_PASSWORD = "testpassword123"
# Generate a consistent salt for testing
TEST_SALT = bcrypt.gensalt()
TEST_HASHED_PASSWORD = bcrypt.hashpw(TEST_PASSWORD.encode("utf-8"), TEST_SALT).decode(
    "utf-8"
)


@pytest.fixture(autouse=True)
def auth_env_vars(monkeypatch):
    """Environment variables for testing authentication."""
    monkeypatch.setenv("SECRET_KEY", "test_secret_key")
    monkeypatch.setenv("ALGORITHM", "HS256")
    monkeypatch.setenv("HASHED", TEST_HASHED_PASSWORD)
    monkeypatch.setenv("USERNAME", TEST_USERNAME)


class TestEnvVars:
    """Tests for environment variables."""

    def test_env_vars(self):
        """Test environment variables."""
        assert os.getenv("SECRET_KEY") == "test_secret_key"
        assert os.getenv("ALGORITHM") == "HS256"
        assert os.getenv("HASHED") == TEST_HASHED_PASSWORD
        assert os.getenv("USERNAME") == TEST_USERNAME


class TestAuthenticate:
    """Tests for the authenticate function."""

    def test_authenticate_success(self):
        """Test successful authentication with correct credentials."""
        assert authenticate(TEST_USERNAME, TEST_PASSWORD) is True

    def test_authenticate_wrong_username(self):
        """Test authentication with incorrect username."""
        assert authenticate("wronguser", TEST_PASSWORD) is False

    def test_authenticate_wrong_password(self):
        """Test authentication with incorrect password."""
        assert authenticate(TEST_USERNAME, "wrongpassword") is False


class TestCreateAccessToken:
    """Tests for the create_access_token function."""

    def test_create_access_token_success(self):
        """Test successful token creation."""
        token = create_access_token(TEST_USERNAME)
        assert isinstance(token, str)
        assert len(token) > 0

    @patch("app.utils.auth.jwt.encode")
    def test_create_access_token_calls_encode(self, mock_encode):
        """Test that jwt.encode is called with correct parameters."""
        test_username = "testuser"
        create_access_token(test_username)

        # Check that encode was called
        assert mock_encode.called

        # Get the arguments passed to encode
        args, kwargs = mock_encode.call_args

        # Check the payload
        payload = args[0]
        assert payload["sub"] == test_username
        assert "exp" in payload

        # Check the key and algorithm
        assert args[1] == os.getenv("SECRET_KEY")
        assert kwargs["algorithm"] == os.getenv("ALGORITHM")


class TestDecodeToken:
    """Tests for the decode_token function."""

    def test_decode_token_success(self):
        """Test successful token decoding."""
        # Create a valid token
        token = create_access_token(TEST_USERNAME)

        # Decode the token
        payload = decode_token(token)

        # Verify the payload
        assert payload is not None
        assert payload["sub"] == TEST_USERNAME
        assert "exp" in payload

    def test_decode_token_invalid(self):
        """Test decoding an invalid token."""
        algorithm = os.getenv("ALGORITHM")
        invalid_token = jwt.encode(
            {
                "sub": TEST_USERNAME,
                "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
            },
            "wrong_secret_key",
            algorithm=algorithm,
        )

        payload = decode_token(invalid_token)

        # Should return None for invalid tokens
        assert payload is None

    @patch("app.utils.auth.jwt.decode")
    def test_decode_token_jwterror_handling(self, mock_decode, caplog):
        """Test that JWTError is properly handled and logged."""
        # Configure the mock to raise JWTError
        mock_decode.side_effect = JWTError("Test error")

        # Call with any token
        result = decode_token("any.token.here")

        # Should return None on error
        assert result is None

        # Should log the error
        assert "Failed to decode token" in caplog.text

    def test_decode_token_expired(self):
        """Test decoding an expired token."""
        # Create an expired token
        expired_payload = {
            "sub": TEST_USERNAME,
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
        }
        secret_key = os.getenv("SECRET_KEY")
        algorithm = os.getenv("ALGORITHM")
        expired_token = jwt.encode(expired_payload, secret_key, algorithm)

        # Try to decode the expired token
        payload = decode_token(expired_token)

        # Should return None for expired tokens
        assert payload is None
