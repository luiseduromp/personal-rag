"""
This file contains pytest fixtures that are available throughout the test suite.
"""
import os
import sys
from pathlib import Path
from typing import Generator

import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fixtures can be added here
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Fixture to get the path to the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Fixture that provides a temporary directory for each test."""
    return tmp_path

# You can add more fixtures here as needed for your tests
