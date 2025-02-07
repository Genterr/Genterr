"""Test module for the main functionality."""
from src.main import hello_world

def test_hello_world():
    """Test the hello_world function."""
    assert hello_world() == "Hello, World!"

def test_example():
    """Simple test to verify the testing setup works."""
    assert True