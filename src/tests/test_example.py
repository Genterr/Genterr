"""Basic test module."""

def test_example():
    """Basic test to ensure testing infrastructure works."""
    assert True

def test_import():
    """Test that we can import from our package."""
    from src.core import utils
    assert utils is not None