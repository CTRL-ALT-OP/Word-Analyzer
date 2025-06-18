"""
Pytest configuration and fixtures for BubbleVisualizer improved tests.
Minimal configuration focused on 4K canvas and placeholder.txt dataset testing.
"""

import pytest
import tempfile
import os


@pytest.fixture
def temp_output_file():
    """Fixture that provides a temporary output file path and cleans it up after test."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        output_path = tmp_file.name

    yield output_path

    # Cleanup
    if os.path.exists(output_path):
        os.unlink(output_path)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "quality: marks tests as quality assurance tests"
    )
    config.addinivalue_line("markers", "cli: marks tests as CLI functionality tests")
