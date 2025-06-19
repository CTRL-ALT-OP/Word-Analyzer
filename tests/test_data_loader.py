"""
Shared test data loader for bubble visualizer tests.
Extracted from existing tests to be reused across multiple test files.
"""

import os
import sys
import string
from collections import Counter
from typing import Counter as CounterType

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from text_extractor import TextExtractorFactory
from dictionary_manager import DictionaryManager
from config import ANALYSIS_CONFIG


class TestDataLoader:
    """Helper class to load and process the placeholder.txt dataset."""

    @staticmethod
    def load_placeholder_dataset() -> CounterType:
        """Load placeholder.txt and process it through word counting algorithm."""
        # Extract text from placeholder.txt
        extractor = TextExtractorFactory.create_extractor("txt")
        text_content = extractor.extract("texts/placeholder.txt")

        # Process text using the same algorithm as specified by user
        words = []

        # Split into words
        for line in text_content.split("\n"):
            for word in line.split():
                if ANALYSIS_CONFIG["strip_punctuation"]:
                    # Remove punctuation
                    word = word.strip(string.punctuation)

                if word and word.isalpha():
                    if not ANALYSIS_CONFIG["case_sensitive"]:
                        word = word.lower()
                    words.append(word)

        return Counter(words)

    @staticmethod
    def get_real_dict_manager() -> DictionaryManager:
        """Get a real dictionary manager with loaded dictionaries."""
        dict_manager = DictionaryManager()
        if not dict_manager.dictionaries:
            dict_manager.load_dictionaries()
        return dict_manager
