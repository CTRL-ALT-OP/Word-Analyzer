"""
Generalized Word Analyzer
Analyzes text to create linguistic fingerprints based on word type patterns.
"""

import string
from collections import Counter
from typing import List, Optional, Tuple, Union
from pathlib import Path

from config import ANALYSIS_CONFIG, FINGERPRINT_CONFIG, OUTPUT_CONFIG
from dictionary_manager import DictionaryManager
from text_extractor import TextExtractorFactory


class WordAnalyzer:
    """Main word analyzer class."""

    def __init__(self, dictionary_manager: Optional[DictionaryManager] = None):
        """
        Initialize the word analyzer.

        Args:
            dictionary_manager: Optional DictionaryManager instance
        """
        self.dict_manager = dictionary_manager or DictionaryManager()
        if not self.dict_manager.dictionaries:
            if OUTPUT_CONFIG["verbose"]:
                print("Loading dictionaries...")
            self.dict_manager.load_dictionaries()

        self.word_counts: Optional[Counter] = None
        self.fingerprint: Optional[List[Tuple[str, str, int]]] = None

    def analyze_text(
        self,
        text_source: Union[str, Path],
        input_type: Optional[str] = None,
        **extractor_kwargs,
    ) -> Counter:
        """
        Analyze text from various sources.

        Args:
            text_source: Path to file or text string
            input_type: Type of input (auto-detected if None)
            **extractor_kwargs: Additional arguments for text extraction

        Returns:
            Counter of word frequencies
        """
        # Auto-detect input type if not specified
        if input_type is None:
            if isinstance(text_source, (str, Path)) and Path(text_source).exists():
                input_type = TextExtractorFactory.detect_file_type(str(text_source))
            else:
                input_type = "string"

        # Extract text
        extractor = TextExtractorFactory.create_extractor(
            input_type, **extractor_kwargs
        )
        text = extractor.extract(text_source)

        # Process text
        self.word_counts = self._process_text(text)
        return self.word_counts

    def _process_text(self, text: str) -> Counter:
        """Process text and count word occurrences."""
        words = []

        # Split into words
        for line in text.split("\n"):
            for word in line.split():
                if ANALYSIS_CONFIG["strip_punctuation"]:
                    # Remove punctuation
                    word = word.strip(string.punctuation)

                if word and word.isalpha():
                    if not ANALYSIS_CONFIG["case_sensitive"]:
                        word = word.lower()
                    words.append(word)

        return Counter(words)

    def generate_fingerprint(
        self, pattern: Optional[List[str]] = None, word_counts: Optional[Counter] = None
    ) -> List[Tuple[str, str, int]]:
        """
        Generate a linguistic fingerprint based on word type pattern.

        Args:
            pattern: List of word types defining the fingerprint pattern
            word_counts: Optional word frequency counter (uses self.word_counts if None)

        Returns:
            List of (pattern_element, word, count) tuples
        """
        if word_counts is None:
            if self.word_counts is None:
                raise ValueError("No word counts available. Run analyze_text first.")
            word_counts = self.word_counts

        if pattern is None:
            pattern = FINGERPRINT_CONFIG["default_pattern"]

        result = []
        used_words = set()

        for pattern_element in pattern:
            word_info = self._find_word_for_pattern(
                pattern_element, word_counts, used_words
            )
            if word_info:
                result.append((pattern_element, word_info[0], word_info[1]))
                used_words.add(word_info[0])
            else:
                # Pattern element not satisfied
                result.append((pattern_element, "???", 0))

        # Capitalize first word if configured
        if result and OUTPUT_CONFIG["capitalize_first"]:
            result[0] = (result[0][0], result[0][1].capitalize(), result[0][2])

        self.fingerprint = result
        return result

    def _find_word_for_pattern(
        self, pattern_element: str, word_counts: Counter, used_words: set
    ) -> Optional[Tuple[str, int]]:
        """Find a word matching the pattern element."""
        # Extract prefix and word type
        prefix = None
        word_type = pattern_element

        # Check for length prefix
        for prefix_key in FINGERPRINT_CONFIG["length_constraints"]:
            if pattern_element.startswith(prefix_key):
                prefix = prefix_key
                word_type = pattern_element[len(prefix) :]
                break

        # Get length constraint
        length_check = self._get_length_checker(prefix)

        # Search for matching word
        for word, count in word_counts.most_common():
            if word in used_words:
                continue

            # Check length constraint
            if not length_check(len(word)):
                continue

            # Check word type
            if self.dict_manager.get_word_type(word) == word_type:
                return (word, count)

        return None

    def _get_length_checker(self, prefix: Optional[str]):
        """Get a function to check word length based on prefix."""
        if prefix is None:
            min_length = FINGERPRINT_CONFIG["default_min_length"]
            return lambda length: length >= min_length

        constraint = FINGERPRINT_CONFIG["length_constraints"][prefix]
        condition = constraint["condition"]
        value = constraint["value"]

        if condition == "<=":
            return lambda length: length <= value
        elif condition == ">=":
            return lambda length: length >= value
        elif condition == "==":
            return lambda length: length == value
        else:
            raise ValueError(f"Unknown condition: {condition}")

    def display_fingerprint(
        self, fingerprint: Optional[List[Tuple[str, str, int]]] = None
    ):
        """Display the fingerprint in a formatted way."""
        if fingerprint is None:
            if self.fingerprint is None:
                raise ValueError(
                    "No fingerprint available. Run generate_fingerprint first."
                )
            fingerprint = self.fingerprint

        # Create sentence
        words = [item[1] for item in fingerprint]
        print("\nFingerprint: ", end="")
        print(" ".join(words) + ".")

        if OUTPUT_CONFIG["show_word_counts"]:
            # Display detailed table
            max_len = max(len(item[1]) for item in fingerprint)

            print("\n" + " " * 23 + "Words: ", end="")
            print(" | ".join(f"{item[1]:<{max_len}}" for item in fingerprint))

            print("Number of times word appears: ", end="")
            print(" | ".join(f"{item[2]:<{max_len}}" for item in fingerprint))

    def compare_fingerprints(
        self,
        fingerprint1: List[Tuple[str, str, int]],
        fingerprint2: List[Tuple[str, str, int]],
    ) -> float:
        """
        Compare two fingerprints for similarity.

        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint

        Returns:
            Similarity score (0-1)
        """
        if len(fingerprint1) != len(fingerprint2):
            raise ValueError("Fingerprints must have the same length")

        matches = sum(
            1
            for i in range(len(fingerprint1))
            if fingerprint1[i][1] == fingerprint2[i][1]
        )

        return matches / len(fingerprint1)

    def save_fingerprint(
        self, filepath: str, fingerprint: Optional[List[Tuple[str, str, int]]] = None
    ):
        """Save a fingerprint to file."""
        if fingerprint is None:
            fingerprint = self.fingerprint

        if fingerprint is None:
            raise ValueError("No fingerprint to save")

        with open(filepath, "w", encoding="utf-8") as f:
            # Save as readable format
            f.write("# Word Analyzer Fingerprint\n")
            f.write("# Pattern | Word | Count\n")
            for pattern, word, count in fingerprint:
                f.write(f"{pattern}\t{word}\t{count}\n")

    def load_fingerprint(self, filepath: str) -> List[Tuple[str, str, int]]:
        """Load a fingerprint from file."""
        fingerprint = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 3:
                        pattern, word, count = parts
                        fingerprint.append((pattern, word, int(count)))

        return fingerprint
