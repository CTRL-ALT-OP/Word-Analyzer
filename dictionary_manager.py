"""
Dictionary management module for word classification.
Handles loading, caching, and word lookups.
"""

import os
from typing import Dict, List, Optional, Set
from config import DICTIONARY_CONFIG, ANALYSIS_CONFIG


class DictionaryManager:
    """Manages word dictionaries for different word types."""

    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        word_types: Optional[List[str]] = None,
        extension: Optional[str] = None,
    ):
        """
        Initialize the dictionary manager.

        Args:
            dictionary_path: Path to dictionary files
            word_types: List of word types to load
            extension: File extension for dictionary files
        """
        self.dictionary_path = dictionary_path or DICTIONARY_CONFIG["dictionary_path"]
        self.word_types = word_types or DICTIONARY_CONFIG["word_types"]
        self.extension = extension or DICTIONARY_CONFIG["dictionary_extension"]
        self.dictionaries: Dict[str, Set[str]] = {}
        self.case_sensitive = ANALYSIS_CONFIG["case_sensitive"]

    def load_dictionaries(self) -> None:
        """Load all dictionaries from files."""
        for word_type in self.word_types:
            filepath = os.path.join(
                self.dictionary_path, f"{word_type}{self.extension}"
            )
            if os.path.exists(filepath):
                self.dictionaries[word_type] = self._load_dictionary_file(filepath)
            else:
                print(f"Warning: Dictionary file not found: {filepath}")
                self.dictionaries[word_type] = set()

    def _load_dictionary_file(self, filepath: str) -> Set[str]:
        """Load a single dictionary file."""
        words = set()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    for word in line.strip().split():
                        if word:
                            if not self.case_sensitive:
                                word = word.lower()
                            words.add(word)
        except Exception as e:
            print(f"Error loading dictionary {filepath}: {e}")
        return words

    def get_word_type(self, word: str, handle_plurals: bool = None) -> Optional[str]:
        """
        Get the word type for a given word.

        Args:
            word: The word to classify
            handle_plurals: Whether to try plural stripping

        Returns:
            The word type or None if not found
        """
        if handle_plurals is None:
            handle_plurals = ANALYSIS_CONFIG["plural_handling"]

        if not self.case_sensitive:
            word = word.lower()

        # Direct lookup
        for word_type, word_set in self.dictionaries.items():
            if word in word_set:
                return word_type

        # Try plural stripping if enabled
        return self._check_with_plural_stripping(word) if handle_plurals else None

    def _check_with_plural_stripping(self, word: str) -> Optional[str]:
        """Try to find word type by stripping plural suffixes."""
        max_attempts = ANALYSIS_CONFIG["max_plural_attempts"]
        current_word = word

        for _ in range(max_attempts):
            if current_word.endswith("es") and len(current_word) > 2:
                current_word = current_word[:-2]
            elif current_word.endswith("s") and len(current_word) > 1:
                current_word = current_word[:-1]
            else:
                break

            for word_type, word_set in self.dictionaries.items():
                if current_word in word_set:
                    return word_type

        return None

    def add_word(self, word: str, word_type: str) -> None:
        """Add a word to a specific dictionary."""
        if word_type not in self.dictionaries:
            self.dictionaries[word_type] = set()

        if not self.case_sensitive:
            word = word.lower()

        self.dictionaries[word_type].add(word)

    def remove_word(self, word: str, word_type: str) -> bool:
        """Remove a word from a specific dictionary."""
        if word_type not in self.dictionaries:
            return False

        if not self.case_sensitive:
            word = word.lower()

        try:
            self.dictionaries[word_type].remove(word)
            return True
        except KeyError:
            return False

    def save_dictionaries(self, path: Optional[str] = None) -> None:
        """Save all dictionaries back to files."""
        save_path = path or self.dictionary_path

        for word_type, words in self.dictionaries.items():
            filepath = os.path.join(save_path, f"{word_type}{self.extension}")
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    for word in sorted(words):
                        f.write(f"{word}\n")
                print(f"Saved {len(words)} words to {filepath}")
            except Exception as e:
                print(f"Error saving dictionary {filepath}: {e}")

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about loaded dictionaries."""
        return {word_type: len(words) for word_type, words in self.dictionaries.items()}

    def remove_duplicates(self) -> Dict[str, int]:
        """Remove duplicates across dictionaries, keeping articles unique."""
        removed_counts = {}

        # Articles should not appear in other dictionaries
        article_words = self.dictionaries.get("art", set())

        for word_type in self.word_types:
            if word_type == "art":
                continue

            removed = 0
            words_to_remove = []

            for word in self.dictionaries.get(word_type, set()):
                if word in article_words:
                    words_to_remove.append(word)
                    removed += 1

            for word in words_to_remove:
                self.dictionaries[word_type].remove(word)

            removed_counts[word_type] = removed

        return removed_counts
