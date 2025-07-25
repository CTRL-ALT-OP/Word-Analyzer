"""
Gemini Word Categorizer
Uses Google's Gemini 2.0 Flash to categorize uncategorized words from text sources.
Optimized for parallel processing and high throughput with robust error handling.
"""

import os
import re
import string
import time
import json
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import google.generativeai as genai
    from google.generativeai.types import BlockedPromptException, StopCandidateException

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print(
        "Warning: google-generativeai not installed. Run: pip install google-generativeai"
    )

from config import DICTIONARY_CONFIG, ANALYSIS_CONFIG
from dictionary_manager import DictionaryManager
from text_extractor import TextExtractorFactory


class OptimizedGeminiWordCategorizer:
    """Uses Gemini 2.0 Flash to categorize words with parallel processing and robust error handling."""

    def __init__(self, api_key_file: str = "gemini.api", max_concurrent: int = 20):
        """
        Initialize the optimized Gemini word categorizer.

        Args:
            api_key_file: Path to file containing the Gemini API key
            max_concurrent: Maximum concurrent API requests (reduced default for stability)
        """
        if not HAS_GEMINI:
            raise ImportError(
                "google-generativeai is required. Run: pip install google-generativeai"
            )

        # Load API key
        self.api_key = self._load_api_key(api_key_file)

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        # Validate API key works
        self._validate_api_key()

        # Initialize dictionary manager
        self.dict_manager = DictionaryManager()
        self.dict_manager.load_dictionaries()

        # Word types from config
        self.word_types = DICTIONARY_CONFIG["word_types"]

        # Create categorization prompt
        self.categorization_prompt = self._create_categorization_prompt()

        # Optimization settings (more conservative for stability)
        self.max_concurrent = min(max_concurrent, 20)  # Cap at 20 for stability
        self.save_interval = 50  # Save every 50 words (more frequent)
        self.progress_file = "categorization_progress.json"

        # Rate limiting: More conservative approach
        self.batch_delay = 1.5  # Increased delay between batches
        self.retry_attempts = 3  # Number of retry attempts
        self.base_retry_delay = 1.0  # Base delay for exponential backoff

        # Error tracking
        self.error_stats = {
            "api_key_errors": 0,
            "rate_limit_errors": 0,
            "network_errors": 0,
            "other_errors": 0,
            "total_retries": 0,
        }

    def _load_api_key(self, api_key_file: str) -> str:
        """Load API key from file."""
        try:
            with open(api_key_file, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            return api_key
        except FileNotFoundError as e:
            raise FileNotFoundError(f"API key file not found: {api_key_file}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading API key: {e}") from e

    def _validate_api_key(self) -> None:
        """Validate that the API key works with a simple test call."""
        try:
            print("Validating API key...")
            test_response = self.model.generate_content("test")
            print("SUCCESS: API key validation successful")
        except Exception as e:
            print(f"ERROR: API key validation failed: {e}")
            if "API key" in str(e).lower() or "authentication" in str(e).lower():
                raise RuntimeError(f"API key is invalid or expired: {e}") from e
            else:
                print(
                    "Warning: API validation failed but continuing (might be temporary)"
                )

    def _create_categorization_prompt(self) -> str:
        """Create the prompt template for word categorization."""
        word_types_desc = {
            "conj": "Conjunction (and, but, or, etc.)",
            "art": "Article (a, an, the)",
            "adj": "Adjective (describes nouns)",
            "adv": "Adverb (describes verbs, adjectives, or other adverbs)",
            "prep": "Preposition (in, on, at, by, etc.)",
            "noun": "Noun (person, place, thing, or idea)",
            "verb": "Verb (action or state of being)",
            "dpron": "Demonstrative pronoun (this, that, these, those)",
            "indpron": "Indefinite pronoun (all, some, any, none, etc.)",
            "intpron": "Interrogative pronoun (who, what, which, etc.)",
            "opron": "Other pronoun",
            "ppron": "Personal pronoun (I, you, he, she, it, we, they)",
            "refpron": "Reflexive pronoun (myself, yourself, himself, etc.)",
            "relpron": "Relative pronoun (who, whom, whose, which, that)",
            "spron": "Subject pronoun (used as subjects in sentences)",
            "pnoun": "Proper noun (person, place, thing, or idea)",
        }

        prompt = """You are a linguistic expert. I need you to categorize English words into specific grammatical types.

Available categories:
"""

        for word_type, description in word_types_desc.items():
            prompt += f"- {word_type}: {description}\n"

        prompt += """
- uncategorized: Use this if the word is not a standard English word, or doesn't fit the above categories clearly.

Rules:
1. Respond with ONLY the category code (e.g., "noun", "verb", "adj", "uncategorized")
2. Consider the most common usage of the word
3. If a word can be multiple types, choose the most primary/common usage
4. Use "uncategorized" for archaic words, or non-standard English words
5. Be consistent with similar words

Word to categorize: """

        return prompt

    def extract_uncategorized_words(self, pdf_path: str) -> Set[str]:
        """
        Extract words from PDF that are not yet categorized in any dictionary.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Set of uncategorized words
        """
        print(f"Extracting text from {pdf_path}...")

        # Extract text from PDF
        extractor = TextExtractorFactory.create_extractor("pdf")
        text = extractor.extract(pdf_path)

        print("Processing text and identifying uncategorized words...")

        # Extract and clean words
        words = set()
        for line in text.split("\n"):
            for word in line.split():
                if ANALYSIS_CONFIG["strip_punctuation"]:
                    word = word.strip(string.punctuation)

                if word and word.isalpha():
                    if not ANALYSIS_CONFIG["case_sensitive"]:
                        word = word.lower()
                    words.add(word)

        uncategorized_words = {
            word for word in words if self.dict_manager.get_word_type(word) is None
        }
        print(
            f"Found {len(words)} total words, {len(uncategorized_words)} uncategorized"
        )
        return uncategorized_words

    def categorize_word_with_retry(self, word: str) -> str:
        """
        Categorize a single word using Gemini with retry logic.

        Args:
            word: The word to categorize

        Returns:
            The category assigned by Gemini
        """
        for attempt in range(self.retry_attempts):
            try:
                prompt = self.categorization_prompt + word
                response = self.model.generate_content(prompt)
                category = response.text.strip().lower()

                # Validate the category
                valid_categories = set(self.word_types + ["uncategorized"])
                if category not in valid_categories:
                    print(
                        f"Warning: Invalid category '{category}' for word '{word}', using 'uncategorized'"
                    )
                    return "uncategorized"

                return category

            except Exception as e:
                error_msg = str(e).lower()

                # Classify error types
                if (
                    "api key" in error_msg
                    or "authentication" in error_msg
                    or "expired" in error_msg
                ):
                    self.error_stats["api_key_errors"] += 1
                    print(f"API key error for word '{word}': {e}")
                    if attempt == self.retry_attempts - 1:
                        print(
                            f"Max retries reached for word '{word}' due to API key error"
                        )
                        return "uncategorized"
                elif (
                    "quota" in error_msg or "rate" in error_msg or "limit" in error_msg
                ):
                    self.error_stats["rate_limit_errors"] += 1
                    print(f"Rate limit error for word '{word}': {e}")
                elif (
                    "network" in error_msg
                    or "connection" in error_msg
                    or "timeout" in error_msg
                ):
                    self.error_stats["network_errors"] += 1
                    print(f"Network error for word '{word}': {e}")
                else:
                    self.error_stats["other_errors"] += 1
                    print(f"Other error for word '{word}': {e}")

                # Exponential backoff with jitter
                if attempt < self.retry_attempts - 1:
                    self.error_stats["total_retries"] += 1
                    delay = self.base_retry_delay * (2**attempt) + random.uniform(0, 1)
                    print(
                        f"  Retrying in {delay:.1f}s... (attempt {attempt + 2}/{self.retry_attempts})"
                    )
                    time.sleep(delay)

        print(
            f"Failed to categorize word '{word}' after {self.retry_attempts} attempts"
        )
        return "uncategorized"

    def categorize_words_batch(self, words: List[str]) -> Dict[str, str]:
        """
        Categorize a batch of words using ThreadPoolExecutor with error handling.

        Args:
            words: List of words to categorize

        Returns:
            Dictionary mapping words to their categories
        """
        categorizations = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            future_to_word = {
                executor.submit(self.categorize_word_with_retry, word): word
                for word in words
            }

            # Collect results as they complete
            for future in as_completed(future_to_word):
                word = future_to_word[future]
                try:
                    category = future.result(timeout=30)  # 30 second timeout per word
                    categorizations[word] = category
                except Exception as e:
                    print(f"Final error processing word '{word}': {e}")
                    categorizations[word] = "uncategorized"
                    self.error_stats["other_errors"] += 1

        return categorizations

    def save_progress(
        self, processed_words: Dict[str, str], remaining_words: Set[str]
    ) -> None:
        """Save progress to file for resumption."""
        progress_data = {
            "processed_words": processed_words,
            "remaining_words": list(remaining_words),
            "timestamp": time.time(),
            "error_stats": self.error_stats,
        }

        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")

    def load_progress(self) -> Tuple[Dict[str, str], Set[str]]:
        """Load progress from file if it exists."""
        if not os.path.exists(self.progress_file):
            return {}, set()

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)

            processed_words = progress_data.get("processed_words", {})
            remaining_words = set(progress_data.get("remaining_words", []))

            # Load error stats if available
            if "error_stats" in progress_data:
                self.error_stats = progress_data["error_stats"]

            print(
                f"Loaded progress: {len(processed_words)} processed, {len(remaining_words)} remaining"
            )
            if self.error_stats.get("total_retries", 0) > 0:
                print(
                    f"Previous session had {self.error_stats['total_retries']} retries"
                )

            return processed_words, remaining_words

        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
            return {}, set()

    def cleanup_progress(self) -> None:
        """Remove progress file after successful completion."""
        try:
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        except Exception as e:
            print(f"Warning: Could not remove progress file: {e}")

    def print_error_stats(self) -> None:
        """Print error statistics for debugging."""
        total_errors = sum(self.error_stats.values()) - self.error_stats.get(
            "total_retries", 0
        )
        if total_errors > 0:
            print("\n" + "=" * 40)
            print("ERROR STATISTICS")
            print("=" * 40)
            print(f"API Key Errors:    {self.error_stats['api_key_errors']}")
            print(f"Rate Limit Errors: {self.error_stats['rate_limit_errors']}")
            print(f"Network Errors:    {self.error_stats['network_errors']}")
            print(f"Other Errors:      {self.error_stats['other_errors']}")
            print(f"Total Retries:     {self.error_stats['total_retries']}")
            print(f"Total Errors:      {total_errors}")

            if (
                self.error_stats["api_key_errors"]
                > self.error_stats["rate_limit_errors"]
            ):
                print("\nWARNING: High API key errors detected. Consider:")
                print("   - Checking if your API key has expired")
                print("   - Verifying your API key quota limits")
                print("   - Checking if billing is set up correctly")

    def categorize_words_optimized(self, words: Set[str]) -> Dict[str, str]:
        """
        Categorize multiple words with optimized parallel processing and error handling.

        Args:
            words: Set of words to categorize

        Returns:
            Dictionary mapping words to their categories
        """
        # Check for existing progress
        processed_words, remaining_words = self.load_progress()

        # If no progress or different word set, start fresh
        if not remaining_words or not remaining_words.issubset(words):
            remaining_words = words.copy()
            processed_words = {}

        # Remove already processed words
        remaining_words = remaining_words - set(processed_words.keys())

        if not remaining_words:
            print("All words already processed!")
            return processed_words

        print(
            f"Categorizing {len(remaining_words)} words using parallel Gemini 2.0 Flash..."
        )
        print(
            f"Using {self.max_concurrent} concurrent requests with {self.batch_delay}s batch delay"
        )

        # Convert to list for batching
        words_list = sorted(list(remaining_words))
        total_words = len(words_list)

        # Process in batches to respect rate limits
        batch_size = self.max_concurrent
        start_time = time.time()

        for i in range(0, total_words, batch_size):
            batch_start = time.time()
            batch = words_list[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_words + batch_size - 1) // batch_size

            print(
                f"\nBatch {batch_num}/{total_batches}: Processing {len(batch)} words..."
            )

            # Process batch in parallel
            batch_results = self.categorize_words_batch(batch)
            processed_words.update(batch_results)

            # Update remaining words
            remaining_words -= set(batch)

            # Show batch results (limited to avoid spam)
            sample_results = list(batch_results.items())[:5]
            for word, category in sample_results:
                print(f"  {word} -> {category}")
            if len(batch_results) > 5:
                print(f"  ... and {len(batch_results) - 5} more")

            # Save progress intermittently
            if (
                len(processed_words) % self.save_interval == 0
                or i + batch_size >= total_words
            ):
                print(f"Saving progress... ({len(processed_words)} words processed)")
                self.save_progress(processed_words, remaining_words)

                # Also save dictionaries intermittently
                self._save_intermediate_results(processed_words)

            # Rate limiting between batches (except for last batch)
            if i + batch_size < total_words:
                elapsed = time.time() - batch_start
                sleep_time = max(0, self.batch_delay - elapsed)
                if sleep_time > 0:
                    print(f"Rate limiting: waiting {sleep_time:.2f}s...")
                    time.sleep(sleep_time)

            # Progress update
            progress = ((i + len(batch)) / total_words) * 100
            elapsed_total = time.time() - start_time
            words_per_second = (
                len(processed_words) / elapsed_total if elapsed_total > 0 else 0
            )
            eta = (
                (total_words - len(processed_words)) / words_per_second
                if words_per_second > 0
                else 0
            )

            print(
                f"Progress: {progress:.1f}% ({len(processed_words)}/{total_words}) | "
                f"Rate: {words_per_second:.1f} words/sec | "
                f"ETA: {eta/60:.1f} minutes"
            )

        print(
            f"\nCompleted! Processed {len(processed_words)} words in {(time.time() - start_time)/60:.1f} minutes"
        )
        return processed_words

    def _save_intermediate_results(self, categorizations: Dict[str, str]) -> None:
        """Save intermediate results to dictionaries without full save."""
        try:
            # Count what we're adding
            temp_counts = {word_type: 0 for word_type in self.word_types}
            temp_counts["uncategorized"] = 0

            for word, category in categorizations.items():
                if category == "uncategorized":
                    temp_counts["uncategorized"] += 1

                elif self.dict_manager.get_word_type(word) is None:
                    self.dict_manager.add_word(word, category)
                    temp_counts[category] += 1
            # Save only if we have new words
            if any(
                count > 0
                for word_type, count in temp_counts.items()
                if word_type != "uncategorized"
            ):
                self.dict_manager.save_dictionaries()

        except Exception as e:
            print(f"Warning: Could not save intermediate results: {e}")

    def add_categorized_words(self, categorizations: Dict[str, str]) -> Dict[str, int]:
        """
        Add categorized words to their respective dictionaries.

        Args:
            categorizations: Dictionary mapping words to categories

        Returns:
            Dictionary showing count of words added to each category
        """
        added_counts = {word_type: 0 for word_type in self.word_types}
        added_counts["uncategorized"] = 0

        for word, category in categorizations.items():
            if category == "uncategorized":
                added_counts["uncategorized"] += 1

            elif self.dict_manager.get_word_type(word) is None:
                self.dict_manager.add_word(word, category)
                added_counts[category] += 1
        return added_counts

    def process_pdf(
        self, pdf_path: str, save_dictionaries: bool = True
    ) -> Dict[str, int]:
        """
        Complete process: extract uncategorized words from PDF and categorize them with optimization.

        Args:
            pdf_path: Path to the PDF file
            save_dictionaries: Whether to save updated dictionaries

        Returns:
            Dictionary showing count of words added to each category
        """
        print("=" * 60)
        print("OPTIMIZED GEMINI WORD CATEGORIZER")
        print("=" * 60)

        # Extract uncategorized words
        uncategorized_words = self.extract_uncategorized_words(pdf_path)

        if not uncategorized_words:
            print("No uncategorized words found!")
            return {}

        # Show sample of words to be categorized
        sample_words = sorted(list(uncategorized_words))[:10]
        print(f"\nSample words to categorize: {', '.join(sample_words)}")
        if len(uncategorized_words) > 10:
            print(f"... and {len(uncategorized_words) - 10} more")

        # Categorize words with optimization
        categorizations = self.categorize_words_optimized(uncategorized_words)

        # Add to dictionaries (final pass)
        print("\nFinalizing categorized words in dictionaries...")
        added_counts = self.add_categorized_words(categorizations)

        # Display results
        print("\n" + "=" * 40)
        print("CATEGORIZATION RESULTS")
        print("=" * 40)

        total_categorized = sum(
            count
            for category, count in added_counts.items()
            if category != "uncategorized"
        )

        for category, count in added_counts.items():
            if count > 0:
                print(f"{category:12s}: {count:4d} words")

        print(f"{'Total':12s}: {total_categorized:4d} words categorized")
        print(f"{'Uncategorized':12s}: {added_counts.get('uncategorized', 0):4d} words")

        # Print error statistics
        self.print_error_stats()

        # Save dictionaries if requested
        if save_dictionaries and total_categorized > 0:
            print("\nSaving final dictionaries...")
            self.dict_manager.save_dictionaries()
            print("Dictionaries saved successfully!")

        # Clean up progress file
        self.cleanup_progress()

        return added_counts


# Backward compatibility alias
GeminiWordCategorizer = OptimizedGeminiWordCategorizer


def main():
    """Main function to run the categorizer."""
    # Configuration
    pdf_path = "texts/edgar_allan_poe__complete_tales_-_edgar_allan_poe.pdf"
    api_key_file = "gemini.api"
    max_concurrent = 15  # Reduced for better stability

    try:
        # Initialize categorizer
        categorizer = OptimizedGeminiWordCategorizer(api_key_file, max_concurrent)

        # Process the PDF
        results = categorizer.process_pdf(pdf_path, save_dictionaries=True)

        print("\n" + "=" * 60)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
