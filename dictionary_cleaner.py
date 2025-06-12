"""
Dictionary Cleaner - Standalone utility for dictionary maintenance
Replaces the original 'dictionary fixer.py' with enhanced functionality
"""

import os
from typing import Dict, List
from collections import defaultdict
from dictionary_manager import DictionaryManager
from config import DICTIONARY_CONFIG


class DictionaryCleaner:
    """Utility for cleaning and maintaining word dictionaries."""

    def __init__(self, dict_manager: DictionaryManager = None):
        """Initialize the dictionary cleaner."""
        self.dict_manager = dict_manager or DictionaryManager()

    def clean_all(self, verbose: bool = True) -> Dict[str, Dict[str, int]]:
        """
        Perform all cleaning operations on dictionaries.

        Returns:
            Dictionary of cleaning statistics
        """
        if verbose:
            print("Loading dictionaries...")
        self.dict_manager.load_dictionaries()

        stats = {
            "duplicates_within": {},
            "duplicates_across": {},
            "case_normalized": {},
            "invalid_removed": {},
        }

        # Remove duplicates within each dictionary
        if verbose:
            print("\nRemoving duplicates within dictionaries...")
        stats["duplicates_within"] = self.remove_duplicates_within()

        # Remove cross-dictionary duplicates
        if verbose:
            print("\nRemoving cross-dictionary duplicates...")
        stats["duplicates_across"] = self.dict_manager.remove_duplicates()

        # Normalize case
        if verbose:
            print("\nNormalizing word case...")
        stats["case_normalized"] = self.normalize_case()

        # Remove invalid entries
        if verbose:
            print("\nRemoving invalid entries...")
        stats["invalid_removed"] = self.remove_invalid_words()

        return stats

    def remove_duplicates_within(self) -> Dict[str, int]:
        """Remove duplicates within each dictionary."""
        removed = {}

        for word_type, words in self.dict_manager.dictionaries.items():
            original_count = len(words)
            # Convert to set and back to remove duplicates
            unique_words = set(word.lower() for word in words)
            self.dict_manager.dictionaries[word_type] = unique_words
            removed[word_type] = original_count - len(unique_words)

        return removed

    def normalize_case(self) -> Dict[str, int]:
        """Ensure all words are lowercase."""
        normalized = {}

        for word_type, words in self.dict_manager.dictionaries.items():
            new_words = set()
            count = 0

            for word in words:
                lower_word = word.lower()
                if word != lower_word:
                    count += 1
                new_words.add(lower_word)

            self.dict_manager.dictionaries[word_type] = new_words
            normalized[word_type] = count

        return normalized

    def remove_invalid_words(self) -> Dict[str, int]:
        """Remove words that contain non-alphabetic characters."""
        removed = {}

        for word_type, words in self.dict_manager.dictionaries.items():
            valid_words = set()
            count = 0

            for word in words:
                if word.isalpha():
                    valid_words.add(word)
                else:
                    count += 1

            self.dict_manager.dictionaries[word_type] = valid_words
            removed[word_type] = count

        return removed

    def find_conflicts(self) -> Dict[str, List[str]]:
        """Find words that appear in multiple dictionaries."""
        word_locations = defaultdict(list)

        for word_type, words in self.dict_manager.dictionaries.items():
            for word in words:
                word_locations[word].append(word_type)

        conflicts = {
            word: locations
            for word, locations in word_locations.items()
            if len(locations) > 1
        }

        return conflicts

    def generate_report(
        self, stats: Dict[str, Dict[str, int]], save_path: str = None
    ) -> str:
        """Generate a cleaning report."""
        report_lines = ["Dictionary Cleaning Report", "=" * 50, ""]

        # Summary statistics
        total_stats = defaultdict(int)
        for operation, counts in stats.items():
            for word_type, count in counts.items():
                total_stats[operation] += count

        report_lines.extend(
            [
                "Summary:",
                f"  Total duplicates within dictionaries: {total_stats['duplicates_within']}",
                f"  Total cross-dictionary duplicates: {total_stats['duplicates_across']}",
                f"  Total case normalizations: {total_stats['case_normalized']}",
                f"  Total invalid words removed: {total_stats['invalid_removed']}",
                "",
            ]
        )

        # Detailed statistics
        report_lines.append("Detailed Statistics by Word Type:")
        report_lines.append("-" * 50)

        for word_type in self.dict_manager.word_types:
            report_lines.append(f"\n{word_type}:")
            for operation, counts in stats.items():
                if word_type in counts and counts[word_type] > 0:
                    operation_name = operation.replace("_", " ").title()
                    report_lines.append(f"  {operation_name}: {counts[word_type]}")

        # Final dictionary sizes
        report_lines.extend(["", "Final Dictionary Sizes:", "-" * 50])

        final_stats = self.dict_manager.get_statistics()
        for word_type, count in sorted(final_stats.items()):
            report_lines.append(f"{word_type:10s}: {count:6d} words")

        report_lines.append("-" * 50)
        report_lines.append(f"{'Total':10s}: {sum(final_stats.values()):6d} words")

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)

        return report

    def merge_dictionaries(self, source_type: str, target_type: str) -> int:
        """Merge one dictionary into another."""
        if source_type not in self.dict_manager.dictionaries:
            raise ValueError(f"Source dictionary '{source_type}' not found")
        if target_type not in self.dict_manager.dictionaries:
            raise ValueError(f"Target dictionary '{target_type}' not found")

        source_words = self.dict_manager.dictionaries[source_type]
        target_words = self.dict_manager.dictionaries[target_type]

        added_count = 0
        for word in source_words:
            if word not in target_words:
                target_words.add(word)
                added_count += 1

        # Clear source dictionary
        self.dict_manager.dictionaries[source_type] = set()

        return added_count


def main():
    """Main function for standalone usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dictionary Cleaner - Maintain word dictionaries"
    )
    parser.add_argument(
        "--dict-path",
        type=str,
        default=DICTIONARY_CONFIG["dictionary_path"],
        help="Path to dictionary files",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Perform all cleaning operations"
    )
    parser.add_argument(
        "--find-conflicts",
        action="store_true",
        help="Find words appearing in multiple dictionaries",
    )
    parser.add_argument("--report", type=str, help="Save cleaning report to file")
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Merge source dictionary into target",
    )
    parser.add_argument("--save", action="store_true", help="Save cleaned dictionaries")

    args = parser.parse_args()

    # Initialize cleaner
    dict_manager = DictionaryManager(dictionary_path=args.dict_path)
    cleaner = DictionaryCleaner(dict_manager)

    # Load dictionaries
    print(f"Loading dictionaries from {args.dict_path}...")
    dict_manager.load_dictionaries()

    # Perform operations
    if args.clean:
        stats = cleaner.clean_all()
        report = cleaner.generate_report(stats, args.report)
        print("\n" + report)

        if args.save:
            print(f"\nSaving cleaned dictionaries to {args.dict_path}...")
            dict_manager.save_dictionaries()
            print("Done!")

    if args.find_conflicts:
        conflicts = cleaner.find_conflicts()
        if conflicts:
            print("\nWords appearing in multiple dictionaries:")
            print("-" * 50)
            for word, locations in sorted(conflicts.items()):
                print(f"{word}: {', '.join(locations)}")
        else:
            print("No conflicts found.")

    if args.merge:
        source, target = args.merge
        added = cleaner.merge_dictionaries(source, target)
        print(f"Merged {added} words from {source} into {target}")

        if args.save:
            dict_manager.save_dictionaries()
            print("Saved merged dictionaries.")


if __name__ == "__main__":
    main()
