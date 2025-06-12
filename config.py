"""
Configuration file for the word analyzer.
Modify this file to customize the analyzer's behavior.
"""

# Dictionary configuration
DICTIONARY_CONFIG = {
    "dictionary_path": "Dict",  # Path to dictionary files
    "dictionary_extension": ".exc",  # File extension for dictionary files
    "word_types": [
        "conj",  # Conjunction
        "art",  # Article
        "adj",  # Adjective
        "adv",  # Adverb
        "prep",  # Preposition
        "noun",  # Noun
        "verb",  # Verb
        "dpron",  # Demonstrative pronoun
        "indpron",  # Indefinite pronoun
        "intpron",  # Interrogative pronoun
        "opron",  # Other pronoun
        "ppron",  # Personal pronoun
        "refpron",  # Reflexive pronoun
        "relpron",  # Relative pronoun
        "spron",  # Subject pronoun
    ],
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "case_sensitive": False,  # Whether to consider case in word matching
    "strip_punctuation": True,  # Whether to remove punctuation
    "plural_handling": True,  # Whether to handle plural forms
    "max_plural_attempts": 3,  # Maximum attempts for plural stripping
}

# Fingerprint configuration
FINGERPRINT_CONFIG = {
    # Default sentence pattern for fingerprint generation
    # Format: "type" or "prefix+type" where prefix indicates word length constraints
    "default_pattern": [
        "indpron",  # All
        "shprep",  # of
        "ldpron",  # these
        "conj",  # and
        "madv",  # there
        "verb",  # was
        "ladv",  # never
        "madj",  # more
    ],
    # Length constraints for prefixes
    "length_constraints": {
        "sh": {"condition": "<=", "value": 2},  # Short: <= 2 characters
        "mm": {"condition": "==", "value": 3},  # Medium-medium: exactly 3 characters
        "ml": {"condition": "==", "value": 4},  # Medium-long: exactly 4 characters
        "m": {"condition": ">=", "value": 4},  # Medium: >= 4 characters
        "ll": {"condition": ">=", "value": 7},  # Long-long: >= 7 characters
        "l": {"condition": ">=", "value": 5},  # Long: >= 5 characters
    },
    # Default minimum length for words without prefix
    "default_min_length": 3,
}

# Output configuration
OUTPUT_CONFIG = {
    "show_word_counts": True,  # Display word occurrence counts
    "capitalize_first": True,  # Capitalize first word in fingerprint
    "timing_info": True,  # Show execution time
    "verbose": True,  # Show detailed progress information
}
