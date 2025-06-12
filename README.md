# Word Analyzer - Generalized Text Analysis Tool

A refactored and generalized version of the word analyzer that creates linguistic fingerprints from text based on word type patterns.

## Features

- **Multiple Input Formats**: Supports PDF, TXT, DOCX, and direct string input
- **Configurable Analysis**: Customizable patterns and analysis parameters
- **Modular Design**: Separated concerns for easy extension and maintenance
- **Dictionary Management**: Comprehensive tools for word classification
- **Fingerprint Generation**: Create unique linguistic patterns from text
- **Command-Line Interface**: Full CLI support with various options

## Installation

```bash
# Basic requirements
pip install PyPDF2  # For PDF support
pip install python-docx  # For DOCX support (optional)
```

## Quick Start

### Basic Usage

```python
from word_analyzer import WordAnalyzer

# Create analyzer
analyzer = WordAnalyzer()

# Analyze text
text = "Your text here..."
word_counts = analyzer.analyze_text(text, input_type='string')

# Generate fingerprint
fingerprint = analyzer.generate_fingerprint()
analyzer.display_fingerprint(fingerprint)
```

### Command Line Usage

```bash
# Analyze a PDF file
python main.py --input document.pdf --author "Author Name"

# Analyze with custom pattern
python main.py --input text.txt --pattern "noun verb adj" --author "Author Name"

# Analyze specific PDF pages
python main.py --input book.pdf --pdf-pages 10 100 --author "Author Name"

# Show dictionary statistics
python main.py --dict-stats

# Clean dictionaries
python dictionary_cleaner.py --clean --save
```

## Configuration

Edit `config.py` to customize the analyzer's behavior:

```python
# Dictionary configuration
DICTIONARY_CONFIG = {
    "dictionary_path": "Dict",
    "dictionary_extension": ".exc",
    "word_types": ["noun", "verb", "adj", ...]
}

# Analysis settings
ANALYSIS_CONFIG = {
    "case_sensitive": False,
    "strip_punctuation": True,
    "plural_handling": True,
}

# Fingerprint patterns
FINGERPRINT_CONFIG = {
    "default_pattern": ["indpron", "shprep", "ldpron", ...],
    "length_constraints": {
        "sh": {"condition": "<=", "value": 2},  # Short words
        "l": {"condition": ">=", "value": 5},   # Long words
        ...
    }
}
```

## Pattern Syntax

Patterns define the word types to search for in order. You can use prefixes to specify length constraints:

- `noun` - Any noun
- `lnoun` - Long noun (5+ characters)
- `shverb` - Short verb (≤2 characters)
- `madj` - Medium adjective (4+ characters)

Available prefixes:
- `sh` - Short (≤2 characters)
- `l` - Long (≥5 characters)
- `ll` - Very long (≥7 characters)
- `m` - Medium (≥4 characters)
- `mm` - Exactly 3 characters
- `ml` - Exactly 4 characters

## Module Structure

- **`word_analyzer.py`**: Main analyzer class
- **`dictionary_manager.py`**: Handles word dictionaries and classification
- **`text_extractor.py`**: Text extraction from various formats
- **`config.py`**: Configuration settings
- **`main.py`**: CLI interface and example usage
- **`dictionary_cleaner.py`**: Dictionary maintenance utilities

## Advanced Usage

### Custom Patterns

```python
# Define a custom pattern
pattern = ['art', 'ladj', 'noun', 'verb', 'prep', 'art', 'noun']

# Generate fingerprint with custom pattern
fingerprint = analyzer.generate_fingerprint(pattern=pattern)
```

### Comparing Fingerprints

```python
# Compare two fingerprints
similarity = analyzer.compare_fingerprints(fingerprint1, fingerprint2)
print(f"Similarity: {similarity:.2%}")
```

### Dictionary Management

```python
from dictionary_manager import DictionaryManager

# Create manager
dict_manager = DictionaryManager()
dict_manager.load_dictionaries()

# Add custom words
dict_manager.add_word("blockchain", "noun")
dict_manager.add_word("googling", "verb")

# Save changes
dict_manager.save_dictionaries()
```

## Example Output

```
Fingerprinting Edgar Allan Poe...
Analyzed 267,453 words, 14,231 unique.

Fingerprint: All of these and more was never more.

                       Words: All | of | these | and  | more | was | never | more
Number of times word appears: 523 | 7821 | 187  | 5422 | 234  | 3901 | 432  | 234

Completed fingerprint in 3.24 seconds
```

## Extending the Analyzer

To add support for new file formats:

1. Create a new extractor class in `text_extractor.py`:
```python
class CustomExtractor(TextExtractor):
    def extract(self, source):
        # Your extraction logic
        return extracted_text
```

2. Register it in the factory:
```python
extractors['custom'] = CustomExtractor
```

## License

This is a refactored version of the original word analyzer, generalized for broader use cases.
