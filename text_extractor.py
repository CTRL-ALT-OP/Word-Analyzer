"""
Text extraction module supporting multiple input formats.
"""

import os
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

# Optional imports with fallback
try:
    import PyPDF2

    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")

try:
    import docx

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    print("Warning: python-docx not installed. DOCX support disabled.")


class TextExtractor(ABC):
    """Abstract base class for text extractors."""

    @abstractmethod
    def extract(self, source: Union[str, object]) -> str:
        """Extract text from the source."""
        pass


class PDFExtractor(TextExtractor):
    """Extract text from PDF files."""

    def __init__(self, page_range: Optional[Tuple[int, int]] = None):
        """
        Initialize PDF extractor.

        Args:
            page_range: Optional tuple of (start_page, end_page) for page selection
        """
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 is required for PDF extraction")
        self.page_range = page_range

    def extract(self, source: Union[str, object]) -> str:
        """Extract text from PDF file."""
        if isinstance(source, str):
            with open(source, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                return self._extract_from_reader(pdf_reader)
        else:
            # Assume it's already a file-like object
            pdf_reader = PyPDF2.PdfReader(source)
            return self._extract_from_reader(pdf_reader)

    def _extract_from_reader(self, pdf_reader) -> str:
        """Extract text from PDF reader object."""
        total_pages = len(pdf_reader.pages)

        if self.page_range:
            start_page = max(0, self.page_range[0])
            end_page = min(total_pages, self.page_range[1])
        else:
            start_page = 0
            end_page = total_pages

        text_parts = []
        for page_num in range(start_page, end_page):
            page = pdf_reader.pages[page_num]
            text_parts.append(page.extract_text())

        print(f"Extracted text from {end_page - start_page} pages")
        return "\n".join(text_parts)


class TextFileExtractor(TextExtractor):
    """Extract text from plain text files."""

    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize text file extractor.

        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding

    def extract(self, source: Union[str, object]) -> str:
        """Extract text from text file."""
        if not isinstance(source, str):
            # Assume it's already a file-like object
            return source.read()
        with open(source, "r", encoding=self.encoding) as f:
            return f.read()


class DOCXExtractor(TextExtractor):
    """Extract text from DOCX files."""

    def __init__(self):
        """Initialize DOCX extractor."""
        if not HAS_DOCX:
            raise ImportError("python-docx is required for DOCX extraction")

    def extract(self, source: Union[str, object]) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(source)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)


class StringExtractor(TextExtractor):
    """Extract text from a string (pass-through)."""

    def extract(self, source: Union[str, object]) -> str:
        """Return the string as-is."""
        return str(source)


class TextExtractorFactory:
    """Factory for creating appropriate text extractors."""

    @staticmethod
    def create_extractor(input_type: str, **kwargs) -> TextExtractor:
        """
        Create a text extractor based on input type.

        Args:
            input_type: Type of input ('pdf', 'txt', 'docx', 'string')
            **kwargs: Additional arguments for specific extractors

        Returns:
            Appropriate TextExtractor instance
        """
        extractors = {
            "pdf": PDFExtractor,
            "txt": TextFileExtractor,
            "text_file": TextFileExtractor,
            "docx": DOCXExtractor,
            "string": StringExtractor,
        }

        input_type = input_type.lower()
        if input_type not in extractors:
            raise ValueError(f"Unsupported input type: {input_type}")

        extractor_class = extractors[input_type]

        # Filter kwargs for the specific extractor
        if input_type == "pdf":
            return extractor_class(page_range=kwargs.get("page_range"))
        elif input_type in {"txt", "text_file"}:
            return extractor_class(encoding=kwargs.get("encoding", "utf-8"))
        else:
            return extractor_class()

    @staticmethod
    def detect_file_type(filepath: str) -> str:
        """
        Detect file type based on extension.

        Args:
            filepath: Path to the file

        Returns:
            Detected file type
        """
        ext = os.path.splitext(filepath)[1].lower()
        type_map = {
            ".pdf": "pdf",
            ".txt": "txt",
            ".text": "txt",
            ".docx": "docx",
        }
        return type_map.get(ext, "txt")  # Default to text
