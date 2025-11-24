"""
Audiobook Utilities - Document Parser

This module provides utilities for parsing various document formats (PDF, DOC, DOCX)
and extracting clean text suitable for TTS conversion.

Optimized for Apple Silicon (M4) with efficient text processing.
"""

import re
from pathlib import Path
from typing import List, Tuple
import PyPDF2
from docx import Document


class DocumentParser:
    """Parse documents and extract clean text for TTS processing."""
    
    def __init__(self):
        """Initialize the document parser."""
        self.supported_formats = ['.pdf', '.doc', '.docx']
    
    def parse_document(self, file_path: str) -> Tuple[str, dict]:
        """
        Parse a document and extract text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (extracted_text, metadata)
            metadata includes: page_count, word_count, char_count
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(
                f"Unsupported format: {extension}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )
        
        # Parse based on file type
        if extension == '.pdf':
            text, page_count = self._parse_pdf(path)
        elif extension in ['.doc', '.docx']:
            text, page_count = self._parse_docx(path)
        else:
            raise ValueError(f"Unsupported format: {extension}")
        
        # Clean the text
        text = self._clean_text(text)
        
        # Generate metadata
        metadata = {
            'page_count': page_count,
            'word_count': len(text.split()),
            'char_count': len(text),
            'format': extension
        }
        
        return text, metadata
    
    def _parse_pdf(self, path: Path) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Args:
            path: Path to PDF file
            
        Returns:
            Tuple of (text, page_count)
        """
        text_parts = []
        
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return '\n\n'.join(text_parts), page_count
    
    def _parse_docx(self, path: Path) -> Tuple[str, int]:
        """
        Extract text from DOCX file.
        
        Args:
            path: Path to DOCX file
            
        Returns:
            Tuple of (text, estimated_page_count)
        """
        doc = Document(path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        full_text = '\n\n'.join(text_parts)
        
        # Estimate page count (roughly 500 words per page)
        word_count = len(full_text.split())
        estimated_pages = max(1, word_count // 500)
        
        return full_text, estimated_pages
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text for TTS processing.
        
        Removes:
        - Excessive whitespace
        - Page numbers (standalone numbers)
        - Common header/footer patterns
        - Multiple consecutive newlines
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove standalone page numbers (numbers on their own line)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove multiple consecutive newlines (keep max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, max_chars: int = 500) -> List[str]:
        """
        Split text into chunks suitable for TTS processing.
        
        Tries to split at sentence boundaries to maintain natural flow.
        Optimized for parallel processing on M4.
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk (default: 500)
            
        Returns:
            List of text chunks
        """
        # Split into sentences (basic sentence detection)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds max_chars, save current chunk
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


def estimate_processing_time(char_count: int, chars_per_second: int = 50) -> Tuple[int, int]:
    """
    Estimate processing time for audiobook generation.
    
    Args:
        char_count: Total character count
        chars_per_second: Processing speed (chars/sec)
                         Default: 50 (conservative estimate for M4)
        
    Returns:
        Tuple of (minutes, seconds)
    """
    total_seconds = char_count / chars_per_second
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return minutes, seconds


if __name__ == "__main__":
    # Simple test
    parser = DocumentParser()
    print("Document Parser initialized successfully!")
    print(f"Supported formats: {', '.join(parser.supported_formats)}")
