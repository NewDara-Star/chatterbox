"""
Audiobook Utilities - Document Parser

This module provides utilities for parsing various document formats (PDF, DOC, DOCX)
and extracting clean text suitable for TTS conversion.

Optimized for Apple Silicon (M4) with efficient text processing.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import PyPDF2
from docx import Document


class DocumentParser:
    """Parse documents and extract clean text for TTS processing."""
    
    def __init__(self, use_llm_cleanup: bool = False, llm_provider: str = None, api_key: str = None):
        """
        Initialize the document parser.
        
        Args:
            use_llm_cleanup: Whether to use LLM for intelligent text cleanup
            llm_provider: LLM provider for cleanup ("anthropic" or "openai")
            api_key: Optional API key (uses .env if not provided)
        """
        self.supported_formats = ['.pdf', '.doc', '.docx', '.txt']
        self.use_llm_cleanup = use_llm_cleanup
        self.llm_provider = llm_provider
        self.api_key = api_key
        
        # Initialize LLM client if cleanup is enabled
        if self.use_llm_cleanup:
            self._init_llm_client()
    
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
            # Apply LLM cleanup if enabled specifically for DOCX, before general cleanup
            if self.use_llm_cleanup:
                text = self._clean_text_with_llm(text) # Assuming this method exists or will be added
        elif extension == '.txt':
            text, page_count = self._parse_txt(path)
        else:
            raise ValueError(f"Unsupported format: {extension}")
        
        # Clean the text (Regex-based)
        text = self._clean_text(text)
        
        # Apply LLM cleanup if enabled (this is the general one, after regex cleanup)
        if self.use_llm_cleanup and extension != '.pdf': # Only apply if not already applied for PDF
            text = self._clean_text_with_llm(text) # Assuming this method exists or will be added
        
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
            Tuple of (text, page_count)
        """
        doc = Document(path)
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text_parts.append(paragraph.text)
                
        # Estimate page count (approx 500 words per page)
        text = '\n\n'.join(text_parts)
        word_count = len(text.split())
        page_count = max(1, word_count // 500)
        
        return text, page_count

    def _parse_txt(self, path: Path) -> Tuple[str, int]:
        """
        Extract text from TXT file.
        
        Args:
            path: Path to TXT file
            
        Returns:
            Tuple of (text, page_count)
        """
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Estimate page count (approx 500 words per page)
        word_count = len(text.split())
        page_count = max(1, word_count // 500)
        
        return text, page_count
        

        
        return full_text, estimated_pages
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text for TTS processing.
        
        Advanced cleaning includes:
        - Hyphenation fixes (rejoin words split across lines)
        - Page number removal (standalone numbers, "Page X of Y")
        - Header/footer patterns (common formats)
        - Line merging (fix PDF hard wraps)
        - Image caption removal
        - Excessive whitespace cleanup
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # 1. Fix hyphenated words at line breaks
        # Pattern: "word-\n" or "word-\r\n" -> "word"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # 2. Remove common page number patterns
        # "Page 5", "Page 5 of 20", "5 | Chapter Name", etc.
        text = re.sub(r'\n\s*Page\s+\d+(\s+of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\|\s*[\w\s]+\s*\n', '\n', text)  # "5 | Chapter"
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers
        
        # 3. Remove common header/footer patterns
        # "Chapter X", "© Copyright", timestamps, URLs
        text = re.sub(r'\n\s*Chapter\s+\d+\s*\n', '\n\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*©.*?\d{4}.*?\n', '\n', text)  # Copyright lines
        text = re.sub(r'\n\s*https?://\S+\s*\n', '\n', text)  # URLs on their own line
        
        # 4. Remove image/figure captions
        # "Figure 1:", "Image:", "Photo by", etc.
        text = re.sub(r'\n\s*(Figure|Fig\.|Image|Photo|Illustration)\s*\d*\s*:?.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # 5. Merge broken lines (PDF hard wraps)
        # If a line ends with a lowercase letter and the next starts with lowercase, merge them
        # But preserve paragraph breaks (double newlines)
        lines = text.split('\n')
        merged_lines = []
        i = 0
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                merged_lines.append('')
                i += 1
                continue
                
            # Check if we should merge with next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Merge if current ends with lowercase and next starts with lowercase
                # (indicates mid-sentence break)
                if (current and next_line and 
                    current[-1].islower() and 
                    next_line[0].islower() and
                    not current.endswith(('.', '!', '?', ':', ';'))):
                    merged_lines.append(current + ' ' + next_line)
                    i += 2
                    continue
            
            merged_lines.append(current)
            i += 1
        
        text = '\n'.join(merged_lines)
        
        # 6. Clean up excessive whitespace
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        # 7. Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 8. Remove empty lines at start and end
        text = text.strip()
        
        return text
    
    def _init_llm_client(self):
        """Initialize LLM client for text cleanup."""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if not self.llm_provider:
            # Default to Anthropic
            self.llm_provider = "anthropic"
        
        if self.llm_provider == "anthropic":
            import anthropic
            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found. Set in .env or pass as parameter.")
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            self.llm_model = "claude-3-5-sonnet-20241022"
        elif self.llm_provider == "openai":
            import openai
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found. Set in .env or pass as parameter.")
            self.llm_client = openai.OpenAI(api_key=api_key)
            self.llm_model = "gpt-4-turbo-preview"
        else:
            raise ValueError(f"Invalid LLM provider: {self.llm_provider}")
    
    def _clean_text_with_llm(self, text: str) -> str:
        """
        Use LLM to intelligently clean OCR errors and formatting issues.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text suitable for TTS
        """
        print("🧠 Cleaning text with LLM (this may take a moment)...")
        
        system_prompt = """You are a text cleanup expert. Your task is to clean OCR-extracted text for audiobook narration.

CRITICAL RULES:
1. **Preserve Story Content**: Keep ALL story text verbatim. Do NOT summarize, paraphrase, or skip any narrative content.
2. **Remove Non-Story Elements**:
   - Page numbers (e.g., "Page 42", "- 78 -")
   - Headers/footers (book title, chapter title at top/bottom of pages)
   - Copyright notices, ISBN, publisher info
   - Table of contents entries mixed in body text
3. **Fix OCR Errors**:
   - Broken words: "th e" → "the", "wor ld" → "world"
   - Merged words: "thetree" → "the tree"
   - Character substitutions: "0" (zero) → "O", "1" (one) → "I" or "l"
   - Garbled punctuation: "dont" → "don't", "Its" → "It's"
4. **Clean Formatting**:
   - Remove excessive line breaks (keep paragraph structure)
   - Fix chapter headings (keep them, but clean format)
   - Preserve dialogue formatting
5. **Output**: Return ONLY the cleaned text. No explanations, no comments."""

        user_prompt = f"""Clean this text for audiobook narration:

{text[:20000]}

Return the cleaned text."""

        try:
            if self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=16000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                cleaned_text = response.content[0].text
            else:  # openai
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4096
                )
                cleaned_text = response.choices[0].message.content
            
            print(f"✅ Text cleaned: {len(text)} → {len(cleaned_text)} characters")
            return cleaned_text
            
        except Exception as e:
            print(f"⚠️  LLM cleanup failed: {e}. Using original text.")
            return text

    
    def split_into_chapters(self, text: str, max_chars: int = 30000) -> List[Tuple[str, str]]:
        """
        Split text into chapters based on common headings.
        Enforces a maximum character limit per chapter by splitting into parts if needed.
        
        Args:
            text: Full text
            max_chars: Maximum characters per chapter (default: 30000)
            
        Returns:
            List of (chapter_title, chapter_text) tuples
        """
        # Common chapter patterns
        # 1. "Chapter 1", "Chapter One", "CHAPTER 1"
        # 2. "Prologue", "Epilogue"
        # 3. "Part 1", "Book One"
        chapter_pattern = re.compile(
            r'\n\s*(?:Chapter|Part|Book)\s+(?:\d+|[IVXLCDM]+|[A-Za-z]+).*?\n|'
            r'\n\s*(?:Prologue|Epilogue|Introduction|Preface).*?\n',
            re.IGNORECASE
        )
        
        matches = list(chapter_pattern.finditer(text))
        
        raw_chapters = []
        
        if not matches:
            # No clear chapters found, treat as single block
            raw_chapters.append(("Full Text", text))
        else:
            # Handle text before first chapter (e.g. title page, dedication)
            if matches[0].start() > 0:
                preamble = text[:matches[0].start()].strip()
                if preamble:
                    raw_chapters.append(("Preamble", preamble))
            
            # Extract chapters
            for i, match in enumerate(matches):
                title = match.group().strip()
                start = match.end()
                
                if i < len(matches) - 1:
                    end = matches[i+1].start()
                else:
                    end = len(text)
                    
                content = text[start:end].strip()
                if content:
                    raw_chapters.append((title, content))
        
        # Post-process to enforce max_chars
        final_chapters = []
        for title, content in raw_chapters:
            if len(content) <= max_chars:
                final_chapters.append((title, content))
            else:
                # Split large chapter
                parts = self._split_large_content(content, max_chars)
                for i, part in enumerate(parts, 1):
                    final_chapters.append((f"{title} (Part {i})", part))
                    
        return final_chapters

    def _split_large_content(self, content: str, max_chars: int) -> List[str]:
        """Helper to split large content at paragraph boundaries."""
        parts = []
        while len(content) > max_chars:
            # Find split point (last double newline before max_chars)
            # Try to split at paragraph break
            split_idx = content.rfind('\n\n', 0, max_chars)
            
            if split_idx == -1:
                # Fallback to single newline
                split_idx = content.rfind('\n', 0, max_chars)
            
            if split_idx == -1:
                # Fallback to sentence end
                split_idx = content.rfind('. ', 0, max_chars)
                
            if split_idx == -1:
                # Hard fallback
                split_idx = max_chars
                
            parts.append(content[:split_idx].strip())
            content = content[split_idx:].strip()
            
        if content:
            parts.append(content)
        return parts

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
