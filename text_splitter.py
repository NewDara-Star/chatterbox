"""
Semantic Text Splitter

Programmatically splits chapter text into scenes for verbatim TTS narration.
Uses sentence-level chunking to preserve exact text without AI hallucination.

Optimized for Apple Silicon (M4) with efficient text processing.
"""

import re
from typing import List, Tuple
from pathlib import Path


class SemanticTextSplitter:
    """
    Split text into semantic chunks for TTS narration.
    Ensures verbatim text preservation (no AI summarization).
    """
    
    def __init__(self, target_duration: int = 35, words_per_minute: int = 150):
        """
        Initialize the text splitter.
        
        Args:
            target_duration: Target duration per chunk in seconds (default 35)
            words_per_minute: Average reading speed (default 150 WPM)
        """
        self.target_duration = target_duration
        self.words_per_minute = words_per_minute
        # Calculate target word count per chunk
        self.target_words = int(words_per_minute * target_duration / 60)
    
    def split_chapter(self, chapter_text: str) -> List[str]:
        """
        Split chapter into ~35-second chunks (verbatim).
        
        Args:
            chapter_text: Full chapter text
        
        Returns:
            List of text segments (exact text, no modifications)
        """
        # Split into sentences
        sentences = self._split_sentences(chapter_text)
        
        # Group into chunks
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = len(sentence.split())
            
            # Check if adding this sentence would exceed target (with 20% tolerance)
            if current_word_count + words > self.target_words * 1.2 and current_chunk:
                # Finalize current chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = words
            else:
                current_chunk.append(sentence)
                current_word_count += words
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses regex to split on sentence boundaries (.!?) while preserving
        abbreviations and decimal numbers.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on sentence-ending punctuation
        # This regex handles:
        # - Period, exclamation, question mark followed by space/newline
        # - Handles quotes after punctuation
        # - Preserves abbreviations (Dr., Mr., etc.) and numbers (3.14)
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def split_chapter_with_meta(self, chapter_text: str) -> List[Tuple[str, dict]]:
        """
        Split chapter and return chunks with metadata.
        
        Args:
            chapter_text: Full chapter text
        
        Returns:
            List of tuples: (text_segment, metadata_dict)
            metadata includes: word_count, estimated_duration, chunk_index
        """
        chunks = self.split_chapter(chapter_text)
        
        result = []
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            estimated_duration = (word_count / self.words_per_minute) * 60
            
            metadata = {
                "chunk_index": i + 1,
                "word_count": word_count,
                "estimated_duration": round(estimated_duration, 1),
                "char_count": len(chunk)
            }
            
            result.append((chunk, metadata))
        
        return result


if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    The Avatar's journey had never been easy. She walked through the crowded market,
    her eyes scanning the faces around her. Each person carried their own burdens,
    their own stories. But today, she was looking for someone specific.
    
    "Yangchen!" A voice called out from behind. She turned to see Kavik running
    towards her, his expression worried. "We need to talk. It's about the spirits."
    
    She knew this day would come. The balance between worlds was fragile, and 
    maintaining it required constant vigilance. "Let's go somewhere more private,"
    she said, leading him away from the crowds.
    """
    
    splitter = SemanticTextSplitter(target_duration=30)
    chunks = splitter.split_chapter(sample_text)
    
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        words = len(chunk.split())
        print(f"\n--- Chunk {i} ({words} words) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
