"""
Book Analyzer - Full Context Character Extraction

Analyzes entire books to extract character descriptions, world-building,
and validates text extraction quality. Creates a "Character Bible" for
consistent visual generation across all chapters.

Optimized for Apple Silicon (M4) with LLM-based analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BookAnalyzer:
    """
    Analyzes full books to extract character descriptions and validate text quality.
    """
    
    def __init__(self, provider: str = "anthropic"):
        """
        Initialize the book analyzer.
        
        Args:
            provider: LLM provider ("anthropic" or "openai")
        
        Raises:
            ValueError: If provider is invalid or API key is missing
        """
        self.provider = provider.lower() if provider else None
        
        if self.provider not in ["anthropic", "openai"]:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                "Must be 'anthropic' or 'openai'."
            )
        
        #Initialize API client
        if self.provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
        elif self.provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
    
    def analyze_full_book(self, book_text: str, book_title: str = "Untitled") -> Tuple[Dict, Dict]:
        """
        Analyze entire book to extract character bible and validate text quality.
        
        Args:
            book_text: Full book text
            book_title: Title of the book
        
        Returns:
            Tuple of (character_bible_dict, extraction_report_dict)
        """
        print(f"Analyzing book: {book_title}")
        print(f"Book length: {len(book_text)} characters, {len(book_text.split())} words")
        
        # Step 1: Extract Character Bible
        character_bible = self._extract_character_bible(book_text, book_title)
        
        # Step 2: Validate text extraction (check for missing/garbled sections)
        extraction_report = self._validate_text_quality(book_text)
        
        return character_bible, extraction_report
    
    def _extract_character_bible(self, book_text: str, book_title: str) -> Dict:
        """
        Extract character descriptions, world-building, and story summary.
        
        Args:
            book_text: Full book text
            book_title: Title of the book
        
        Returns:
            Character Bible dictionary
        """
        system_prompt = """You are a Character Design Expert analyzing a book to create a comprehensive "Character Bible" for AI image generation.

Your task: Extract EVERY character's physical appearance, clothing, and personality traits with extreme precision.

Output MUST be valid JSON with this EXACT structure:
{
  "book_title": "...",
  "characters": [
    {
      "name": "Full Character Name",
      "importance": "Main/Secondary/Minor",
      "physical_traits": {
        "race_ethnicity": "Specific description (e.g., East Asian, Black, Caucasian, Mixed)",
        "age_range": "Specific years or description",
        "height": "tall/average/short or specific measurement",
        "build": "slender/athletic/stocky/muscular/etc",
        "hair": {
          "color": "#HEXCODE (estimate from description)",
          "style": "Detailed description",
          "texture": "straight/wavy/curly/coily"
        },
        "eyes": {
          "color": "#HEXCODE or descriptive name",
          "shape": "almond/round/hooded/etc"
        },
        "skin_tone": "#HEXCODE (estimate)",
        "distinguishing_marks": ["List all scars, tattoos, birthmarks, freckles, etc."]
      },
      "clothing_palette": {
        "dominant_colors": ["#HEXCODE1", "#HEXCODE2"],
        "signature_items": ["Specific clothing items they wear"],
        "style_keywords": ["modern", "vintage", "warrior", "elegant", etc.]
      },
      "personality_traits": "Brief character summary"
    }
  ],
  "world_style": {
    "setting": "Time period and location",
    "genre": "Fantasy/Sci-Fi/Historical/etc",
    "color_palette": ["#HEX1", "#HEX2", "#HEX3"],
    "lighting_mood": "Overall lighting atmosphere",
    "architectural_style": "Building and environment descriptions"
  }
}

CRITICAL RULES:
1. Extract EVERY named character (even minor ones)
2. Use hex color codes for all colors (estimate from text descriptions)
3. Be specific about race/ethnicity (don't say "unclear" - infer from context if needed)
4. Include micro-details (scars, jewelry, specific clothing items)
5. If a trait isn't explicitly stated, make an educated inference based on context
"""
        
        user_prompt = f"""Analyze this book and create a Character Bible:

Title: {book_title}

Text (first 50,000 characters):
{book_text[:50000]}

Generate the complete Character Bible JSON."""
        
        # Call LLM
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            result_text = response.content[0].text
        else:  # openai
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=16000
            )
            result_text = response.choices[0].message.content
        
        # Parse JSON
        # Extract JSON from markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        character_bible = json.loads(result_text)
        
        print(f"✅ Extracted {len(character_bible.get('characters', []))} characters")
        
        return character_bible
    
    def _validate_text_quality(self, book_text: str) -> Dict:
        """
        Validate text extraction quality (check for missing sections, garbled text).
        
        Args:
            book_text: Full book text
        
        Returns:
            Extraction report dictionary
        """
        # Simple heuristic checks (non-LLM)
        issues = []
        
        # Check 1: Excessive consecutive capital letters (likely OCR errors)
        import re
        caps_blocks = re.findall(r'[A-Z]{20,}', book_text)
        if caps_blocks:
            issues.append({
                "type": "excessive_capitals",
                "severity": "medium",
                "description": f"Found {len(caps_blocks)} blocks of 20+ consecutive capital letters (possible OCR errors)"
            })
        
        # Check 2: Very short paragraphs (might indicate missing content)
        paragraphs = book_text.split('\n\n')
        very_short = [p for p in paragraphs if len(p.split()) < 3 and len(p.strip()) > 0]
        if len(very_short) > len(paragraphs) * 0.3:  # More than 30% very short
            issues.append({
                "type": "fragmented_paragraphs",
                "severity": "high",
                "description": f"{len(very_short)} out of {len(paragraphs)} paragraphs are very short (< 3 words)"
            })
        
        # Check 3: Malformed sentences (no ending punctuation)
        sentences = re.split(r'[.!?]\s+', book_text)
        no_punctuation = sum(1 for s in sentences if s and not s.strip()[-1:] in '.!?')
        if no_punctuation > len(sentences) * 0.2:
            issues.append({
                "type": "missing_punctuation",
                "severity": "medium",
                "description": f"{no_punctuation} sentences lack proper ending punctuation"
            })
        
        # Calculate quality score
        if not issues:
            quality_score = 100
        elif any(i['severity'] == 'high' for i in issues):
            quality_score = 60
        elif any(i['severity'] == 'medium' for i in issues):
            quality_score = 80
        else:
            quality_score = 90
        
        return {
            "quality_score": quality_score,
            "total_characters": len(book_text),
            "total_words": len(book_text.split()),
            "total_paragraphs": len(paragraphs),
            "issues": issues,
            "recommendation": "Good quality" if quality_score >= 80 else "Review flagged sections"
        }


if __name__ == "__main__":
    # Test with sample text
    sample_book = """
    Yangchen walked through the bustling market of Bin-Er, her monk's robes billowing
    in the coastal wind. Her gray eyes scanned the crowd with practiced ease. She was
    young for an Avatar, barely twenty, with pale skin that burned easily in the sun.
    Her dark brown hair was kept short in the traditional Air Nomad style.
    
    Kavik stood apart from the crowd, his Water Tribe clothing marking him as an outsider.
    He was tall and lean, with bronze skin and black hair tied back. His blue eyes
    followed Yangchen's movements with quiet intensity.
    """
    
    analyzer = BookAnalyzer(provider="anthropic")
    bible, report = analyzer.analyze_full_book(sample_book, "Dawn of Yangchen (Sample)")
    
    print("\n=== CHARACTER BIBLE ===")
    print(json.dumps(bible, indent=2))
    
    print("\n=== EXTRACTION REPORT ===")
    print(json.dumps(report, indent=2))
