"""
The Director - AI Creative Lead (Enhanced)

This module interfaces with Large Language Models (Claude 3.7 Sonnet, GPT-4)
to analyze books and generate ultra-detailed direction sheets.

Two modes:
1. Book Analysis: Extract Character Bible from full book
2. Chapter Direction: Generate scene breakdowns with ultra-detailed image prompts

Optimized for Apple Silicon (M4) with LLM-based analysis.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from book_analyzer import BookAnalyzer
from text_splitter import SemanticTextSplitter

load_dotenv()

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None


class Director:
    """
    The Director has two operational modes:
    - Mode 1 (Book Analysis): Extracts Character Bible
    - Mode 2 (Chapter Direction): Generates ultra-detailed scene breakdowns
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = None, model: Optional[str] = None):
        """
        Initialize the Director.
        
        Args:
            api_key: API key for the chosen provider
            provider: "anthropic" or "openai" (REQUIRED)
            model: Model name (optional, defaults to provider best)
        """
        if not provider:
            raise ValueError("Provider must be specified ('anthropic' or 'openai')")
            
        self.provider = provider.lower()
        if self.provider not in ["anthropic", "openai"]:
            raise ValueError(f"Unknown provider: {provider}")
            
        self.api_key = api_key
        self.client = None
        
        if self.provider == "anthropic":
            self.model = model or "claude-3-5-sonnet-20241022"
            self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if self.api_key:
                if not anthropic:
                    raise ImportError("anthropic package not installed")
                self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "openai":
            self.model = model or "gpt-4-turbo-preview"
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                if not openai:
                    raise ImportError("openai package not installed")
                self.client = openai.OpenAI(api_key=self.api_key)

        if not self.api_key:
            raise ValueError(f"No API key found for {self.provider}")
    
    # =========================================================================
    # MODE 1: BOOK ANALYSIS
    # =========================================================================
    
    def analyze_full_book(self, book_text: str, book_title: str = "Untitled") -> Dict[str, Any]:
        """
        Mode 1: Analyze entire book to extract Character Bible.
        
        Args:
            book_text: Full book text
            book_title: Title of the book
        
        Returns:
            Dictionary containing:
            - character_bible: Character descriptions, world-building
            - extraction_report: Text quality validation
        """
        analyzer = BookAnalyzer(provider=self.provider)
        character_bible, extraction_report = analyzer.analyze_full_book(book_text, book_title)
        
        return {
            "character_bible": character_bible,
            "extraction_report": extraction_report
        }
    
    # =========================================================================
    # MODE 2: CHAPTER DIRECTION
    # =========================================================================
    
    def analyze_chapter(
        self, 
        chapter_text: str, 
        chapter_title: str = "Chapter",
        character_bible: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Mode 2: Generate ultra-detailed scene breakdowns for a chapter.
        
        Args:
            chapter_text: Chapter text
            chapter_title: Chapter title
            character_bible: Character Bible dictionary (from Mode 1)
        
        Returns:
            Direction Sheet with ultra-detailed JSON prompts
        """
        if not self.client:
            raise ValueError(f"Director not initialized. Check API key.")
        
        # Step 1: Split text programmatically (verbatim, no AI)
        splitter = SemanticTextSplitter(target_duration=35)
        text_segments = splitter.split_chapter(chapter_text)
        
        print(f"Split chapter into {len(text_segments)} scenes")
        
        # Step 2: Generate ultra-detailed prompts for each scene
        scenes = []
        for i, text_segment in enumerate(text_segments):
            scene = self._generate_scene_prompt(
                scene_id=i + 1,
                text_segment=text_segment,
                character_bible=character_bible
            )
            scenes.append(scene)
        
        return {
            "chapter_title": chapter_title,
            "overall_mood": self._infer_overall_mood(text_segments),
            "scenes": scenes
        }
    
    def _generate_scene_prompt(
        self,
        scene_id: int,
        text_segment: str,
        character_bible: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate ultra-detailed image prompt for a single scene.
        
        Args:
            scene_id: Scene number
            text_segment: Exact text for this scene (verbatim)
            character_bible: Character Bible (if available)
        
        Returns:
            Scene dictionary with ultra-detailed JSON prompt
        """
        # Build character context
        char_context = ""
        if character_bible and "characters" in character_bible:
            char_context = "\\n\\nCHARACTER BIBLE (for consistency):\\n"
            for char in character_bible["characters"][:10]:  # Top 10 chars
                char_context += f"- {char['name']}: {json.dumps(char.get('physical_traits', {}), indent=2)}\\n"
        
        system_prompt = f"""You are a Film Director working with a pre-established Character Bible.

Your task: Generate an ULTRA-DETAILED visual prompt for this scene using this JSON schema:

{{
  "text_segment": "EXACT text from scene (DO NOT MODIFY)",
  "visual_prompt": {{
    "global_context": {{
      "scene_description": "Comprehensive paragraph",
      "time_of_day": "Dawn/Midday/Dusk/Night/specific time",
      "weather_atmosphere": "Foggy/Clear/Rainy/Stormy/Serene",
      "lighting": {{
        "source": "Natural sunlight/Artificial/Mixed/Candlelight",
        "direction": "Top-down/Backlit/Side-lit/Frontal",
        "quality": "Hard/Soft/Diffused/Dramatic",
        "color_temp": "Warm (3000K)/Cool (6500K)/Neutral"
      }}
    }},
    "color_palette": {{
      "dominant_hex": ["#RRGGBB", "#RRGGBB"],
      "accent_colors": ["Neon pink", "Electric blue"],
      "contrast_level": "High/Medium/Low"
    }},
    "composition": {{
      "camera_angle": "Eye-level/High-angle/Low-angle/Dutch-tilt/Bird's-eye/Worm's-eye",
      "framing": "Extreme-close-up/Close-up/Medium-shot/Wide-shot/Extreme-wide-shot",
      "depth_of_field": "Shallow (f/1.4-f/2.8)/Medium (f/4-f/5.6)/Deep (f/11-f/22)",
      "focal_point": "Character's eyes/The artifact/The doorway",
      "aspect_ratio": "16:9/2.39:1/1:1/9:16",
      "lens": {{
        "focal_length": "14mm/24mm/50mm/85mm/200mm",
        "type": "Prime/Zoom/Fisheye/Tilt-shift",
        "distortion": "None/Barrel/Pincushion"
      }},
      "camera_settings": {{
        "iso": "100/800/3200",
        "shutter_speed": "1/1000s/1/60s/1s",
        "white_balance": "Daylight (5500K)/Tungsten (3200K)/Fluorescent (4000K)"
      }}
    }},
    "objects": [
      {{
        "id": "char_001",
        "label": "CHARACTER_NAME",
        "category": "Person",
        "character_bible_ref": "char_bible_id_001",
        "location": "Center-frame/Left-third/Background-right",
        "prominence": "Foreground/Midground/Background",
        "visual_attributes": {{
          "pose": "Standing/Sitting/Running",
          "expression": "Determined/Fearful/Angry",
          "clothing": "EXACT from Character Bible",
          "color": "Hex codes from Bible",
          "texture": "Leather (worn)/Silk/Metal",
          "lighting_on_object": "Rim-lit/Shadow-side"
        }},
        "micro_details": ["At least 3-5 specific details"]
      }}
    ],
    "semantic_relationships": ["Object A casts shadow on B"]
  }},
  "audio_cues": [
    {{"timestamp": "start", "sfx": "door_creak", "description": "detail"}}
  ],
  "mood": "Suspenseful/Calm/Tense/etc",
  "pacing": "Slow/Medium/Fast"
}}

CRITICAL RULES:
1. **Character Consistency**: If a character appears, MUST reference Character Bible
2. **Micro-details**: 3-5 specific details per character/object
3. **Hex Colors**: Use hex values for all colors
4. **Camera Specs**: ALWAYS include aspect ratio, lens, ISO, shutter speed
5. **Verbatim Text**: Copy `text_segment` EXACTLY{char_context}
"""
        
        user_prompt = f"Generate ultra-detailed prompt for Scene {scene_id}:\\n\\nText:\\n{text_segment}"
        
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0.7,
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
                    temperature=0.7,
                    max_tokens=4096
                )
                result_text = response.choices[0].message.content
            
            # Parse JSON
            scene_data = self._parse_json_response(result_text)
            scene_data["id"] = scene_id
            scene_data["text_segment"] = text_segment  # Ensure verbatim
            
            return scene_data
            
        except Exception as e:
            print(f"Error generating scene {scene_id}: {e}")
            # Fallback: minimal scene data
            return {
                "id": scene_id,
                "text_segment": text_segment,
                "visual_prompt": "Scene visual description unavailable",
                "audio_cues": [],
                "mood": "Unknown",
                "pacing": "Medium"
            }
    
    def _infer_overall_mood(self, text_segments: List[str]) -> str:
        """Infer overall chapter mood from first segment."""
        if not text_segments:
            return "Unknown"
        
        # Simple keyword-based inference
        first_text = text_segments[0].lower()
        if any(word in first_text for word in ["dark", "shadow", "fear", "tense"]):
            return "Tense"
        elif any(word in first_text for word in ["bright", "warm", "happy", "joy"]):
            return "Uplifting"
        else:
            return "Neutral"
    
    def _parse_json_response(self, text: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        try:
            # Extract JSON from markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            # Find JSON start/end
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = text[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print("Raw output:", text[:500])
            raise ValueError("Invalid JSON output from Director")


if __name__ == "__main__":
    # Test Mode 2
    director = Director(provider="anthropic")
    
    sample_chapter = """
    Yangchen walked through the market, her gray eyes scanning the crowd.
    She noticed Kavik standing by the fish stall, his blue eyes watching her.
    """
    
    sample_bible = {
        "characters": [
            {
                "name": "Yangchen",
                "physical_traits": {
                    "race_ethnicity": "Air Nomad",
                    "eyes": {"color": "#808080", "shape": "round"}
                }
            }
        ]
    }
    
    result = director.analyze_chapter(sample_chapter, "Test Chapter", sample_bible)
    print(json.dumps(result, indent=2))
