"""
The Director - AI Creative Lead
This module interfaces with Large Language Models (Claude 3.7 Sonnet, GPT-4)
to analyze book chapters and generate detailed direction sheets for video production.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import os
import json
import time
from typing import Dict, List, Optional, Any
import anthropic
from dotenv import load_dotenv

load_dotenv()

try:
    import openai
except ImportError:
    openai = None

class Director:
    """
    The Director analyzes text and outputs a JSON Direction Sheet.
    It identifies scenes, visual prompts, sound effects, and mood.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = None, model: Optional[str] = None):
        """
        Initialize the Director.
        
        Args:
            api_key: API key for the chosen provider.
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
            self.model = model or "claude-3-7-sonnet-20250219"
            self.api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.provider == "openai":
            self.model = model or "gpt-4o"
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if self.api_key and openai:
                self.client = openai.OpenAI(api_key=self.api_key)
        else:
            print(f"Warning: Unknown provider {provider}")

        if not self.api_key:
            print(f"Warning: No API key found for {self.provider}. Director will not function.")

    def analyze_chapter(self, chapter_text: str, chapter_title: str = "Chapter") -> Dict[str, Any]:
        """
        Analyze a full chapter and return a Direction Sheet.
        """
        if not self.client:
            raise ValueError(f"Director not initialized (Provider: {self.provider}). Check API key.")

        system_prompt = """You are a visionary Film Director adapting a book into a cinematic video.
Your goal is to break down the text into distinct SCENES and provide detailed direction for visuals and audio.

CRITICAL RULES:
1. **VERBATIM TEXT**: The `text_segment` MUST be the EXACT text from the book. Do NOT summarize, paraphrase, or skip a single word. The concatenation of all `text_segment`s must equal the original chapter text.
2. **RICH VISUALS**: `visual_prompt` must be highly detailed, suitable for a top-tier AI image generator (Midjourney/DALL-E 3). Include:
   - Subject & Action
   - Lighting (e.g., "volumetric lighting", "chiaroscuro", "golden hour")
   - Camera (e.g., "wide shot", "extreme close-up", "35mm lens", "f/1.8 aperture")
   - Style (e.g., "cinematic", "hyper-realistic", "8k resolution", "unreal engine 5 render")
   - Mood/Atmosphere (e.g., "dust particles in air", "foggy", "neon-drenched")
3. **AUDIO SYNC**: `audio_cues` must be specific and timed (start/mid/end).

Output must be valid JSON with this structure:
{
  "chapter_title": "...",
  "overall_mood": "...",
  "scenes": [
{
  "id": 1,
  "text_segment": "The exact text from the book...",
  "visual_prompt": "Cinematic wide shot of a cyberpunk street, neon rain reflecting on wet pavement, 35mm lens, high contrast...",
  "audio_cues": [
    {"timestamp": "start", "sfx": "door_creak", "description": "old rusty hinge squeak"},
    {"timestamp": "end", "sfx": "thunder_rumble", "description": "distant rolling thunder"}
  ],
  "mood": "Suspenseful",
  "pacing": "Slow"
}
  ]
}"""

        user_prompt = f"Analyze this chapter:\n\nTitle: {chapter_title}\n\nText:\n{chapter_text}"

        try:
            print(f"Director ({self.provider}) analyzing {chapter_title}...")
            
            if self.provider == "anthropic":
                return self._analyze_with_anthropic(system_prompt, user_prompt)
            elif self.provider == "openai":
                return self._analyze_with_openai(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
        except Exception as e:
            print(f"Director error: {e}")
            raise

    def _analyze_with_anthropic(self, system_prompt, user_prompt):
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return self._parse_json_response(message.content[0].text)

    def _analyze_with_openai(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return self._parse_json_response(response.choices[0].message.content)

    def _parse_json_response(self, text: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        try:
            # Find JSON start/end
            start = text.find('{')
            end = text.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = text[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Simple retry or fallback logic could go here
            print("Failed to parse JSON. Raw output:")
            print(text[:500] + "...")
            raise ValueError("Invalid JSON output from Director")

if __name__ == "__main__":
    # Test stub
    print("Director module ready.")
