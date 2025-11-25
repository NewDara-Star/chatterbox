"""
The Director - AI Creative Lead
This module interfaces with Large Language Models (Claude 3.7 Sonnet, GPT-4)
to analyze book chapters and generate detailed direction sheets for video production.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any
import anthropic

class Director:
    """
    The Director analyzes text and outputs a JSON Direction Sheet.
    It identifies scenes, visual prompts, sound effects, and mood.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the Director.
        
        Args:
            api_key: Anthropic API key. If None, looks for ANTHROPIC_API_KEY env var.
            model: Model to use (default: claude-3-7-sonnet-20250219)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            print("Warning: No Anthropic API key found. Director will not function.")
            
        self.model = model
        self.client = None
        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_chapter(self, chapter_text: str, chapter_title: str = "Chapter") -> Dict[str, Any]:
        """
        Analyze a full chapter and return a Direction Sheet.
        
        Args:
            chapter_text: The full text of the chapter.
            chapter_title: Title for context.
            
        Returns:
            JSON Dict containing scenes, mood, and cues.
        """
        if not self.client:
            raise ValueError("Director not initialized with API key.")

        system_prompt = """You are a visionary Film Director adapting a book into a cinematic video.
Your goal is to break down the text into distinct SCENES and provide detailed direction for visuals and audio.

Output must be valid JSON with this structure:
{
  "chapter_title": "...",
  "overall_mood": "...",
  "scenes": [
    {
      "id": 1,
      "text_segment": "The exact text from the book corresponding to this scene...",
      "visual_prompt": "Detailed image generation prompt (style, lighting, camera angle, subject)...",
      "audio_cues": [
        {"timestamp": "start", "sfx": "door_creak", "description": "old rusty hinge squeak"},
        {"timestamp": "end", "sfx": "thunder_rumble", "description": "distant rolling thunder"}
      ],
      "mood": "Suspenseful",
      "pacing": "Slow"
    }
  ]
}

Rules:
1. Cover the ENTIRE text. Every sentence must belong to a scene.
2. Visual prompts should be descriptive and artistic (e.g. "Cinematic shot, 35mm, low angle...").
3. Audio cues should be specific.
4. JSON must be strictly valid."""

        user_prompt = f"Analyze this chapter:\n\nTitle: {chapter_title}\n\nText:\n{chapter_text}"

        try:
            print(f"Director analyzing {chapter_title}...")
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract JSON from response
            response_text = message.content[0].text
            return self._parse_json_response(response_text)
            
        except Exception as e:
            print(f"Director error: {e}")
            raise

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
