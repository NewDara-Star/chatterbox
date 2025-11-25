"""
Text Enhancer - SLM-Powered Text Processing

This module uses Llama 3.2 1B to:
1. Clean and enhance text extracted from PDFs (remove headers/footers intelligently)
2. Detect sound effect opportunities for audiobook production

Optimized for Apple Silicon (M4) with MPS acceleration.
"""

import json
import re
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextEnhancer:
    """SLM-powered text enhancement for audiobook generation."""
    
    def __init__(self, device: str = None):
        """
        Initialize the text enhancer with Llama 3.2 1B.
        
        Args:
            device: Device to use (cuda, mps, cpu). Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _load_model(self):
        """Lazy-load the model (only when needed)."""
        if self._loaded:
            return
            
        print(f"Loading Qwen2.5-1.5B on {self.device}...")
        
        # Using Qwen2.5-1.5B-Instruct (ungated, high quality, fast)
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        self._loaded = True
        print("Model loaded successfully!")
    
    def clean_text_with_llm(self, text: str, max_length: int = 2000) -> str:
        """
        Use LLM to intelligently clean text.
        
        This goes beyond regex to:
        - Remove context-aware headers/footers
        - Fix OCR errors
        - Improve readability
        
        Args:
            text: Raw text to clean
            max_length: Max characters to process (longer texts are chunked)
            
        Returns:
            Cleaned text
        """
        self._load_model()
        
        # For very long text, process in chunks
        if len(text) > max_length:
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            cleaned_chunks = [self._clean_chunk(chunk) for chunk in chunks]
            return '\n\n'.join(cleaned_chunks)
        
        return self._clean_chunk(text)
    
    def _clean_chunk(self, text: str) -> str:
        """Clean a single chunk of text."""
        prompt = f"""You are a strict text cleaning tool.

Task: Remove headers, footers, page numbers, and artifacts from the text below.
Constraint: Output ONLY the cleaned text. Do NOT add any introductory or concluding remarks.

Example Input:
Page 12
The night was dark.
Chapter 4
The wind howled.

Example Output:
The night was dark.
The wind howled.

Input Text:
{text}

Cleaned Text:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(text) + 100,
                temperature=0.1,  # Lower temp for more deterministic behavior
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the cleaned text (after the prompt)
        # Use the exact string from the prompt end
        split_marker = "Cleaned Text:"
        if split_marker in result:
            cleaned = result.split(split_marker)[-1].strip()
        else:
            # Fallback: try to find where the prompt ends if exact match fails
            # (Unlikely with deterministic generation but good safety)
            cleaned = result.replace(prompt, "").strip()
        
        return cleaned
    
    def analyze_scene(self, text: str) -> Dict:
        """
        Analyze text for mood, ambience, and sound effects.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing:
            - mood: str
            - intensity: int (1-10)
            - ambience: List[str]
            - sfx: List[Dict]
        """
        self._load_model()
        
        prompt = f"""You are an expert sound engineer for audiobooks.

Task: Analyze the text below for mood, ambience, and sound effects.

Output Format (JSON):
{{
  "mood": "Dominant mood (e.g. Suspenseful, Joyful)",
  "intensity": 1-10,
  "ambience": ["continuous background sound 1", "continuous background sound 2"],
  "sfx": [
    {{"timestamp": "start/middle/end", "name": "short name", "description": "detailed audio prompt"}}
  ]
}}

Example (for a sunny park scene):
{{
  "mood": "Joyful",
  "intensity": 3,
  "ambience": ["birds chirping", "distant laughter"],
  "sfx": [
    {{"timestamp": "middle", "name": "bicycle bell", "description": "cheerful bicycle bell ringing twice"}}
  ]
}}

Now analyze THIS text (it is NOT a sunny park):
Text: "{text[:1000]}"

Your JSON analysis:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Strip prompt
        if "Your JSON analysis:" in result:
            result = result.split("Your JSON analysis:")[-1]
        
        # Extract JSON with brace counting
        try:
            start = result.find('{')
            if start != -1:
                brace_count = 0
                json_str = ""
                found_end = False
                
                for i, char in enumerate(result[start:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    
                    if brace_count == 0:
                        json_str = result[start:start+i+1]
                        found_end = True
                        break
                
                if found_end:
                    data = json.loads(json_str)
                    return data
        except Exception as e:
            print(f"Error parsing JSON from LLM: {e}")
            print(f"Raw LLM Output: {result}")
            # Fallback
            return {
                "mood": "Neutral",
                "intensity": 5,
                "ambience": [],
                "sfx": []
            }
        
        return {
            "mood": "Neutral",
            "intensity": 5,
            "ambience": [],
            "sfx": []
        }

    def detect_sound_effects(self, text: str, max_effects: int = 10) -> List[Dict[str, str]]:
        """Legacy wrapper for backward compatibility."""
        scene_data = self.analyze_scene(text)
        effects = scene_data.get("sfx", [])
        
        # Normalize format if needed
        normalized = []
        for effect in effects[:max_effects]:
            normalized.append({
                "timestamp": effect.get("timestamp", "middle"),
                "effect": effect.get("name", "unknown"),
                "description": effect.get("description", "")
            })
            
        return normalized


if __name__ == "__main__":
    # Simple test
    enhancer = TextEnhancer()
    
    # Test text cleaning
    sample_text = """
    Page 5
    
    The old wooden door creaked as Sarah pushed it open. Rain pattered against the window.
    
    Figure 1: Illustration of the door
    
    © 2024 Publisher Inc.
    
    Page 6
    """
    
    print("Testing text cleaning...")
    cleaned = enhancer.clean_text_with_llm(sample_text)
    print(f"Cleaned: {cleaned}")
    
    # Test SFX detection
    print("\nTesting SFX detection...")
    effects = enhancer.detect_sound_effects(sample_text)
    print(f"Suggested effects: {json.dumps(effects, indent=2)}")
