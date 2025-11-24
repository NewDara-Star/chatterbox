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
        prompt = f"""You are a text cleaning assistant for audiobook production.

Task: Clean the following text extracted from a PDF. Remove:
- Headers and footers (page numbers, chapter titles at top/bottom)
- Copyright notices
- Image captions and figure labels
- OCR errors and artifacts

Keep all narrative content intact. Output ONLY the cleaned text, no explanations.

Text:
{text}

Cleaned text:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=len(text) + 100,  # Allow some expansion
                temperature=0.3,  # Low temp for deterministic cleaning
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the cleaned text (after the prompt)
        if "Cleaned text:" in result:
            cleaned = result.split("Cleaned text:")[-1].strip()
        else:
            cleaned = result.strip()
        
        return cleaned
    
    def detect_sound_effects(self, text: str, max_effects: int = 10) -> List[Dict[str, str]]:
        """
        Analyze text and suggest sound effects.
        
        Args:
            text: Text to analyze
            max_effects: Maximum number of effects to suggest
            
        Returns:
            List of dicts with keys: 'timestamp', 'effect', 'description'
        """
        self._load_model()
        
        # Simpler prompt with explicit format
        prompt = f"""Analyze this text and suggest sound effects for an audiobook.

Text: "{text[:800]}"

List 3-5 sound effects that would enhance this scene. For each effect, provide:
1. When it should play (start/middle/end)
2. A brief name (e.g., "door creaking")
3. A detailed description for audio generation

Format your response EXACTLY like this example:
start | door creaking | old wooden door slowly opening with rusty hinges
middle | rain sounds | gentle rain pattering on window glass
end | footsteps | slow footsteps on wooden floorboards

Your response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response after "Your response:"
        if "Your response:" in result:
            response_text = result.split("Your response:")[-1].strip()
        else:
            response_text = result.strip()
        
        # Parse line-by-line format
        effects = []
        lines = response_text.split('\n')
        
        for line in lines[:max_effects]:
            line = line.strip()
            if not line or '|' not in line:
                continue
                
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                effects.append({
                    'timestamp': parts[0],
                    'effect': parts[1],
                    'description': parts[2]
                })
        
        return effects


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
