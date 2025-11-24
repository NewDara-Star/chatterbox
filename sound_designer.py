"""
Sound Designer - AI-Powered Sound Effects Generation and Mixing

This module uses Bark to generate sound effects from text prompts
and mixes them into audiobooks at contextually appropriate timestamps.

Optimized for Apple Silicon (M4) with MPS acceleration.
"""

import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.signal import resample


class SoundDesigner:
    """Generate and mix sound effects for audiobooks."""
    
    def __init__(self, device: str = None):
        """
        Initialize the sound designer.
        
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
        self._loaded = False
        self.sample_rate = SAMPLE_RATE  # Bark's sample rate (24kHz)
    
    def _load_models(self):
        """Lazy-load Bark models."""
        if self._loaded:
            return
        
        print(f"Loading Bark models...")
        
        # Preload Bark models (downloads on first use)
        preload_models()
        
        self._loaded = True
        print("Bark models loaded successfully!")
    
    def generate_sfx(
        self,
        prompt: str,
        duration: float = 5.0
    ) -> np.ndarray:
        """
        Generate a sound effect from a text prompt.
        
        Args:
            prompt: Text description of the sound effect
            duration: Target duration in seconds (Bark will generate ~5-10s)
            
        Returns:
            Audio waveform as numpy array
        """
        self._load_models()
        
        print(f"Generating SFX: '{prompt}'")
        
        # Bark uses special prompts for non-speech sounds
        # Add sound effect markers
        bark_prompt = f"[sound effect: {prompt}]"
        
        # Generate audio
        audio = generate_audio(bark_prompt)
        
        # Trim or pad to target duration
        target_samples = int(duration * self.sample_rate)
        if len(audio) > target_samples:
            # Trim
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            # Pad with silence
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        return audio
    
    def map_sfx_to_timestamps(
        self,
        sfx_suggestions: List[Dict],
        text: str,
        audio_duration: float
    ) -> List[Dict]:
        """
        Map relative SFX timestamps to absolute positions based on text content.
        
        Args:
            sfx_suggestions: List of SFX with relative timestamps
            text: Full text of the audiobook
            audio_duration: Total duration of the narration audio (seconds)
            
        Returns:
            SFX list with absolute timestamps in seconds
        """
        mapped_sfx = []
        
        for sfx in sfx_suggestions:
            # Extract keywords from the effect description
            effect_keywords = sfx['effect'].lower().split()
            
            # Find where these keywords appear in the text
            text_lower = text.lower()
            best_position = 0.0
            
            # Try to find the effect keywords in the text
            for keyword in effect_keywords:
                if keyword in text_lower:
                    # Find the position of this keyword
                    idx = text_lower.find(keyword)
                    position_ratio = idx / len(text_lower)
                    best_position = position_ratio
                    break
            
            # If no keywords found, use relative mapping
            if best_position == 0.0:
                relative = sfx.get('timestamp', 'middle')
                if relative == 'start':
                    best_position = 0.05  # 5% in
                elif relative == 'end':
                    best_position = 0.90  # 90% in
                else:  # middle
                    best_position = 0.50  # 50% in
            
            # Convert to absolute timestamp
            timestamp_seconds = best_position * audio_duration
            
            mapped_sfx.append({
                'timestamp': timestamp_seconds,
                'effect': sfx['effect'],
                'description': sfx['description']
            })
        
        return mapped_sfx
    
    def mix_audio(
        self,
        narration_path: str,
        sfx_list: List[Dict],
        output_path: str,
        sfx_volume: float = 0.3
    ) -> str:
        """
        Mix sound effects into the narration audio.
        
        Args:
            narration_path: Path to the narration audio file
            sfx_list: List of SFX with absolute timestamps
            output_path: Path to save the mixed audio
            sfx_volume: Volume multiplier for SFX (0.0-1.0, default: 0.3)
            
        Returns:
            Path to the mixed audio file
        """
        print(f"Mixing {len(sfx_list)} sound effects into audiobook...")
        
        # Load narration
        narration, sr = sf.read(narration_path)
        
        # Ensure mono
        if len(narration.shape) > 1:
            narration = narration.mean(axis=1)
        
        # Generate and mix each SFX
        for i, sfx in enumerate(sfx_list):
            print(f"  [{i+1}/{len(sfx_list)}] {sfx['effect']} @ {sfx['timestamp']:.1f}s")
            
            # Generate the SFX
            sfx_audio = self.generate_sfx(
                sfx['description'],
                duration=5.0  # Default 5-second SFX
            )
            
            # Resample SFX to match narration sample rate if needed
            if self.sample_rate != sr:
                target_samples = int(len(sfx_audio) * sr / self.sample_rate)
                sfx_audio = resample(sfx_audio, target_samples)
            
            # Calculate insertion point
            insert_idx = int(sfx['timestamp'] * sr)
            
            # Mix the SFX (add to narration with volume adjustment)
            end_idx = min(insert_idx + len(sfx_audio), len(narration))
            sfx_length = end_idx - insert_idx
            
            # Apply volume and mix
            narration[insert_idx:end_idx] += sfx_audio[:sfx_length] * sfx_volume
        
        # Normalize to prevent clipping
        max_val = np.abs(narration).max()
        if max_val > 1.0:
            narration = narration / max_val * 0.95
        
        # Save mixed audio
        sf.write(output_path, narration, sr)
        print(f"Mixed audiobook saved to {output_path}")
        
        return output_path


if __name__ == "__main__":
    # Simple test
    designer = SoundDesigner()
    
    # Test SFX generation
    print("Testing SFX generation...")
    sfx = designer.generate_sfx("door creaking open", duration=3.0)
    sf.write("test_sfx.wav", sfx, designer.sample_rate)
    print(f"Generated test SFX: test_sfx.wav ({len(sfx)/designer.sample_rate:.1f}s)")
    
    # Test timestamp mapping
    print("\nTesting timestamp mapping...")
    suggestions = [
        {"timestamp": "start", "effect": "door creaking", "description": "old wooden door opening"},
        {"timestamp": "middle", "effect": "rain", "description": "rain on window"},
    ]
    text = "The door creaked. Rain fell outside."
    mapped = designer.map_sfx_to_timestamps(suggestions, text, 10.0)
    print(f"Mapped SFX: {json.dumps(mapped, indent=2)}")
