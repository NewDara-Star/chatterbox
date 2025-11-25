"""
Sound Designer - AI-Powered Sound Effects Generation

This module uses Meta's AudioGen (via AudioCraft) to generate high-quality
sound effects from text prompts. It runs AudioGen in an isolated virtual
environment (venv_sfx) to avoid dependency conflicts.

Falls back to procedural generation if the isolated environment is missing
or generation fails.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional
from scipy.signal import resample
import subprocess
import os
import tempfile

class SoundDesigner:
    """Mix sound effects into audiobooks."""
    
    def __init__(self):
        """Initialize the sound designer."""
        self.sample_rate = 24000  # Standard sample rate for audiobook
        
        # Path to isolated python
        self.venv_python = Path("venv_sfx/bin/python").resolve()
        self.cli_script = Path("generate_sfx_cli.py").resolve()
        
        # Built-in simple SFX (fallback)
        self.sfx_generators = {
            'rain': self._generate_rain,
            'thunder': self._generate_thunder,
            'door': self._generate_door_creak,
            'footsteps': self._generate_footsteps,
            'wind': self._generate_wind,
        }
    
    def _generate_rain(self, duration: float = 5.0) -> np.ndarray:
        """Generate rain sound using white noise."""
        samples = int(duration * self.sample_rate)
        # White noise filtered to sound like rain
        rain = np.random.normal(0, 0.1, samples)
        # Apply envelope for natural fade
        envelope = np.linspace(0.5, 1.0, samples)
        return rain * envelope
    
    def _generate_thunder(self, duration: float = 3.0) -> np.ndarray:
        """Generate thunder sound using low-frequency rumble."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        # Low frequency rumble
        thunder = np.sin(2 * np.pi * 40 * t) * np.exp(-t/2)
        thunder += np.random.normal(0, 0.05, samples)
        return thunder * 0.5
    
    def _generate_door_creak(self, duration: float = 2.0) -> np.ndarray:
        """Generate door creaking sound."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        # Creaking sound (frequency sweep)
        freq_sweep = 200 + 100 * np.sin(2 * np.pi * 2 * t)
        creak = np.sin(2 * np.pi * freq_sweep * t)
        envelope = np.exp(-t/duration * 2)
        return creak * envelope * 0.3
    
    def _generate_footsteps(self, duration: float = 3.0) -> np.ndarray:
        """Generate footsteps sound."""
        samples = int(duration * self.sample_rate)
        # Create individual footstep sounds
        step_duration = 0.3  # seconds per step
        num_steps = int(duration / step_duration)
        footsteps = np.zeros(samples)
        
        for i in range(num_steps):
            step_start = int(i * step_duration * self.sample_rate)
            step_samples = int(0.2 * self.sample_rate)
            t = np.linspace(0, 0.2, step_samples)
            # Thud sound
            step = np.sin(2 * np.pi * 100 * t) * np.exp(-t * 20)
            step += np.random.normal(0, 0.02, step_samples)
            footsteps[step_start:step_start+step_samples] = step * 0.4
        
        return footsteps
    
    def _generate_wind(self, duration: float = 5.0) -> np.ndarray:
        """Generate wind sound."""
        samples = int(duration * self.sample_rate)
        # Low-frequency noise for wind
        wind = np.random.normal(0, 0.08, samples)
        # Apply slow modulation
        t = np.linspace(0, duration, samples)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        return wind * modulation
    
    def generate_sfx(
        self,
        prompt: str,
        duration: float = 5.0
    ) -> np.ndarray:
        """
        Generate a sound effect from a text prompt.
        
        Args:
            prompt: Text description of the sound effect
            duration: Target duration in seconds
            
        Returns:
            Audio waveform as numpy array
        """
        # Try AI generation via isolated environment
        if self.venv_python.exists() and self.cli_script.exists():
            print(f"Generating AI SFX: '{prompt}' (via AudioGen)")
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
                
            try:
                # Call the wrapper script
                cmd = [
                    str(Path("run_sfx.sh").resolve()),
                    "--prompt", prompt,
                    "--duration", str(duration),
                    "--output", temp_wav
                ]
                
                # Prepare environment (clear PYTHONPATH to avoid conflicts)
                env = os.environ.copy()
                # Clear variables that might confuse the subprocess python
                for var in ["PYTHONPATH", "PYTHONHOME", "__PYVENV_LAUNCHER__"]:
                    if var in env:
                        del env[var]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    check=True,
                    env=env
                )
                
                # Read the generated audio
                if os.path.exists(temp_wav):
                    audio, sr = sf.read(temp_wav)
                    
                    # Resample if needed (AudioGen is usually 16k or 32k)
                    if sr != self.sample_rate:
                        target_samples = int(len(audio) * self.sample_rate / sr)
                        audio = resample(audio, target_samples)
                        
                    # Cleanup
                    os.unlink(temp_wav)
                    return audio
                    
            except subprocess.CalledProcessError as e:
                print(f"AI Generation failed: {e.stderr}")
                print("Switching to procedural fallback.")
            except Exception as e:
                print(f"Error reading AI audio: {e}")
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
        else:
            print("AudioGen environment not found. Using procedural fallback.")
        
        # Fallback to procedural
        print(f"Generating Procedural SFX: '{prompt}'")
        
        # Match keywords to generators
        prompt_lower = prompt.lower()
        
        if 'rain' in prompt_lower:
            return self._generate_rain(duration)
        elif 'thunder' in prompt_lower or 'rumbl' in prompt_lower:
            return self._generate_thunder(duration)
        elif 'door' in prompt_lower or 'creak' in prompt_lower:
            return self._generate_door_creak(duration)
        elif 'footstep' in prompt_lower or 'step' in prompt_lower or 'walk' in prompt_lower:
            return self._generate_footsteps(duration)
        elif 'wind' in prompt_lower:
            return self._generate_wind(duration)
        else:
            # Default: gentle ambient noise
            print(f"  No specific generator for '{prompt}', using ambient noise")
            samples = int(duration * self.sample_rate)
            return np.random.normal(0, 0.05, samples)
    
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
