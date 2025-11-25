import argparse
import sys
import os
import torch
import torchaudio
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def generate_sfx(prompt, duration, output_path):
    print(f"Loading AudioGen model for prompt: '{prompt}'...")
    try:
        from audiocraft.models import AudioGen
        from audiocraft.data.audio import audio_write
        
        # Load model
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        model.set_generation_params(duration=duration)
        
        # Generate
        print("Generating audio...")
        wav = model.generate([prompt])  # [1, 1, samples]
        
        # Save
        print(f"Saving to {output_path}...")
        # AudioGen output is [batch, channels, time]
        # audio_write expects [channels, time] and adds extension
        # But we want specific path.
        
        audio_tensor = wav[0].cpu()
        sample_rate = model.sample_rate
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torchaudio.save(output_path, audio_tensor, sample_rate)
        print("Success!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFX using AudioGen")
    parser.add_argument("--prompt", required=True, help="Text description of sound")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    
    args = parser.parse_args()
    
    generate_sfx(args.prompt, args.duration, args.output)
