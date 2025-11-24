"""
Audiobook Generator

This module integrates document parsing, voice management, and TTS generation
to create full audiobooks.

Optimized for Apple Silicon (M4) with parallel processing.
"""

import os
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm import tqdm

from chatterbox.tts import ChatterboxTTS
from audiobook_utils import DocumentParser, estimate_processing_time
from voice_manager import VoiceManager


class AudiobookGenerator:
    """Generate audiobooks from documents using Chatterbox TTS."""
    
    def __init__(self, device: str = None):
        """
        Initialize the audiobook generator.
        
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
        print(f"Initializing AudiobookGenerator on {device}...")
        
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.parser = DocumentParser()
        self.voice_manager = VoiceManager()
        
        # M4 Optimization: Determine optimal worker count
        # Leave 2 cores for system/UI
        self.num_workers = max(1, min(multiprocessing.cpu_count() - 2, 8))
        print(f"Parallel processing enabled: {self.num_workers} workers")
    
    def generate_audiobook(
        self,
        input_path: str,
        output_path: str,
        voice_name: Optional[str] = None,
        voice_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Generate an audiobook from a document.
        
        Args:
            input_path: Path to input document (PDF, DOCX)
            output_path: Path to save output audio (WAV)
            voice_name: Name of saved voice to use
            voice_path: Path to specific voice file (overrides voice_name)
            progress_callback: Function to call with progress updates (0.0-1.0, status_msg)
            
        Returns:
            Path to generated audiobook file
        """
        start_time = time.time()
        
        # 1. Parse Document
        if progress_callback:
            progress_callback(0.05, "Parsing document...")
            
        print(f"Parsing document: {input_path}")
        text, metadata = self.parser.parse_document(input_path)
        chunks = self.parser.chunk_text(text)
        
        print(f"Document parsed: {metadata['page_count']} pages, {len(chunks)} chunks")
        
        # 2. Prepare Voice
        ref_voice_path = None
        if voice_path:
            ref_voice_path = voice_path
        elif voice_name:
            ref_voice_path = self.voice_manager.get_voice_path(voice_name)
            if not ref_voice_path:
                raise ValueError(f"Voice '{voice_name}' not found")
        
        if ref_voice_path:
            print(f"Using voice reference: {ref_voice_path}")
        else:
            print("Using default model voice")
            
        # 3. Generate Audio Chunks
        if progress_callback:
            progress_callback(0.1, f"Generating audio for {len(chunks)} chunks...")
            
        print(f"Generating audio with {self.num_workers} parallel workers...")
        
        # Helper function for parallel processing
        def process_chunk(chunk_data):
            idx, text_chunk = chunk_data
            try:
                # Generate audio
                wav = self.model.generate(
                    text_chunk,
                    audio_prompt_path=str(ref_voice_path) if ref_voice_path else None,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
                wav_np = wav.squeeze(0).cpu().numpy()
                return (idx, wav_np)
            except Exception as e:
                print(f"Error generating chunk {idx}: {e}")
                # Return silence on error to maintain sync
                silence = np.zeros(int(self.model.sr * 1.0))
                return (idx, silence)

        # Prepare chunk data with indices
        chunk_data = list(enumerate(chunks))
        results = []
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_chunk, data) for data in chunk_data]
            
            # Monitor progress
            completed = 0
            for future in tqdm(futures, desc="Generating Audio"):
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress = 0.1 + (0.8 * completed / len(chunks))
                    progress_callback(progress, f"Generating chunk {completed}/{len(chunks)}")
        
        # Sort results by index to ensure correct order
        results.sort(key=lambda x: x[0])
        
        # Extract audio segments and add silence
        audio_segments = []
        silence = np.zeros(int(self.model.sr * 0.5))  # 0.5s silence
        
        for _, wav_data in results:
            audio_segments.append(wav_data)
            audio_segments.append(silence)
        
        # 4. Combine Audio
        if progress_callback:
            progress_callback(0.9, "Combining audio segments...")
            
        print("Combining audio segments...")
        if not audio_segments:
            raise RuntimeError("No audio generated")
            
        full_audio = np.concatenate(audio_segments)
        
        # 5. Save Output
        if progress_callback:
            progress_callback(0.95, "Saving output file...")
            
        print(f"Saving to {output_path}")
        # Convert back to tensor for saving
        full_audio_tensor = torch.from_numpy(full_audio).unsqueeze(0)
        torchaudio.save(output_path, full_audio_tensor, self.model.sr)
        
        elapsed = time.time() - start_time
        print(f"Audiobook generated in {elapsed:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, "Done!")
            
        return output_path


if __name__ == "__main__":
    # Test script
    generator = AudiobookGenerator()
    print("Audiobook Generator initialized successfully!")
