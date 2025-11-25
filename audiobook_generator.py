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
    
    def __init__(self, device: str = None, use_llm_cleanup: bool = False):
        """
        Initialize the audiobook generator.
        
        Args:
            device: Device to use (cuda, mps, cpu). Auto-detected if None.
            use_llm_cleanup: If True, use SLM for intelligent text cleanup
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
        self.parser = DocumentParser(use_llm_cleanup=use_llm_cleanup)
        self.voice_manager = VoiceManager()
        
        # M4 Optimization: Determine optimal worker count
        # MPS is not thread-safe for parallel inference, so we force sequential
        if self.device == "mps":
            print("MPS detected: Forcing sequential processing for stability")
            self.num_workers = 1
        else:
            # Leave 2 cores for system/UI
            self.num_workers = max(1, min(multiprocessing.cpu_count() - 2, 8))
            print(f"Parallel processing enabled: {self.num_workers} workers")
    
    def _convert_to_wav(self, input_path: str) -> str:
        """Convert audio to WAV for stable loading."""
        try:
            import librosa
            import soundfile as sf
            
            path = Path(input_path)
            if path.suffix.lower() == '.wav':
                return input_path
                
            print(f"Converting {path.name} to WAV for stability...")
            y, sr = librosa.load(input_path, sr=None)
            
            # Save as temp wav
            temp_wav = path.with_suffix('.temp.wav')
            sf.write(str(temp_wav), y, sr)
            return str(temp_wav)
        except Exception as e:
            print(f"Warning: Failed to convert audio to WAV: {e}")
            return input_path

    def generate_audiobook(
        self,
        input_path: str,
        output_path: str,
        voice_name: Optional[str] = None,
        voice_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        detect_sfx: bool = False,
        # Advanced audio settings
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        min_p: float = 0.05,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate an audiobook from a document.
        
        Args:
            input_path: Path to input document (PDF, DOCX)
            output_path: Path to save output audio (WAV)
            voice_name: Name of saved voice to use
            voice_path: Path to specific voice file (overrides voice_name)
            progress_callback: Function to call with progress updates
            detect_sfx: If True, detect and save sound effect suggestions
            exaggeration: Exaggeration factor (0.5 = neutral)
            temperature: Sampling temperature (0.8 default)
            cfg_weight: Classifier-free guidance weight (0.5 default)
            min_p: Min-p sampling (0.05 default)
            top_p: Top-p sampling (1.0 default)
            repetition_penalty: Repetition penalty (1.2 default)
            
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
        
        # 1.5. Detect sound effects (optional)
        if detect_sfx:
            if progress_callback:
                progress_callback(0.08, "Detecting sound effects...")
            
            print("Analyzing text for sound effects...")
            from text_enhancer import TextEnhancer
            enhancer = TextEnhancer(device=self.device)
            sfx_suggestions = enhancer.detect_sound_effects(text)
            
            # Save suggestions to JSON
            import json
            sfx_path = Path("sfx_suggestions.json")
            with open(sfx_path, 'w') as f:
                json.dump(sfx_suggestions, f, indent=2)
            
            print(f"Saved {len(sfx_suggestions)} SFX suggestions to {sfx_path}")
        
        # 2. Prepare Voice
        ref_voice_path = None
        if voice_path:
            ref_voice_path = voice_path
        elif voice_name:
            ref_voice_path = self.voice_manager.get_voice_path(voice_name)
            if not ref_voice_path:
                # List available voices for debugging
                available = [v['name'] for v in self.voice_manager.list_voices()]
                raise ValueError(
                    f"Voice '{voice_name}' not found in voice library.\n"
                    f"Available voices: {', '.join(available) if available else '(none)'}\n"
                    f"Tip: Check spelling and capitalization, or refresh the voice list in the UI."
                )
            # Verify the file actually exists
            if not ref_voice_path.exists():
                raise FileNotFoundError(
                    f"Voice file missing: {ref_voice_path}\n"
                    f"The voice '{voice_name}' is registered but its audio file is gone.\n"
                    f"Try re-uploading the voice or deleting and re-adding it."
                )
        
        if ref_voice_path:
            # Convert to WAV for stability
            ref_voice_path = self._convert_to_wav(str(ref_voice_path))
            print(f"Using voice reference: {ref_voice_path}")
        else:
            print("Using default model voice")
            
        # 3. Generate Audio Chunks
        if progress_callback:
            progress_callback(0.1, f"Generating audio for {len(chunks)} chunks...")
            
        print(f"Generating audio with {self.num_workers} workers...")
        
        # Create temp directory for chunks
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        chunk_files = []
        
        import soundfile as sf
        import gc
        
        # Helper function for parallel processing
        def process_chunk(chunk_data):
            idx, text_chunk = chunk_data
            try:
                # Generate audio
                wav = self.model.generate(
                    text_chunk,
                    audio_prompt_path=str(ref_voice_path) if ref_voice_path else None,
                    exaggeration=exaggeration,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                wav_np = wav.squeeze(0).cpu().numpy()
                
                # Save to temp file immediately to free memory
                chunk_filename = temp_dir / f"chunk_{idx:05d}.wav"
                sf.write(str(chunk_filename), wav_np, self.model.sr)
                
                # Explicit cleanup
                del wav
                del wav_np
                if self.device == "mps":
                    torch.mps.empty_cache()
                
                return (idx, str(chunk_filename))
            except Exception as e:
                print(f"Error generating chunk {idx}: {e}")
                return (idx, None)

        # Prepare chunk data with indices
        chunk_data = list(enumerate(chunks))
        
        # Process in parallel (or sequential for MPS)
        # Note: Even with sequential, we use the executor for consistency
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_chunk, data) for data in chunk_data]
            
            # Monitor progress
            completed = 0
            for future in tqdm(futures, desc="Generating Audio"):
                idx, chunk_path = future.result()
                if chunk_path:
                    chunk_files.append((idx, chunk_path))
                
                completed += 1
                
                # Aggressive memory cleanup
                if completed % 5 == 0:
                    gc.collect()
                    if self.device == "mps":
                        torch.mps.empty_cache()
                
                if progress_callback:
                    progress = 0.1 + (0.8 * completed / len(chunks))
                    progress_callback(progress, f"Generating chunk {completed}/{len(chunks)}")
        
        # Sort results by index
        chunk_files.sort(key=lambda x: x[0])
        
        # 4. Combine Audio
        if progress_callback:
            progress_callback(0.9, "Combining audio segments...")
            
        print("Combining audio segments...")
        if not chunk_files:
            raise RuntimeError("No audio generated")
            
        # Combine from files
        audio_segments = [] # Renamed from full_audio_segments to avoid conflict with user's change
        silence = np.zeros(int(self.model.sr * 0.5))
        
        for _, fpath in chunk_files:
            data, _ = sf.read(fpath)
            audio_segments.append(data)
            audio_segments.append(silence)
            
        combined_audio = np.concatenate(audio_segments)
        
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Normalize
        max_val = np.abs(combined_audio).max()
        if max_val > 0:
            combined_audio = combined_audio / max_val * 0.95
            
        print(f"Normalized audio (gain: {0.95/max_val:.2f}x)")
        
        # Save raw narration first
        narration_path = output_path.replace(".wav", "_narration.wav")
        sf.write(narration_path, combined_audio, self.model.sr)
        
        # 6. Generate and Mix SFX (if enabled)
        final_output_path = output_path
        
        if detect_sfx and Path("sfx_suggestions.json").exists():
            if progress_callback:
                progress_callback(0.98, "Generating and mixing sound effects...")
            
            print("Mixing sound effects...")
            try:
                import json
                from sound_designer import SoundDesigner
                
                # Load suggestions
                with open("sfx_suggestions.json", 'r') as f:
                    sfx_suggestions = json.load(f)
                
                if sfx_suggestions:
                    designer = SoundDesigner()
                    
                    # Calculate total duration
                    duration = len(combined_audio) / self.model.sr
                    
                    # Map timestamps
                    mapped_sfx = designer.map_sfx_to_timestamps(sfx_suggestions, text, duration)
                    
                    # Mix
                    final_output_path = designer.mix_audio(
                        narration_path=narration_path,
                        sfx_list=mapped_sfx,
                        output_path=output_path,
                        sfx_volume=0.3
                    )
                    print(f"Mixed audio saved to {final_output_path}")
                else:
                    print("No SFX suggestions found to mix.")
                    # If no SFX, just rename narration to output
                    import shutil
                    shutil.copy(narration_path, output_path)
                    
            except Exception as e:
                print(f"Error mixing SFX: {e}")
                print("Falling back to narration only.")
                import shutil
                shutil.copy(narration_path, output_path)
        # 5. Save Output (if not already saved by SFX mixer)
        if not detect_sfx or not Path("sfx_suggestions.json").exists():
            if progress_callback:
                progress_callback(0.95, "Saving output file...")
                
            print(f"Saving to {output_path}")
            # Convert back to tensor for saving
            final_audio_tensor = torch.from_numpy(combined_audio).unsqueeze(0)
            torchaudio.save(output_path, final_audio_tensor, self.model.sr)
        
        total_time = time.time() - start_time
        print(f"Audiobook generated in {total_time:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, "Done!")
            
        return final_output_path


if __name__ == "__main__":
    # Test script
    generator = AudiobookGenerator()
    print("Audiobook Generator initialized successfully!")
