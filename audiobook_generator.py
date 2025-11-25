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
import gc
import shutil
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
            # Prevent memory fragmentation limits
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            self.num_workers = 1
        else:
            # Leave 2 cores for system/UI
            self.num_workers = max(1, min(multiprocessing.cpu_count() - 2, 8))
            print(f"Parallel processing enabled: {self.num_workers} workers")
    
    def prepare_chapters(self, input_path: str, output_dir: str) -> List[str]:
        """
        Parse document, clean text, and split into chapter files for review.
        
        Args:
            input_path: Path to input document
            output_dir: Directory to save chapter text files
            
        Returns:
            List of absolute paths to generated text files
        """
        print(f"Preparing chapters from: {input_path}")
        text, metadata = self.parser.parse_document(input_path)
        
        # Split into chapters (enforcing 30k limit)
        chapters = self.parser.split_into_chapters(text)
        print(f"Split into {len(chapters)} chapters/parts.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, (title, content) in enumerate(chapters):
            # Sanitize title for filename
            safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()
            safe_title = safe_title.replace(' ', '_')
            filename = f"chapter_{i+1:02d}_{safe_title}.txt"
            file_path = output_path / filename
            
            with open(file_path, 'w') as f:
                f.write(f"Title: {title}\n\n")
                f.write(content)
                
            saved_files.append(str(file_path.absolute()))
            
        return saved_files

    def analyze_chapter_with_director(self, chapter_file_path: str, api_key: Optional[str] = None) -> str:
        """
        Analyze a chapter text file using the Director (Claude).
        
        Args:
            chapter_file_path: Path to the chapter text file
            api_key: Anthropic API key (optional)
            
        Returns:
            Path to the generated JSON direction sheet
        """
        print(f"Directing chapter: {chapter_file_path}")
        
        # Read chapter text
        path = Path(chapter_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Chapter file not found: {chapter_file_path}")
            
        with open(path, 'r') as f:
            content = f.read()
            
        # Extract title if present (first line)
        lines = content.split('\n')
        title = "Unknown Chapter"
        if lines and lines[0].startswith("Title:"):
            title = lines[0].replace("Title:", "").strip()
            
        # Initialize Director
        from director import Director
        director = Director(api_key=api_key)
        
        # Analyze
        direction_sheet = director.analyze_chapter(content, chapter_title=title)
        
        # Save JSON
        output_path = path.with_suffix('.json')
        import json
        with open(output_path, 'w') as f:
            json.dump(direction_sheet, f, indent=2)
            
        print(f"Direction sheet saved to: {output_path}")
        return str(output_path)

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

    def _generate_chapter(
        self,
        chapter_idx: int,
        chapter_title: str,
        text: str,
        output_dir: Path,
        ref_voice_path: Optional[str],
        detect_sfx: bool,
        # Gen params
        exaggeration: float,
        temperature: float,
        cfg_weight: float,
        min_p: float,
        top_p: float,
        repetition_penalty: float,
        progress_callback: Optional[Callable] = None
    ) -> Optional[str]:
        """Generate audio for a single chapter."""
        print(f"Processing Chapter {chapter_idx}: {chapter_title}")
        
        # 1. Chunk text
        chunks = self.parser.chunk_text(text)
        print(f"  - {len(chunks)} chunks")
        
        # 2. SFX Detection (Tier 1/2)
        sfx_suggestions = []
        if detect_sfx:
            # Optimization: Keyword filtering to reduce SLM load
            sfx_keywords = [
                "sound", "noise", "voice", "cry", "scream", "shout", "whisper",
                "bang", "crash", "thud", "click", "creak", "snap", "knock",
                "rain", "wind", "thunder", "storm", "water", "river", "sea",
                "bird", "dog", "wolf", "horse", "animal",
                "step", "walk", "run", "footstep", "door", "window",
                "gun", "shot", "explosion", "fire", "burn", "ring", "bell"
            ]
            
            # Fast check (case-insensitive)
            text_lower = text.lower()
            has_keywords = any(k in text_lower for k in sfx_keywords)
            
            if not has_keywords:
                print("  - No SFX keywords found, skipping expensive analysis.")
            else:
                try:
                    print("  - Analyzing for SFX...")
                    from text_enhancer import TextEnhancer
                    enhancer = TextEnhancer(device=self.device)
                    # Analyze first 2000 chars for mood (Sparse Analysis)
                    scene_data = enhancer.analyze_scene(text[:2000])
                    
                    # Use legacy detection for now, but scoped to chapter
                    sfx_suggestions = enhancer.detect_sound_effects(text)
                    
                    # Save chapter SFX
                    sfx_path = output_dir / f"chapter_{chapter_idx}_sfx.json"
                    import json
                    with open(sfx_path, 'w') as f:
                        json.dump(sfx_suggestions, f, indent=2)
                except Exception as e:
                    print(f"  - Warning: SFX detection failed: {e}")

        # 3. Generate Audio
        temp_dir = output_dir / f"temp_chunks_{chapter_idx}"
        temp_dir.mkdir(exist_ok=True)
        chunk_files = []
        
        import soundfile as sf
        
        # Helper for parallel processing
        def process_chunk(chunk_data):
            idx, text_chunk = chunk_data
            try:
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
                
                chunk_filename = temp_dir / f"chunk_{idx:05d}.wav"
                sf.write(str(chunk_filename), wav_np, self.model.sr)
                
                del wav
                del wav_np
                gc.collect()
                if self.device == "mps":
                    torch.mps.empty_cache()
                
                return (idx, str(chunk_filename))
            except Exception as e:
                print(f"Error generating chunk {idx}: {e}")
                return (idx, None)

        # Execute generation
        chunk_data = list(enumerate(chunks))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_chunk, data) for data in chunk_data]
            
            completed = 0
            for future in tqdm(futures, desc=f"Chapter {chapter_idx}", leave=False):
                idx, chunk_path = future.result()
                if chunk_path:
                    chunk_files.append((idx, chunk_path))
                
                completed += 1
                if completed % 5 == 0:
                    gc.collect()
                    if self.device == "mps":
                        torch.mps.empty_cache()
                
                if progress_callback:
                    # Map chapter progress to overall progress (simplified)
                    # Ideally we'd know total chapters, but local progress is fine
                    pass

        chunk_files.sort(key=lambda x: x[0])
        
        if not chunk_files:
            return None
            
        # 4. Combine Chapter Audio
        audio_segments = []
        silence = np.zeros(int(self.model.sr * 0.5))
        
        for _, fpath in chunk_files:
            data, _ = sf.read(fpath)
            audio_segments.append(data)
            audio_segments.append(silence)
            
        combined_audio = np.concatenate(audio_segments)
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Normalize
        max_val = np.abs(combined_audio).max()
        if max_val > 0:
            combined_audio = combined_audio / max_val * 0.95
            
        # Save Chapter Narration
        chapter_filename = output_dir / f"chapter_{chapter_idx:02d}.wav"
        sf.write(str(chapter_filename), combined_audio, self.model.sr)
        
        # 5. Mix SFX (if enabled)
        if detect_sfx and sfx_suggestions:
            try:
                print(f"  - Mixing SFX for Chapter {chapter_idx}...")
                from sound_designer import SoundDesigner
                designer = SoundDesigner()
                
                duration = len(combined_audio) / self.model.sr
                mapped_sfx = designer.map_sfx_to_timestamps(sfx_suggestions, text, duration)
                
                mixed_path = output_dir / f"chapter_{chapter_idx:02d}_mixed.wav"
                final_path = designer.mix_audio(
                    narration_path=str(chapter_filename),
                    sfx_list=mapped_sfx,
                    output_path=str(mixed_path),
                    sfx_volume=0.3
                )
                return str(final_path)
            except Exception as e:
                print(f"  - Error mixing SFX: {e}")
                return str(chapter_filename)
        
        return str(chapter_filename)

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
        Generate an audiobook from a document (Chapter-Based).
        
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
        
        # Split into chapters
        chapters = self.parser.split_into_chapters(text)
        print(f"Document parsed: {metadata['page_count']} pages, {len(chapters)} chapters")
        
        # 2. Prepare Voice
        ref_voice_path = None
        if voice_path:
            ref_voice_path = voice_path
        elif voice_name:
            ref_voice_path = self.voice_manager.get_voice_path(voice_name)
            if not ref_voice_path:
                available = [v['name'] for v in self.voice_manager.list_voices()]
                raise ValueError(f"Voice '{voice_name}' not found.")
            if not ref_voice_path.exists():
                raise FileNotFoundError(f"Voice file missing: {ref_voice_path}")
        
        if ref_voice_path:
            ref_voice_path = self._convert_to_wav(str(ref_voice_path))
            print(f"Using voice reference: {ref_voice_path}")
        else:
            print("Using default model voice")
            
        # 3. Process Chapters
        output_dir = Path(output_path).parent
        temp_chapter_dir = output_dir / "temp_chapters"
        temp_chapter_dir.mkdir(exist_ok=True)
        
        chapter_files = []
        
        for i, (title, content) in enumerate(chapters):
            idx = i + 1
            if progress_callback:
                progress = 0.1 + (0.8 * i / len(chapters))
                progress_callback(progress, f"Processing Chapter {idx}/{len(chapters)}: {title}")
            
            # Full Memory Reset between chapters
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
            
            chapter_path = self._generate_chapter(
                chapter_idx=idx,
                chapter_title=title,
                text=content,
                output_dir=temp_chapter_dir,
                ref_voice_path=ref_voice_path,
                detect_sfx=detect_sfx,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                progress_callback=progress_callback
            )
            
            if chapter_path:
                chapter_files.append(chapter_path)
        
        # 4. Combine All Chapters
        if progress_callback:
            progress_callback(0.95, "Finalizing audiobook...")
            
        print("Combining chapters...")
        if not chapter_files:
            raise RuntimeError("No audio generated")
            
        import soundfile as sf
        
        # Use a stream-like approach to avoid loading everything into RAM?
        # soundfile doesn't support append easily.
        # But we can read-write in blocks if needed.
        # For now, let's assume chapters fit in RAM (they are smaller than full book).
        # Actually, let's do it properly: read each file and write to output.
        
        # We need to know the samplerate and channels from the first file
        data, sr = sf.read(chapter_files[0])
        channels = data.shape[1] if len(data.shape) > 1 else 1
        
        # Create output file
        with sf.SoundFile(output_path, 'w', samplerate=sr, channels=channels) as f:
            for fpath in chapter_files:
                data, _ = sf.read(fpath)
                f.write(data)
                # Add small silence between chapters?
                silence = np.zeros((int(sr * 1.0),) if channels == 1 else (int(sr * 1.0), channels))
                f.write(silence)
        
        # Cleanup
        shutil.rmtree(temp_chapter_dir, ignore_errors=True)
        
        total_time = time.time() - start_time
        print(f"Audiobook generated in {total_time:.1f}s")
        
        if progress_callback:
            progress_callback(1.0, "Done!")
            
        return output_path


if __name__ == "__main__":
    # Test script
    generator = AudiobookGenerator()
    print("Audiobook Generator initialized successfully!")
