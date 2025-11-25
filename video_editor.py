import os
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from moviepy import (
    ImageClip, AudioFileClip, concatenate_videoclips, 
    CompositeVideoClip, vfx, CompositeAudioClip, ColorClip
)




from audiobook_generator import AudiobookGenerator


class EasingFunctions:
    """
    Easing functions for smooth video transitions.
    Based on research: cubic ease-in-out produces most natural-looking transitions.
    """
    
    @staticmethod
    def ease_in_out_cubic(t):
        """
        Cubic ease-in-out (smooth start and end).
        Perfect for crossfades and opacity changes.
        """
        if t < 0.5:
            return 4 * t * t * t
        else:
            p = 2 * t - 2
            return 0.5 * p * p * p + 1
    
    @staticmethod
    def ease_in_quad(t):
        """Quadratic ease-in (slow start, accelerating)."""
        return t * t
    
    @staticmethod
    def ease_out_quad(t):
        """Quadratic ease-out (fast start, decelerating)."""
        return t * (2 - t)
    
    @staticmethod
    def linear(t):
        """Linear (no easing)."""
        return t


class VideoEditor:
    """
    Assembles the final video from the Direction Sheet (JSON),
    Images (uploaded by user), and Audio (generated per scene).
    """
    
    def __init__(self, output_dir: str = "output_video"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = AudiobookGenerator() # For TTS
        
    def generate_scene_audio(self, scene: Dict, voice_name: str = None) -> str:
        """
        Generate TTS audio for a single scene.
        Returns path to the generated WAV file.
        """
        scene_id = scene.get("id", "unknown")
        text = scene.get("text_segment", "")
        if not text:
            return None
            
        print(f"  - Generating audio for Scene {scene_id}...")
        
        # Use the generator to create a short clip
        # We use a temporary path for the scene audio
        filename = f"scene_{scene_id:03d}_audio.wav"
        output_path = self.output_dir / filename
        
        # We need to bypass the full book generation logic and just do one chunk
        # But AudiobookGenerator is designed for files. 
        # Let's use the model directly or a helper.
        # Actually, AudiobookGenerator.model.generate_audio is what we want, 
        # but it's not exposed easily.
        # Let's use a temporary text file trick to reuse existing logic 
        # or expose a method in AudiobookGenerator.
        
        # Better: Add `generate_clip` to AudiobookGenerator.
        # For now, I'll assume I can add it.
        
        # Workaround: Save text to temp file and run generate_audiobook 
        # (A bit heavy but reliable for now)
        temp_txt = self.output_dir / f"scene_{scene_id}.txt"
        with open(temp_txt, 'w') as f:
            f.write(text)
            
        self.generator.generate_audiobook(
            input_path=str(temp_txt),
            output_path=str(output_path),
            voice_name=voice_name,
            detect_sfx=False # We handle SFX separately in the video editor
        )
        
        # Cleanup temp text
        if temp_txt.exists():
            temp_txt.unlink()
            
        return str(output_path)

    def create_ken_burns_clip(self, image_path: str, duration: float) -> ImageClip:
        """
        Create a video clip from an image with a Ken Burns (pan/zoom) effect.
        """
        if not os.path.exists(image_path):
            # Fallback to black screen or placeholder
            print(f"Warning: Image not found {image_path}")
            return None
            
        clip = ImageClip(image_path).with_duration(duration)
        
        # Simple Zoom In effect
        # Resize from 100% to 110% over the duration
        w, h = clip.size
        
        def zoom(t):
            scale = 1 + 0.1 * (t / duration)
            return scale
            
        clip = clip.resized(zoom)
        
        # Center crop to keep aspect ratio (assuming 16:9 output)
        # For now, just resizing. A real Ken Burns needs complex cropping.
        # Let's keep it simple: Zoom In Center.
        
        return clip.with_position("center")

    def assemble_video(self, json_path: str, voice_name: str = None, output_filename: str = "final_movie.mp4"):
        """
        Main driver: Loads JSON, generates assets, stitches video.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        scenes = data.get("scenes", [])
        video_clips = []
        
        print(f"Assembling video for {len(scenes)} scenes...")
        
        for scene in scenes:
            scene_id = scene.get("id")
            image_path = scene.get("image_path") # Populated by Scene Manager
            
            # 1. Generate Audio
            audio_path = self.generate_scene_audio(scene, voice_name)
            if not audio_path:
                continue
                
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration + 0.5 # Add 0.5s pause
            
            # 2. Create Video Clip
            if image_path and os.path.exists(image_path):
                video_clip = self.create_ken_burns_clip(image_path, duration)
            else:
                # Black screen with text? Or just skip image?
                # Let's make a black clip
                # Black screen with text? Or just skip image?
                # Let's make a black clip
                video_clip = ColorClip(size=(1920, 1080), color=(0,0,0), duration=duration)
            
            # 3. Sync Audio
            video_clip = video_clip.with_audio(audio_clip)
            
            # 4. Add Fade In/Out
            video_clip = video_clip.with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.5)])
            
            video_clips.append(video_clip)
            
        # Concatenate
        final_video = concatenate_videoclips(video_clips, method="compose")
        
        # Write file
        out_path = str(self.output_dir / output_filename)
        final_video.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac')
        
        return out_path
