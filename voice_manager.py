"""
Voice Manager - Named Voice Reference System

This module manages voice references for audiobook generation.
Allows users to save, load, and manage multiple voice recordings with custom names.

Optimized for Apple Silicon (M4) with efficient file I/O.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class VoiceManager:
    """Manage voice references for TTS generation."""
    
    def __init__(self, voices_dir: str = "voices"):
        """
        Initialize the voice manager.
        
        Args:
            voices_dir: Directory to store voice references (default: "voices")
        """
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.voices_dir / "voices.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load voice metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save voice metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_voice(self, audio_path: str, voice_name: str, description: str = "") -> bool:
        """
        Save a voice reference with a custom name.
        
        Args:
            audio_path: Path to the audio file
            voice_name: Custom name for this voice
            description: Optional description of the voice
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If voice name already exists
        """
        source_path = Path(audio_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if voice_name in self.metadata:
            raise ValueError(
                f"Voice name '{voice_name}' already exists. "
                "Use a different name or delete the existing voice first."
            )
        
        # Create safe filename from voice name
        safe_name = self._sanitize_filename(voice_name)
        extension = source_path.suffix
        dest_filename = f"{safe_name}{extension}"
        dest_path = self.voices_dir / dest_filename
        
        # Copy audio file
        shutil.copy2(source_path, dest_path)
        
        # Save metadata
        self.metadata[voice_name] = {
            'filename': dest_filename,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'file_size': dest_path.stat().st_size,
            'format': extension[1:]  # Remove the dot
        }
        self._save_metadata()
        
        return True
    
    def get_voice_path(self, voice_name: str) -> Optional[Path]:
        """
        Get the file path for a saved voice.
        
        Args:
            voice_name: Name of the voice
            
        Returns:
            Path to the voice file, or None if not found
        """
        if voice_name not in self.metadata:
            return None
        
        filename = self.metadata[voice_name]['filename']
        return self.voices_dir / filename
    
    def list_voices(self) -> List[Dict]:
        """
        List all saved voices with their metadata.
        
        Returns:
            List of dictionaries containing voice information
        """
        voices = []
        for name, data in self.metadata.items():
            voices.append({
                'name': name,
                'description': data.get('description', ''),
                'created_at': data.get('created_at', ''),
                'file_size': data.get('file_size', 0),
                'format': data.get('format', 'unknown')
            })
        
        # Sort by creation date (newest first)
        voices.sort(key=lambda x: x['created_at'], reverse=True)
        return voices
    
    def delete_voice(self, voice_name: str) -> bool:
        """
        Delete a saved voice.
        
        Args:
            voice_name: Name of the voice to delete
            
        Returns:
            True if successful, False if voice not found
        """
        if voice_name not in self.metadata:
            return False
        
        # Delete audio file
        filename = self.metadata[voice_name]['filename']
        file_path = self.voices_dir / filename
        if file_path.exists():
            file_path.unlink()
        
        # Remove from metadata
        del self.metadata[voice_name]
        self._save_metadata()
        
        return True
    
    def rename_voice(self, old_name: str, new_name: str) -> bool:
        """
        Rename a saved voice.
        
        Args:
            old_name: Current name of the voice
            new_name: New name for the voice
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If old_name doesn't exist or new_name already exists
        """
        if old_name not in self.metadata:
            raise ValueError(f"Voice '{old_name}' not found")
        
        if new_name in self.metadata:
            raise ValueError(f"Voice name '{new_name}' already exists")
        
        # Update metadata
        self.metadata[new_name] = self.metadata.pop(old_name)
        self._save_metadata()
        
        return True
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict]:
        """
        Get detailed information about a voice.
        
        Args:
            voice_name: Name of the voice
            
        Returns:
            Dictionary with voice information, or None if not found
        """
        if voice_name not in self.metadata:
            return None
        
        data = self.metadata[voice_name].copy()
        data['name'] = voice_name
        data['path'] = str(self.get_voice_path(voice_name))
        
        return data
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """
        Create a safe filename from a voice name.
        
        Args:
            name: Voice name
            
        Returns:
            Sanitized filename
        """
        # Replace spaces and special characters
        safe = name.replace(' ', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c in ('_', '-'))
        return safe.lower()


if __name__ == "__main__":
    # Simple test
    manager = VoiceManager()
    print("Voice Manager initialized successfully!")
    print(f"Voices directory: {manager.voices_dir}")
    print(f"Saved voices: {len(manager.list_voices())}")
