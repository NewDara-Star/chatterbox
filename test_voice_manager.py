"""
Test script for voice manager.

Tests saving, listing, retrieving, and deleting voice references.
"""

import os
import shutil
from pathlib import Path
from voice_manager import VoiceManager
import wave
import struct

def create_dummy_wav(filename="dummy.wav", duration=1.0):
    """Create a dummy WAV file for testing."""
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Write silence
        data = struct.pack('<h', 0) * num_samples
        wav_file.writeframes(data)
    
    print(f"✓ Created dummy WAV: {filename}")

def test_voice_manager():
    """Test the voice manager functionality."""
    print("\n" + "="*60)
    print("VOICE MANAGER TEST")
    print("="*60)
    
    # Setup test directory
    test_dir = "test_voices"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    manager = VoiceManager(voices_dir=test_dir)
    print(f"\n✓ Initialized VoiceManager in '{test_dir}'")
    
    # Create dummy audio
    create_dummy_wav("test_audio_1.wav")
    create_dummy_wav("test_audio_2.wav")
    
    try:
        # Test 1: Save voices
        print("\n💾 Testing voice saving...")
        manager.save_voice("test_audio_1.wav", "Narrator", "Main narration voice")
        print("✓ Saved 'Narrator'")
        
        manager.save_voice("test_audio_2.wav", "Character 1", "Voice for main character")
        print("✓ Saved 'Character 1'")
        
        # Test 2: List voices
        print("\n📋 Testing voice listing...")
        voices = manager.list_voices()
        print(f"Found {len(voices)} voices:")
        for v in voices:
            print(f"  - {v['name']}: {v['description']}")
        
        if len(voices) != 2:
            print("✗ Failed to list all voices")
        else:
            print("✓ Listed voices correctly")
            
        # Test 3: Get voice path
        print("\n🔍 Testing get voice path...")
        path = manager.get_voice_path("Narrator")
        if path and path.exists():
            print(f"✓ Found path for 'Narrator': {path}")
        else:
            print("✗ Failed to get path for 'Narrator'")
            
        # Test 4: Rename voice
        print("\n✏️  Testing voice renaming...")
        manager.rename_voice("Character 1", "Hero")
        print("✓ Renamed 'Character 1' to 'Hero'")
        
        voices = manager.list_voices()
        names = [v['name'] for v in voices]
        if "Hero" in names and "Character 1" not in names:
            print("✓ Rename verified in list")
        else:
            print("✗ Rename verification failed")
            
        # Test 5: Delete voice
        print("\n🗑️  Testing voice deletion...")
        manager.delete_voice("Hero")
        print("✓ Deleted 'Hero'")
        
        voices = manager.list_voices()
        if len(voices) == 1 and voices[0]['name'] == "Narrator":
            print("✓ Deletion verified")
        else:
            print("✗ Deletion verification failed")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n" + "="*60)
        print("CLEANUP")
        print("="*60)
        
        if os.path.exists("test_audio_1.wav"):
            os.remove("test_audio_1.wav")
        if os.path.exists("test_audio_2.wav"):
            os.remove("test_audio_2.wav")
        
        # Clean up test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✓ Removed test directory '{test_dir}'")
            
        print("\n✅ TEST COMPLETE")

if __name__ == "__main__":
    test_voice_manager()
