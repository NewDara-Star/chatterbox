"""
Test script for audiobook generator.

Tests the full pipeline: parsing -> voice -> generation.
"""

import os
from audiobook_generator import AudiobookGenerator
from test_parser import create_sample_pdf
from test_voice_manager import create_dummy_wav
from voice_manager import VoiceManager

def test_generator():
    """Test the audiobook generator."""
    print("\n" + "="*60)
    print("AUDIOBOOK GENERATOR TEST")
    print("="*60)
    
    # Setup
    input_pdf = "test_book.pdf"
    voice_file = "test_voice.wav"
    output_wav = "test_audiobook.wav"
    
    # Create sample files
    # Create sample PDF with SFX keywords
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    c = canvas.Canvas(input_pdf, pagesize=letter)
    c.drawString(100, 750, "Chapter 1: The Spooky House")
    c.drawString(100, 700, "The door creaked open slowly.")
    c.drawString(100, 680, "Rain fell heavily against the window pane.")
    c.save()
    
    create_dummy_wav(voice_file)
    
    # Setup voice manager
    vm = VoiceManager()
    try:
        vm.save_voice(voice_file, "TestNarrator", "Testing voice")
        print("✓ Saved test voice")
    except ValueError:
        print("✓ Test voice already exists")
    
    try:
        # Initialize generator
        generator = AudiobookGenerator()
        print("✓ Generator initialized")
        
        # Test generation
        print("\n🎧 Generating audiobook with SFX...")
        
        def progress_callback(p, msg):
            print(f"  [{int(p*100)}%] {msg}")
            
        output_path = generator.generate_audiobook(
            input_path=input_pdf,
            output_path=output_wav,
            voice_name="TestNarrator",
            progress_callback=progress_callback,
            detect_sfx=True  # Enable SFX detection
        )
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✓ Audiobook generated successfully!")
            print(f"  Path: {output_path}")
            print(f"  Size: {size_mb:.2f} MB")
        else:
            print("\n✗ Failed to generate audiobook")
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n" + "="*60)
        print("CLEANUP")
        print("="*60)
        
        if os.path.exists(input_pdf):
            os.remove(input_pdf)
        if os.path.exists(voice_file):
            os.remove(voice_file)
        # Keep the output wav for manual inspection if needed
        # if os.path.exists(output_wav):
        #     os.remove(output_wav)
            
        # Clean up voice from manager
        vm.delete_voice("TestNarrator")
        print("✓ Cleanup complete")

if __name__ == "__main__":
    test_generator()
