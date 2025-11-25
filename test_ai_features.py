"""
Quick test script to verify AI text processing features.
Creates a sample PDF with messy formatting and tests the pipeline.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

# Create a test PDF with messy formatting
def create_test_pdf():
    pdf_path = "test_ai_cleanup.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Page 1 with messy formatting
    c.drawString(100, 750, "Page 1 of 2")
    c.drawString(100, 730, "Chapter 1: The Beginning")
    c.drawString(100, 700, "")
    c.drawString(100, 680, "The old wooden door creaked as Sarah pushed it open. Rain pat-")
    c.drawString(100, 660, "tered against the window. She stepped inside, her footsteps echo-")
    c.drawString(100, 640, "ing on the wooden floorboards.")
    c.drawString(100, 600, "Figure 1: Illustration of the door")
    c.drawString(100, 560, "© 2024 Test Publisher Inc.")
    c.drawString(100, 540, "https://example.com/book")
    
    c.showPage()
    
    # Page 2
    c.drawString(100, 750, "Page 2 of 2")
    c.drawString(100, 730, "Chapter 1 (continued)")
    c.drawString(100, 700, "")
    c.drawString(100, 680, "Thunder rumbled in the distance. The room was dark and mus-")
    c.drawString(100, 660, "ty, filled with old furniture covered in white sheets.")
    c.drawString(100, 620, "2")
    
    c.save()
    print(f"Created test PDF: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    # Check if reportlab is installed
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("Installing reportlab...")
        import subprocess
        subprocess.run(["./venv/bin/pip", "install", "reportlab"], check=True)
        from reportlab.pdfgen import canvas
    
    pdf_path = create_test_pdf()
    print(f"\nTest PDF ready: {pdf_path}")
    print("\nExpected AI improvements:")
    print("1. Remove 'Page X of Y' headers")
    print("2. Rejoin hyphenated words (pat-tered -> pattered)")
    print("3. Remove 'Figure 1' caption")
    print("4. Remove copyright and URL")
    print("5. Suggest SFX: door creaking, rain, footsteps, thunder")
