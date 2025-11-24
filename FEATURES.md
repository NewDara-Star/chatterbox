# Audiobook Converter Features

This document explains the features implemented in the audiobook converter system, their design rationale, and implementation details.

---

## Feature 1: Document Parser

**Status**: ✅ Completed  
**Date**: 2025-11-24

### Overview

The document parser extracts and cleans text from PDF and DOC/DOCX files, preparing it for TTS conversion.

### Implementation Details

**File**: `audiobook_utils.py`

**Key Components**:

1. **DocumentParser Class**
   - Supports PDF (.pdf) and Word (.doc, .docx) formats
   - Extracts text while preserving structure
   - Generates metadata (page count, word count, character count)

2. **Text Cleaning**
   - Removes excessive whitespace
   - Strips standalone page numbers
   - Normalizes line breaks
   - Preserves paragraph structure

3. **Intelligent Chunking**
   - Splits text at sentence boundaries
   - Configurable chunk size (default: 500 characters)
   - Designed for parallel processing on M4 Mac
   - Maintains natural flow for better TTS output

4. **Processing Time Estimation**
   - Estimates audiobook generation time
   - Based on character count and processing speed
   - Conservative estimate: 50 chars/second on M4

### Design Rationale

**Why sentence-based chunking?**
- Maintains natural speech flow
- Prevents awkward breaks mid-sentence
- Enables parallel processing without context loss

**Why 500 character chunks?**
- Balance between processing efficiency and memory usage
- Allows 4-8 parallel workers on M4 without overwhelming GPU
- Typical sentence length fits comfortably within limit

**Why clean text aggressively?**
- Page numbers and headers disrupt TTS narration
- Excessive whitespace causes unnatural pauses
- Clean text = better audio quality

### M4 Optimization

The chunking strategy is specifically designed for Apple Silicon:
- Chunk size optimized for MPS (Metal Performance Shaders) batch processing
- Enables parallel generation across M4's multiple cores
- Reduces memory pressure on unified memory architecture

### Testing

**Test Coverage**:
- ✅ PDF parsing (multi-page documents)
- ✅ DOCX parsing (paragraphs and headings)
- ✅ Text cleaning and normalization
- ✅ Sentence-based chunking
- ✅ Processing time estimation

**Test Results**:
- 2-page PDF: 294 characters, 4 chunks, ~5s estimated
- 1-page DOCX: 286 characters, 4 chunks, ~5s estimated

### Dependencies Added

- `PyPDF2==3.0.1` - PDF text extraction
- `python-docx==1.2.0` - DOCX text extraction
- `reportlab==4.4.5` - Test file generation (dev only)

### Future Enhancements

Potential improvements for future versions:
- Support for EPUB format
- Better header/footer detection
- Chapter detection and metadata
- Multi-language text detection
- OCR support for scanned PDFs

---
