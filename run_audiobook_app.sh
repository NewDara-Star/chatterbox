#!/bin/bash
# Launch the Audiobook Converter App

# Activate virtual environment
source venv/bin/activate

# Fix MPS memory fragmentation (prevent "out of memory" on M-series chips)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run the app
python gradio_audiobook_app.py
