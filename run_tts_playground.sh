#!/usr/bin/env bash
# Simple launcher for the TTS playground (short‑text Gradio UI)
# Run this from the project root:
#   ./run_tts_playground.sh

# Activate virtual environment
source venv/bin/activate

# Run the app
python gradio_tts_app.py
