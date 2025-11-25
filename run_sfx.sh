#!/bin/bash
# Activate virtual environment
source venv_sfx/bin/activate

# Fix MPS memory fragmentation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Run the CLI script with all arguments
python generate_sfx_cli.py "$@"
