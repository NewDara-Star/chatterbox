#!/bin/bash
# Activate the isolated environment
source "$(dirname "$0")/venv_sfx/bin/activate"

# Run the CLI script with arguments
python "$(dirname "$0")/generate_sfx_cli.py" "$@"
