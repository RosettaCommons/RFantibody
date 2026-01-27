#!/bin/bash

# Example: Predict antibody structures using RF2
#
# This example uses the rf2 CLI to predict/refine antibody structures.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRIPT_DIR/example_outputs"

uv run rf2 \
    --input-dir "$SCRIPT_DIR/example_inputs" \
    --output-dir "$SCRIPT_DIR/example_outputs"
