#!/bin/bash

# Example: Design antibody sequences using ProteinMPNN
#
# This example uses the proteinmpnn CLI to design sequences
# for antibody CDR loops.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run proteinmpnn \
    --input-dir "$SCRIPT_DIR/example_inputs" \
    --output-dir "$SCRIPT_DIR/example_outputs"
