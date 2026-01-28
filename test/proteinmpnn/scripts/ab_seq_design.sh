#!/bin/bash

# Test script for ProteinMPNN antibody sequence design
#
# Uses the proteinmpnn CLI to run deterministic tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"$TEST_DIR/example_outputs"}

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

uv run proteinmpnn \
    --input-dir "$TEST_DIR/inputs_for_test" \
    --output-dir "$OUTPUT_DIR" \
    --loops "H1,H2,H3,L1,L2,L3" \
    --seqs-per-struct 2 \
    --temperature 0.2 \
    --augment-eps 0 \
    --deterministic \
    --debug
