#!/bin/bash

# Test script for RF2 antibody structure prediction
#
# Uses the rf2 CLI to run deterministic tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"$TEST_DIR/example_outputs"}

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export TORCH_USE_CUDA_DSA=0

uv run rf2 \
    --input-dir "$TEST_DIR/inputs_for_test" \
    --output-dir "$OUTPUT_DIR" \
    --num-recycles 1 \
    --no-cautious \
    --seed 42 \
    --extra "inference.hotspot_show_proportion=0"
