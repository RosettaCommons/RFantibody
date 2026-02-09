#!/bin/bash

# Test script for RF2 antibody structure prediction from JSON input
# Uses the older weights (RFab_noframework) with default inference configs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="$(cd "$TEST_DIR/../.." && pwd)"

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"$TEST_DIR/example_outputs"}

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export TORCH_USE_CUDA_DSA=0

uv run rf2 \
    --input-json "$TEST_DIR/inputs_for_test/rfab_targets.json" \
    --output-dir "$OUTPUT_DIR" \
    --weights "$PROJECT_DIR/weights/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1" \
    --num-recycles 10 \
    --no-cautious \
    --seed 42 \
    --hotspot-show-prop 0
