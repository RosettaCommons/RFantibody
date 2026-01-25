#!/bin/bash

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"/home/test/rf2/example_outputs"}

# Use test inputs directory
INPUT_DIR=/home/test/rf2/inputs_for_test

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export TORCH_USE_CUDA_DSA=0

# Run RF2 prediction
uv run python /home/scripts/rf2_predict.py \
    input.pdb_dir=${INPUT_DIR} \
    output.pdb_dir=${OUTPUT_DIR} \
    inference.num_recycles=3 \
    inference.cautious=False \
    +inference.seed=42