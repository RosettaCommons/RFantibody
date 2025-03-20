#!/bin/bash

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"/home/tests/rf2/example_outputs"}

# Create input directory with reference antibody structure
mkdir -p ${OUTPUT_DIR}/input
cp /home/scripts/examples/rf2/example_inputs/ab_proteinmpnn_output.pdb ${OUTPUT_DIR}/input/

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export TORCH_USE_CUDA_DSA=0

# Run RF2 prediction
poetry run python /home/scripts/rf2_predict.py \
    input.pdb_dir=${OUTPUT_DIR}/input \
    output.pdb_dir=${OUTPUT_DIR} \
    inference.num_recycles=3 \
    inference.cautious=False \
    +inference.seed=42