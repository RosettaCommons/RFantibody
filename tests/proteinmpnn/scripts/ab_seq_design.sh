#!/bin/bash

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"/home/tests/proteinmpnn/example_outputs"}

# Create input directory with reference structure
mkdir -p ${OUTPUT_DIR}/input
cp /home/scripts/examples/example_inputs/hu-4D5-8_Fv.pdb ${OUTPUT_DIR}/input/

# Set random seed via environment variable to ensure deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

poetry run python /home/scripts/proteinmpnn_interface_design.py \
    -pdbdir ${OUTPUT_DIR}/input \
    -outpdbdir ${OUTPUT_DIR} \
    -loop_string 'H1,H2,H3,L1,L2,L3' \
    -seqs_per_struct 2 \
    -temperature 0.2 \
    -augment_eps 0 \
    -deterministic \
    -debug