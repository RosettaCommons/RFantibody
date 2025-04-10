#!/bin/bash

# Script to convert Chothia-formatted antibody PDB files to HLT format

set -e  # Exit on any error

# Get output directory from command line argument or use current directory
OUTPUT_DIR=${1:-$(pwd)}
echo "Using output directory: $OUTPUT_DIR"

# Use test inputs directory
INPUT_DIR="/home/tests/util/inputs_for_test"

# Convert the nanobody file
echo "Converting nanobody file..."
python /home/scripts/util/chothia2HLT.py \
  "${INPUT_DIR}/h-NbBCII10.pdb" \
  --heavy H \
  --output "${OUTPUT_DIR}/nanobody_HLT.pdb"

# Convert the antibody file
echo "Converting antibody file..."
python /home/scripts/util/chothia2HLT.py \
  "${INPUT_DIR}/hu-4D5-8_Fv.pdb" \
  --heavy H \
  --light L \
  --output "${OUTPUT_DIR}/antibody_HLT.pdb"

echo "HLT conversion completed."