#!/bin/bash

# Script to convert Chothia-formatted antibody PDB files to HLT format

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/../.." && pwd)"

# Get output directory from command line argument or use current directory
OUTPUT_DIR=${1:-$(pwd)}
echo "Using output directory: $OUTPUT_DIR"

# Use test inputs directory
INPUT_DIR="$TEST_DIR/inputs_for_test"

# Convert the antibody file
echo "Converting antibody file..."
python "$PROJECT_ROOT/scripts/util/chothia2HLT.py" \
  "${INPUT_DIR}/6mh2_chothia.pdb" \
  --heavy D \
  --light C \
  --Hcrop 113 \
  --Lcrop 107 \
  --output "${OUTPUT_DIR}/antibody_HLT.pdb"

echo "HLT conversion completed."
