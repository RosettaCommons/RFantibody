#!/bin/bash

# Script to convert Chothia-formatted nanobody PDB files to HLT format

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/../.." && pwd)"

# Get output directory from command line argument or use current directory
OUTPUT_DIR=${1:-$(pwd)}
echo "Using output directory: $OUTPUT_DIR"

# Use test inputs directory
INPUT_DIR="$TEST_DIR/inputs_for_test"

# Convert the nanobody file
echo "Converting nanobody file..."
python "$PROJECT_ROOT/scripts/util/chothia2HLT.py" \
  "${INPUT_DIR}/3eak_chothia.pdb" \
  --heavy B \
  --output "${OUTPUT_DIR}/nanobody_HLT.pdb"

echo "HLT conversion completed."
