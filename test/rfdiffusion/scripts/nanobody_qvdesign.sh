#!/bin/bash

# Test script for RFdiffusion nanobody design with Quiver output
#
# Uses the rfdiffusion CLI to run deterministic tests.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"$TEST_DIR/example_outputs"}

uv run rfdiffusion \
    --target "$TEST_DIR/inputs_for_test/rsv_site3.pdb" \
    --framework "$TEST_DIR/inputs_for_test/h-NbBCII10.pdb" \
    --output-quiver "$OUTPUT_DIR/nb_designs.qv" \
    --num-designs 2 \
    --design-loops "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13" \
    --hotspots "T305,T456" \
    --diffuser-t 50 \
    --final-step 48 \
    --deterministic
