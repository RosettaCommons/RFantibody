#!/bin/bash

# Example: Design antibody backbones and output to Quiver file
#
# This example uses the rfdiffusion CLI to design antibody CDR loops
# and saves the results to a Quiver file for downstream processing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

uv run rfdiffusion \
    --target "$EXAMPLES_DIR/example_inputs/rsv_site3.pdb" \
    --framework "$EXAMPLES_DIR/example_inputs/hu-4D5-8_Fv.pdb" \
    --output-quiver "$EXAMPLES_DIR/example_outputs/ab_designs.qv" \
    --num-designs 2 \
    --design-loops "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13" \
    --hotspots "T305,T456" \
    --diffuser-t 50 \
    --deterministic
