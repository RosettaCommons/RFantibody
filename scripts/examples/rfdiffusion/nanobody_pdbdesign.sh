#!/bin/bash

# Example: Design nanobody backbones targeting RSV site 3
#
# This example uses the rfdiffusion CLI to design nanobody CDR loops
# that bind to a target antigen (RSV site 3).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

uv run rfdiffusion \
    --target "$EXAMPLES_DIR/example_inputs/rsv_site3.pdb" \
    --framework "$EXAMPLES_DIR/example_inputs/h-NbBCII10.pdb" \
    --output "$EXAMPLES_DIR/example_outputs/nb_des" \
    --num-designs 2 \
    --design-loops "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13" \
    --hotspots "T305,T456" \
    --diffuser-t 50 \
    --deterministic
