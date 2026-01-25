#!/bin/bash

# Get the output directory from the first argument or use the default
OUTPUT_DIR=${1:-"/home/scripts/examples/example_outputs"}

uv run python /home/scripts/rfdiffusion_inference.py \
    --config-name antibody \
    antibody.target_pdb=/home/test/rfdiffusion/inputs_for_test/rsv_site3.pdb \
    antibody.framework_pdb=/home/test/rfdiffusion/inputs_for_test/h-NbBCII10.pdb \
    inference.ckpt_override_path=/home/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.final_step=48 \
    diffuser.T=50 \
    inference.deterministic=True \
    inference.output_prefix=${OUTPUT_DIR}/nb_des