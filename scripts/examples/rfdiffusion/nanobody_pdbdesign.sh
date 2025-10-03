#!/bin/bash

poetry run python  /opt/RFantibody/src/rfantibody/scripts/rfdiffusion_inference.py \
    --config-name antibody \
    antibody.target_pdb=/opt/RFantibody/scripts/examples/example_inputs/rsv_site3.pdb \
    antibody.framework_pdb=/opt/RFantibody/scripts/examples/example_inputs/h-NbBCII10.pdb \
    inference.ckpt_override_path=/opt/RFantibody/weights/RFdiffusion_Ab.pt \
    'ppi.hotspot_res=[T305,T456]' \
    'antibody.design_loops=[L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13]' \
    inference.num_designs=2 \
    inference.final_step=48 \
    inference.deterministic=True \
    diffuser.T=50 \
    inference.output_prefix=/opt/RFantibody/scripts/examples/example_outputs/nb_des
