#!/bin/bash

poetry run python /opt/RFantibody/scripts/proteinmpnn_interface_design.py \
    -pdbdir /opt/RFantibody/scripts/examples/proteinmpnn/example_inputs \
    -outpdbdir /opt/RFantibody/scripts/examples/proteinmpnn/example_outputs
