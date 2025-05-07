#!/bin/bash

mkdir -p /opt/RFantibody/scripts/examples/rf2/example_outputs

poetry run python /opt/RFantibody/scripts/rf2_predict.py \
    input.pdb_dir=/opt/RFantibody/scripts/examples/rf2/example_inputs \
    output.pdb_dir=/opt/RFantibody/scripts/examples/rf2/example_outputs