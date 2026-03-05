#!/bin/bash

# Get the absolute path of the directory where this script is located
currdir=$(cd `dirname $0` && pwd) &&

# make a weights directory one level up from the current directory
if [ ! -d "${currdir}/../weights" ]; then
    mkdir -p ${currdir}/../weights &&
    cd ${currdir}/../weights &&

    # download the weights
    echo "Downloading RFdiffusion weights..." &&
    wget https://files.ipd.uw.edu/pub/RFantibody/RFdiffusion_Ab.pt &&

    echo "Downloading ProteinMPNN weights..." &&
    wget https://files.ipd.uw.edu/pub/RFantibody/ProteinMPNN_v48_noise_0.2.pt &&

    echo "Downloading RF2 weights..." &&
    wget https://files.ipd.uw.edu/pub/RFantibody/RF2_ab.pt &&

    echo "Downloading RF2 TCR weights..." &&
    wget -O RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt https://zenodo.org/records/17488258/files/RFab_noframework-nosidechains-5-10-23_trainingparamsadded.pt?download=1 &&

    echo "All weights downloaded successfully!"
else
    echo "Weights directory already exists! Skipping download."
fi
