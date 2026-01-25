#!/bin/bash

# Get the absolute path of the directory where this script is located
currdir=$(cd `dirname $0` && pwd) &&

mkdir -p $currdir/dgl &&
cd $currdir/dgl &&

wget https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl &&

# Build the USAlign binaries
cd $currdir/USalign &&
make &&

# Build the Python package
cd /home &&
uv sync --all-extras

echo "Setup successful."
