#!/bin/bash
export HOME=/opt/RFantibody
cd $HOME

python3 -m venv venv
source venv/bin/activate
pip install poetry

mkdir -p dgl
cd $HOME/include/dgl
wget -nc https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl

cd $HOME/include/USalign
make

cd $HOME
poetry install

echo "Setup successful."
