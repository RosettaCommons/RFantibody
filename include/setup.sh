#!/bin/bash

# Get the absolute path of the directory where this script is located
currdir=$(cd `dirname $0` && pwd) &&


# Build the USAlign binaries
cd $currdir/USalign &&
make &&

# Build the Python package
cd /home &&
poetry install &&

echo "Setup successful."
