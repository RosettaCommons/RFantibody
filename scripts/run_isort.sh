#\!/usr/bin/env bash
# Script to run isort on the codebase

set -e

echo "Running isort to organize imports..."

# Use poetry to run isort
poetry run isort src/
poetry run isort scripts/
poetry run isort tests/

echo "Import formatting complete\!"
