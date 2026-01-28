#\!/usr/bin/env bash
# Script to run isort on the codebase

set -e

echo "Running isort to organize imports..."

# Use uv to run isort
uv run isort src/
uv run isort scripts/
uv run isort test/

echo "Import formatting complete\!"
