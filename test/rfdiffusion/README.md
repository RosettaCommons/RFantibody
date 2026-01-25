# RFDiffusion Test Suite

This directory contains a pytest-based test suite for validating that the RFDiffusion model is correctly set up and working as expected.

## Overview

The test suite runs four example scripts as separate parametrized tests and compares their outputs with reference outputs in the `reference_outputs` directory. This ensures that the model is properly installed and working correctly.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- A supported GPU (A4000 or H100)
- pytest

## Usage

### Running the Tests

To run the RFdiffusion tests, use the run_tests.py script from the parent directory:

```bash
# Run all tests (each script runs separately as a parameterized test)
uv run python -m test.run_tests --module rfdiffusion

# Run tests with verbose output
uv run python -m test.run_tests --module rfdiffusion -v

# Keep output files for inspection
uv run python -m test.run_tests --module rfdiffusion --keep-outputs
```

By default, tests now use a temporary directory for outputs that is automatically cleaned up when tests complete. Use the `--keep-outputs` flag if you want to inspect the output files after the tests run. When using this flag, outputs will be stored in the `test/rfdiffusion/example_outputs` directory.

### Creating Reference Files

If you need to create or update the reference files:

```bash
uv run python -m test.run_tests --module rfdiffusion --create-refs
```

This will run all the example scripts and copy their outputs to the appropriate reference output directory based on the GPU being used.

## Test Structure

- `test_rfdiffusion.py`: Main test file with parametrized tests for each script
- `conftest.py`: Pytest configuration and fixtures
- `rfab_test_utils.py`: Utility functions for running commands and comparing files
- `scripts/`: Directory containing test script wrappers
- `reference_outputs/`: Directory containing reference output files
  - `A4000_references/`: Reference outputs from A4000 GPU
  - `H100_references/`: Reference outputs from H100 GPU
- `reports/`: Directory containing individual test reports

## Scripts Tested

The following scripts are tested independently through parametrization:

1. `antibody_pdbdesign.sh`: Design an antibody using a PDB framework
2. `antibody_qvdesign.sh`: Design an antibody using quiver output format
3. `nanobody_pdbdesign.sh`: Design a nanobody using a PDB framework
4. `nanobody_qvdesign.sh`: Design a nanobody using quiver output format

## Test Reports

The test suite now generates detailed individual reports:

- Each test produces a report file in `test/rfdiffusion/reports/`
- Report format: `[script_name]_report.txt` (e.g., `antibody_pdbdesign_report.txt`)
- Reports include:
  - Test status (passed/failed)
  - Detailed error messages for failures
  - Information about file differences
  - Test environment details (GPU type, reference directory)

This improved reporting structure makes it easier to identify which specific tests failed and why, while keeping each test independent.