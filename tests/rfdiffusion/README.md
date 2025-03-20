# RFDiffusion Test Suite

This directory contains a pytest-based test suite for validating that the RFDiffusion model is correctly set up and working as expected.

## Overview

The test suite runs four example scripts and compares their outputs with reference outputs in the `reference_outputs` directory. This ensures that the model is properly installed and working correctly.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- A supported GPU (A4000 or H100)
- pytest

## Usage

### Running the Tests

To run the RFdiffusion tests, use the run_tests.py script from the parent directory:

```bash
# Run all tests individually (fails on first error)
./tests/run_tests.py

# Run all tests as a suite and generate a report
./tests/run_tests.py --run-all

# Run tests with verbose output
./tests/run_tests.py -v

# Keep output files for inspection
./tests/run_tests.py --keep-outputs
```

By default, tests now use a temporary directory for outputs that is automatically cleaned up when tests complete. Use the `--keep-outputs` flag if you want to inspect the output files after the tests run. When using this flag, outputs will be stored in the `tests/rfdiffusion/example_outputs` directory.

### Creating Reference Files

If you need to create or update the reference files:

```bash
./tests/run_tests.py --create-refs
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

## Scripts Tested

The following scripts are tested:

1. `antibody_pdbdesign.sh`: Design an antibody using a PDB framework
2. `antibody_qvdesign.sh`: Design an antibody using quiver output format
3. `nanobody_pdbdesign.sh`: Design a nanobody using a PDB framework
4. `nanobody_qvdesign.sh`: Design a nanobody using quiver output format

## Test Reports

When running with `--run-all`, a test report is generated in `tests/rfdiffusion/test_report.txt` with a summary of the test results.