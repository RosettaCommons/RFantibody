# RFantibody Test Suite

This directory contains test suites for the RFantibody components.

## Test Structure

The tests are organized by module:

- `rfdiffusion/`: Tests for the RFdiffusion module
- (More modules will be added as needed)

## Running Tests

To run all tests:

```bash
# Run tests individually (fails on first error)
./tests/run_tests.py

# Run all tests as a suite and generate a report
./tests/run_tests.py --run-all

# Run tests with verbose output
./tests/run_tests.py -v

# Keep output files in standard location instead of using temporary directory
./tests/run_tests.py --keep-outputs
```

By default, tests use a temporary directory for outputs that is automatically cleaned up when tests complete. Use the `--keep-outputs` flag if you want to inspect the output files after the tests run. When using this flag, outputs will be stored in the module-specific output directory (e.g., `tests/rfdiffusion/example_outputs`).

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- A supported GPU (A4000 or H100)
- pytest