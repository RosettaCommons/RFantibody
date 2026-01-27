# Testing Guide

## Requirements

- Python 3.10
- Supported GPU: NVIDIA A4000 or H100
- Dependencies installed: `uv sync --all-extras`

## Running Tests

### Run all tests

```bash
uv run python -m test.run_tests
```

### Run a specific module

```bash
uv run python -m test.run_tests --module rfdiffusion
uv run python -m test.run_tests --module proteinmpnn
uv run python -m test.run_tests --module rf2
uv run python -m test.run_tests --module util
```

### Keep test outputs for inspection

By default, tests use temporary directories that are cleaned up automatically. To preserve outputs:

```bash
uv run python -m test.run_tests --keep-outputs
```

Outputs will be saved to `test/<module>/example_outputs_<timestamp>/`.

### Verbose output

```bash
uv run python -m test.run_tests --verbose
```

## Creating Reference Files

Reference files are GPU-specific (A4000 vs H100 produce different outputs). To regenerate them:

```bash
uv run python -m test.run_tests --create-refs
```

This will:
1. Run all test scripts
2. Copy outputs to the appropriate GPU-specific reference directory:
   - `test/<module>/reference_outputs/A4000_references/`
   - `test/<module>/reference_outputs/H100_references/`

## Test Structure

```
test/
├── run_tests.py          # Main test runner
├── conftest.py           # Root pytest config
├── rfdiffusion/
│   ├── conftest.py       # Module pytest config
│   ├── test_rfdiffusion.py
│   ├── scripts/          # Test shell scripts
│   ├── inputs_for_test/  # Test input files
│   └── reference_outputs/
│       ├── A4000_references/
│       └── H100_references/
├── proteinmpnn/
│   └── (same structure)
├── rf2/
│   └── (same structure)
└── util/
    └── (same structure)
```

## How Tests Work

1. Test scripts in `scripts/` run the CLI commands with deterministic settings
2. Outputs are compared against GPU-specific reference files
3. Tests pass if outputs match references (with tolerance for floating-point differences)

## Running Individual Test Scripts

```bash
# RFdiffusion
bash test/rfdiffusion/scripts/antibody_pdbdesign.sh /path/to/output

# ProteinMPNN
bash test/proteinmpnn/scripts/ab_seq_design.sh /path/to/output

# RF2
bash test/rf2/scripts/ab_prediction.sh /path/to/output
```
