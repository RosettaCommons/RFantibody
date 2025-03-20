# RFantibody Development Guide

## Commands
- Build/setup: `bash include/setup.sh`
- Run tests: `python tests/run_test_suite.py`
- Run a specific example: `bash scripts/examples/[path/to/example].sh`
- Docker build: `docker build -t rfantibody .`
- Docker run: `docker run --name rfantibody --gpus all -v .:/home --memory 10g -it rfantibody`

## Code Style Guidelines
- Python version: 3.10
- Imports: Standard library first, third-party packages second, local modules third
- Naming: Snake_case for variables/functions, PascalCase for classes
- Documentation: Docstrings with descriptions of parameters and return values
- Error handling: Use try/except blocks with specific exception types
- File organization: Modular design with separate directories for components
- Testing: Deterministic tests with reference outputs for validation

## Development Notes
- The codebase primarily uses PyTorch for deep learning components
- RFantibody consists of three main modules: RFdiffusion, ProteinMPNN, and RF2
- Quiver files (.qv) are used for storing multiple protein designs and scores
- Docker container is the recommended deployment environment

## Testing Infrastructure
- Tests require a supported GPU (A4000 or H100)
- Reference outputs are now organized by GPU type in the reference_outputs directory
- A4000-specific reference outputs are stored in reference_outputs/A4000_references
- H100-specific reference outputs are stored in reference_outputs/H100_references
- The test framework automatically detects GPU type and uses appropriate reference files

## Development Log
### 2025-03-20
- Reorganized test reference outputs to support GPU-specific comparisons
- Added A4000_references directory for A4000 GPU reference outputs
- Added H100_references directory for H100 GPU reference outputs 
- Updated conftest.py to conditionally use GPU-specific reference files based on detected hardware
- Fixed run_test_suite.py to work with both A4000 and H100 GPUs
- Added isort to test dependencies for standardizing import formatting
- Added isort configuration in pyproject.toml to match code style guidelines
- Updated test infrastructure to use temporary directories for test outputs
- Added --keep-outputs flag to optionally preserve test outputs for inspection
- Optimized test output management: by default, tests now use automatically cleaned temporary directories, improving efficiency while still allowing output inspection when needed via the --keep-outputs flag
- Moved test outputs to module-specific output directories (tests/rfdiffusion/example_outputs) for better organization
- Restructured test directory: moved RFdiffusion tests to tests/rfdiffusion/ subdirectory to enable future test modules
- Implemented proper test module organization with __init__.py files and clear directory structure
- Updated path references in test scripts and runners to work with new directory structure
- Improved test output organization with module-specific example_outputs directories
- Added ProteinMPNN test module with deterministic test framework
- Implemented deterministic mode flag (-deterministic) in proteinmpnn_interface_design.py
- Created ab_seq_design.sh test script for ProteinMPNN with fixed random seeds and increased temperature
- Updated run_tests.py to support multiple test modules (rfdiffusion, proteinmpnn)
- Added --module parameter to run_tests.py to selectively run specific test modules
- Added automated reference file generation for each supported GPU type
- Verified test suite works properly with both H100 and A4000 GPUs