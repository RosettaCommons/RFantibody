#!/usr/bin/env python3

"""
Main script to run the RFDiffusion test suite.

This script provides a command-line interface to run the tests
and generate reference files when needed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
from rfdiffusion.rfab_test_utils import copy_reference_files


def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description='Run RFDiffusion tests')
    parser.add_argument('--create-refs', action='store_true', 
                        help='Create reference files from current outputs')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all tests and generate a report without failing on first error')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--keep-outputs', action='store_true',
                        help='Keep test outputs in standard directory instead of using temporary directory')
    args = parser.parse_args()
    
    # Set up paths
    tests_dir = Path(__file__).parent
    output_dir = Path("tests/rfdiffusion/example_outputs")
    ref_dir = tests_dir / "rfdiffusion" / "reference_outputs"
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    
    # If creating reference files, run scripts and copy outputs
    if args.create_refs:
        print("Running scripts to create reference files...")
        
        # First check if we're on the right GPU
        try:
            import torch
            if not torch.cuda.is_available():
                print("Error: No GPU found. Reference files must be created on an A4000 GPU.")
                return 1
            
            gpu_info = torch.cuda.get_device_properties(0)
            if 'A4000' not in gpu_info.name and 'H100' not in gpu_info.name:
                print(f"Error: Wrong GPU type. Found {gpu_info.name}, but reference files must be created on an A4000 GPU.")
                return 1
        except ImportError:
            print("Warning: torch not found, cannot check GPU type")
        
        # Run each of our test scripts with the output directory as an argument
        script_dir = Path("tests/rfdiffusion/scripts")
        for script in script_dir.glob("*.sh"):
            print(f"Running {script.name}...")
            subprocess.run(['bash', str(script), str(output_dir)], check=True)
        
        # Copy output files to reference directory
        print("Copying output files to reference directory...")
        copy_reference_files(ref_dir, output_dir)
        print("Reference files created successfully.")
        return 0
    
    # Run tests
    if args.run_all:
        print("Running all tests as a suite...")
        # Run just the suite test that won't fail on first error
        pytest_args = ["-xvs", "tests/rfdiffusion/test_rfdiffusion.py::test_all_scripts_as_suite"]
    else:
        print("Running tests...")
        pytest_args = ["-xvs", "tests/rfdiffusion/test_rfdiffusion.py"]
        
    if args.verbose:
        pytest_args.append("-v")
    
    # Pass through the keep-outputs flag to pytest
    if args.keep_outputs:
        print("Using tests/rfdiffusion/example_outputs directory (outputs will be kept)")
        pytest_args.append("--keep-outputs")
    else:
        print("Using temporary directory for outputs (will be cleaned up after tests)")
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main()) 