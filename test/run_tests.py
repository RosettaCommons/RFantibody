#!/usr/bin/env python3

"""
Main script to run the test suite for RFAntibody.

This script provides a command-line interface to run the tests
and generate reference files when needed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test.rfdiffusion.rfab_test_utils import copy_reference_files


def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description='Run RFAntibody tests')
    parser.add_argument('--module', type=str, default='all',
                        help='Test module to run (rfdiffusion, proteinmpnn, rf2, util, or all)')
    parser.add_argument('--create-refs', action='store_true', 
                        help='Create reference files from current outputs')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--keep-outputs', action='store_true',
                        help='Keep test outputs in standard directory instead of using temporary directory')
    args = parser.parse_args()
    
    # Determine which modules to test
    modules = []
    if args.module == 'all':
        modules = ['rfdiffusion', 'proteinmpnn', 'rf2', 'util']
    else:
        modules = [args.module]
    
    # Validate module selection
    for module in modules:
        if module not in ['rfdiffusion', 'proteinmpnn', 'rf2', 'util']:
            print(f"Error: Unknown module '{module}'. Choose from: rfdiffusion, proteinmpnn, rf2, util, all")
            return 1
    
    # If creating reference files, run scripts and copy outputs for each module
    if args.create_refs:
        print("Running scripts to create reference files...")
        
        # First check if we're on the right GPU for GPU-dependent modules
        gpu_modules = ['rfdiffusion', 'proteinmpnn', 'rf2']
        gpu_dependent_modules = [m for m in modules if m in gpu_modules]
        
        if gpu_dependent_modules:
            try:
                import torch
                if not torch.cuda.is_available():
                    print("Error: No GPU found. Reference files for GPU modules must be created on a supported GPU (A4000 or H100).")
                    if all(m in gpu_modules for m in modules):
                        return 1
                    else:
                        print("Continuing with non-GPU modules only.")
                        modules = [m for m in modules if m not in gpu_modules]
                
                else:
                    gpu_info = torch.cuda.get_device_properties(0)
                    if 'A4000' not in gpu_info.name and 'H100' not in gpu_info.name:
                        print(f"Error: Unsupported GPU type. Found {gpu_info.name}, tests require A4000 or H100.")
                        if all(m in gpu_modules for m in modules):
                            return 1
                        else:
                            print("Continuing with non-GPU modules only.")
                            modules = [m for m in modules if m not in gpu_modules]
                    else:
                        print(f"Creating reference files for {gpu_info.name} GPU")
            except ImportError:
                print("Warning: torch not found, cannot check GPU type")
        
        # Process each selected module
        for module in modules:
            tests_dir = Path(__file__).parent
            module_dir = tests_dir / module
            output_dir = module_dir / "example_outputs"
            
            # Determine the correct reference directory based on GPU type
            ref_base_dir = module_dir / "reference_outputs"
            try:
                import torch
                gpu_info = torch.cuda.get_device_properties(0)
                if 'A4000' in gpu_info.name:
                    ref_dir = ref_base_dir / "A4000_references"
                elif 'H100' in gpu_info.name:
                    ref_dir = ref_base_dir / "H100_references"
                else:
                    ref_dir = ref_base_dir
            except ImportError:
                ref_dir = ref_base_dir
            
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)
            
            # Run each of our test scripts with the output directory as an argument
            script_dir = module_dir / "scripts"
            for script in script_dir.glob("*.sh"):
                print(f"Running {module}/{script.name}...")
                subprocess.run(['bash', str(script), str(output_dir)], check=True)
            
            # Copy output files to reference directory
            print(f"Copying {module} output files to reference directory...")
            copy_reference_files(ref_dir, output_dir)
        
        print("Reference files created successfully.")
        return 0
    
    # Run tests for each selected module
    all_results = []
    for module in modules:
        print(f"\nRunning {module} tests...")
        
        # Configure pytest arguments
        pytest_args = ["-xvs", f"test/{module}/test_{module}.py"]
            
        if args.verbose:
            pytest_args.append("-v")
        
        # Pass through the keep-outputs flag to pytest
        if args.keep_outputs:
            print(f"Using test/{module}/example_outputs directory (outputs will be kept)")
            pytest_args.append("--keep-outputs")
        else:
            print("Using temporary directory for outputs (will be cleaned up after tests)")
        
        # Run the tests for this module
        result = pytest.main(pytest_args)
        all_results.append(result)
    
    # Return the highest (worst) return code
    return max(all_results)


if __name__ == "__main__":
    sys.exit(main())