#!/usr/bin/env python3

"""
Pytest configuration file for ProteinMPNN tests.
"""

import os
import shutil

import pytest
import torch

# Option is defined in the root conftest.py


@pytest.fixture(scope="session", autouse=True)
def check_gpu():
    """Check if CUDA is available and we're on a supported GPU"""
    if not torch.cuda.is_available():
        pytest.skip("No GPU found, tests require a supported GPU (A4000 or H100)")
    
    gpu_info = torch.cuda.get_device_properties(0)
    if 'A4000' not in gpu_info.name and 'H100' not in gpu_info.name:
        pytest.skip("Tests require a supported GPU (A4000 or H100)")
    
    # Log which GPU and reference data we're using
    print(f"Running tests on {gpu_info.name} GPU")
    if 'A4000' in gpu_info.name:
        print("Using A4000-specific reference outputs")
    elif 'H100' in gpu_info.name:
        print("Using H100-specific reference outputs")


@pytest.fixture(scope="session")
def output_dir(request):
    """
    Create and provide a temporary directory for test results.
    
    By default, uses a system temporary directory that will be automatically
    cleaned up. If --keep-outputs is specified, uses the fixed output path.
    """
    # Check if we should keep outputs in the standard location
    keep_outputs = request.config.getoption("--keep-outputs", default=False)
    
    if keep_outputs:
        # Use a dedicated path in the module test directory for inspection
        output_path = "tests/proteinmpnn/example_outputs"
        os.makedirs(output_path, exist_ok=True)
        return output_path
    else:
        # Create a temporary directory that will be automatically cleaned up
        # We need to keep a reference to temp_dir object so it's not garbage collected
        import tempfile
        temp_dir = tempfile.TemporaryDirectory(prefix="rfantibody_proteinmpnn_test_")
        # Add the temp_dir object as an attribute of the request.config
        # to ensure it stays in scope until the end of testing
        request.config._rfantibody_proteinmpnn_temp_dir = temp_dir
        return temp_dir.name


@pytest.fixture(scope="session")
def ref_dir():
    """
    Provide the reference directory path based on GPU type.
    
    Uses GPU-specific references when running on a supported GPU (A4000 or H100).
    """
    base_ref_dir = "tests/proteinmpnn/reference_outputs"
    
    # Check which GPU we're running on
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        if 'A4000' in gpu_info.name:
            return os.path.join(base_ref_dir, "A4000_references")
        elif 'H100' in gpu_info.name:
            return os.path.join(base_ref_dir, "H100_references")
    
    # Default reference dir for other GPUs
    return base_ref_dir


@pytest.fixture(scope="session")
def clean_output_dir(output_dir):
    """Clean the output directory before and after tests"""
    # Clean before tests
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")
    
    # Run tests
    yield
    
    # By default, temporary directories are automatically cleaned up
    # If using fixed path (--keep-outputs), we leave files for inspection