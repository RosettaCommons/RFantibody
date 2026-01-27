#!/usr/bin/env python3

"""
Pytest configuration file for utility script tests.
"""

import os
import shutil
import tempfile

import pytest

from rfantibody.config import PathConfig

# Get test paths for this module
_test_paths = PathConfig.get_test_paths('util')


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
        output_path = _test_paths['outputs']
        os.makedirs(output_path, exist_ok=True)
        return str(output_path)
    else:
        # Create a temporary directory that will be automatically cleaned up
        # We need to keep a reference to temp_dir object so it's not garbage collected
        temp_dir = tempfile.TemporaryDirectory(prefix="rfantibody_util_test_")
        # Add the temp_dir object as an attribute of the request.config
        # to ensure it stays in scope until the end of testing
        request.config._rfantibody_temp_dir = temp_dir
        return temp_dir.name


@pytest.fixture(scope="session")
def ref_dir():
    """Provide the reference directory path."""
    return str(_test_paths['references'])


@pytest.fixture(scope="session")
def clean_output_dir(output_dir):
    """Clean the output directory before and after tests"""
    # Clean before tests
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                if f != "input" and not os.path.isdir(file_path):
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"Error cleaning {file_path}: {e}")
    
    # Ensure input directory exists
    os.makedirs(os.path.join(output_dir, "input"), exist_ok=True)
    
    # Run tests
    yield