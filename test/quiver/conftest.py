#!/usr/bin/env python3

"""
Pytest configuration file for Quiver CLI tests.
"""

import os
import tempfile

import pytest

from rfantibody.config import PathConfig

# Get test paths for this module
_test_paths = PathConfig.get_test_paths('quiver')


@pytest.fixture(scope="session")
def input_dir():
    """Provide the input directory with test PDB files."""
    return _test_paths['inputs']


@pytest.fixture(scope="function")
def work_dir():
    """
    Create a temporary working directory for each test.
    Automatically cleaned up after test completes.
    """
    with tempfile.TemporaryDirectory(prefix="quiver_test_") as tmpdir:
        yield tmpdir
