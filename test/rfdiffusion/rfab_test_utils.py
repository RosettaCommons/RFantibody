#!/usr/bin/env python3

"""
Utility functions for RFDiffusion tests.

This module re-exports common test utilities from test.util.util_test_utils
for backward compatibility.
"""

# Re-export common utilities from the shared module
from test.util.util_test_utils import (
    compare_files,
    compare_pdb_structures,
    copy_reference_files,
    create_test_report,
    run_command,
)

__all__ = [
    "compare_files",
    "compare_pdb_structures",
    "copy_reference_files",
    "create_test_report",
    "run_command",
]


def filter_comparison_files(file_list):
    """
    Filter a list of files to only include .pdb and .qv files for comparison.
    
    Args:
        file_list: List of file paths to filter
        
    Returns:
        List containing only .pdb and .qv files
    """
    return [f for f in file_list if f.endswith('.pdb') or f.endswith('.qv')]
