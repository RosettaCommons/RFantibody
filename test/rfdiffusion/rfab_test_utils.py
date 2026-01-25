#!/usr/bin/env python3

"""
Utility functions for RFDiffusion tests.
"""

import filecmp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def run_command(cmd, cwd=None):
    """
    Run a shell command and return its output.
    
    Args:
        cmd: Command to run
        cwd: Working directory for the command
        
    Returns:
        Command output as string
        
    Raises:
        RuntimeError: If command fails
    """
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr}")
    return result.stdout


def compare_files(ref_file, output_file):
    """
    Compare two files line by line and return differences.
    
    Args:
        ref_file: Path to reference file
        output_file: Path to output file to compare
        
    Returns:
        True if files match, or a list of differences
    """
    if not os.path.exists(output_file):
        pytest.fail(f"Output file not found: {output_file}")
    
    if not os.path.exists(ref_file):
        pytest.fail(f"Reference file not found: {ref_file}")
    
    # For binary files or exact matching
    if filecmp.cmp(ref_file, output_file, shallow=False):
        return True
    
    # For text files with line-by-line comparison
    differences = []
    with open(ref_file, 'r') as ref, open(output_file, 'r') as out:
        ref_lines = ref.readlines()
        out_lines = out.readlines()
        
        # Check if file lengths match
        if len(ref_lines) != len(out_lines):
            differences.append({
                'line': 0,
                'message': f"File lengths differ: Reference has {len(ref_lines)} lines, output has {len(out_lines)} lines"
            })
        
        # Get differences line by line
        for i, (ref_line, out_line) in enumerate(zip(ref_lines, out_lines)):
            ref_line = ref_line.strip()
            out_line = out_line.strip()
            if ref_line != out_line:
                differences.append({
                    'line': i + 1,
                    'ref': ref_line,
                    'out': out_line
                })
    
    return differences if differences else True


def copy_reference_files(ref_dir, output_dir):
    """
    Copy all reference files to the output directory.
    
    This can be used to create initial reference files when setting up the tests.
    
    Args:
        ref_dir: Path to reference directory
        output_dir: Path to output directory
    """
    for item in os.listdir(output_dir):
        src_path = os.path.join(output_dir, item)
        dst_path = os.path.join(ref_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")


def filter_comparison_files(file_list):
    """
    Filter a list of files to only include .pdb and .qv files for comparison.
    
    Args:
        file_list: List of file paths to filter
        
    Returns:
        List containing only .pdb and .qv files
    """
    return [f for f in file_list if f.endswith('.pdb') or f.endswith('.qv')]


def create_test_report(test_results, output_file="test_report.txt"):
    """
    Create a test report summarizing results.
    
    Args:
        test_results: Dictionary mapping script names to test results
        output_file: Path to write report to
    """
    with open(output_file, 'w') as f:
        f.write("RFDiffusion Test Suite Report\n")
        f.write("============================\n\n")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result['passed'])
        
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {total_tests - passed_tests}\n\n")
        
        f.write("Test Details:\n")
        f.write("------------\n\n")
        
        for script, result in test_results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            f.write(f"{script}: {status}\n")
            
            if not result['passed'] and 'details' in result:
                f.write(f"  Failures:\n")
                for detail in result['details']:
                    f.write(f"  - {detail}\n")
            
            f.write("\n") 