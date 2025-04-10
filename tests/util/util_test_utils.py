#!/usr/bin/env python3

"""
Utility functions for utility script tests.
"""

import filecmp
import os
import shutil
import subprocess
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


def compare_files(ref_file, output_file, ignore_lines=None):
    """
    Compare two files line by line and return differences.
    
    Args:
        ref_file: Path to reference file
        output_file: Path to output file to compare
        ignore_lines: List of line prefixes to ignore (e.g. ["REMARK   1 "])
        
    Returns:
        True if files match, or a list of differences
    """
    if not os.path.exists(output_file):
        pytest.fail(f"Output file not found: {output_file}")
    
    if not os.path.exists(ref_file):
        pytest.fail(f"Reference file not found: {ref_file}")
    
    # For binary files or exact matching
    if ignore_lines is None and filecmp.cmp(ref_file, output_file, shallow=False):
        return True
    
    # For text files with line-by-line comparison
    differences = []
    with open(ref_file, 'r') as ref, open(output_file, 'r') as out:
        ref_lines = ref.readlines()
        out_lines = out.readlines()
        
        # Filter lines if needed
        if ignore_lines:
            ref_lines = [line for line in ref_lines if not any(line.startswith(prefix) for prefix in ignore_lines)]
            out_lines = [line for line in out_lines if not any(line.startswith(prefix) for prefix in ignore_lines)]
        
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
        
        if os.path.isfile(src_path) and item.endswith('.pdb'):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} -> {dst_path}")


def verify_hlt_format(pdb_file):
    """
    Verify that a PDB file conforms to HLT format requirements.
    
    Args:
        pdb_file: Path to PDB file to verify
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'issues': []
    }
    
    # Check if file exists
    if not os.path.exists(pdb_file):
        results['valid'] = False
        results['issues'].append(f"File not found: {pdb_file}")
        return results
    
    # Read file
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    # Check for valid chain IDs (H, L, T)
    chain_ids = set()
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain_id = line[21]
            chain_ids.add(chain_id)
    
    # Check for required chains
    required_chains = {'H'}  # At minimum, H chain should be present
    missing_chains = required_chains - chain_ids
    if missing_chains:
        results['valid'] = False
        results['issues'].append(f"Missing required chains: {', '.join(missing_chains)}")
    
    # Check for invalid chains
    valid_chains = {'H', 'L', 'T', ' '}
    invalid_chains = chain_ids - valid_chains
    if invalid_chains:
        results['valid'] = False
        results['issues'].append(f"Invalid chain IDs found: {', '.join(invalid_chains)}")
    
    # Check for CDR annotations
    cdr_annotations = []
    for line in lines:
        if line.startswith('REMARK PDBinfo-LABEL:'):
            cdr_annotations.append(line.strip())
    
    if not cdr_annotations:
        results['valid'] = False
        results['issues'].append("No CDR annotations found")
    
    return results


def create_test_report(test_results, output_file="test_report.txt"):
    """
    Create a test report summarizing results.
    
    Args:
        test_results: Dictionary mapping script names to test results
        output_file: Path to write report to
    """
    with open(output_file, 'w') as f:
        f.write("Utility Scripts Test Report\n")
        f.write("==========================\n\n")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result['passed'])
        
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Passed: {passed_tests}\n")
        f.write(f"Failed: {total_tests - passed_tests}\n\n")
        
        f.write("Test Details:\n")
        f.write("------------\n\n")
        
        for script, result in test_results.items():
            status = " PASSED" if result['passed'] else " FAILED"
            f.write(f"{script}: {status}\n")
            
            if not result['passed'] and 'details' in result:
                f.write(f"  Failures:\n")
                for detail in result['details']:
                    f.write(f"  - {detail}\n")
            
            f.write("\n")