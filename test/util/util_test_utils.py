#!/usr/bin/env python3

"""
Utility functions for utility script tests.
"""

import filecmp
import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from biotite.structure.io.pdb import PDBFile


def _extract_remark_lines(pdb_path: Union[str, Path]) -> list:
    """
    Extract REMARK lines from a PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        List of REMARK lines (stripped of whitespace)
    """
    remark_lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('REMARK'):
                remark_lines.append(line.strip())
    return remark_lines


def _extract_score_lines(pdb_path: Union[str, Path]) -> dict:
    """
    Extract SCORE lines from a PDB file (RF2 output format).

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary mapping score names to float values
    """
    scores = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('SCORE '):
                # Format: "SCORE metric_name: value"
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    metric_name = parts[0].replace('SCORE ', '').strip()
                    try:
                        value = float(parts[1].strip())
                        scores[metric_name] = value
                    except ValueError:
                        pass
    return scores


def compare_score_lines(
    ref_file: Union[str, Path],
    output_file: Union[str, Path],
    rel_tolerance: float = 0.01,
    abs_tolerance: float = 0.1,
) -> Union[bool, list]:
    """
    Compare SCORE lines between two PDB files.

    RF2 outputs SCORE lines with metrics like:
    - interaction_pae, pae (predicted aligned error)
    - pred_lddt (confidence/pLDDT)
    - Various RMSD metrics

    Args:
        ref_file: Path to reference PDB file
        output_file: Path to output PDB file to compare
        rel_tolerance: Relative tolerance for score comparison (default 1%)
        abs_tolerance: Absolute tolerance for score comparison (default 0.1)

    Returns:
        True if scores match within tolerances, or a list of differences
    """
    ref_path = Path(ref_file)
    out_path = Path(output_file)

    if not out_path.exists():
        return [{'type': 'file_not_found', 'message': f"Output file not found: {output_file}"}]

    if not ref_path.exists():
        return [{'type': 'file_not_found', 'message': f"Reference file not found: {ref_file}"}]

    ref_scores = _extract_score_lines(ref_path)
    out_scores = _extract_score_lines(out_path)

    differences = []

    # Check for missing scores in output
    for metric in ref_scores:
        if metric not in out_scores:
            differences.append({
                'type': 'missing_score',
                'metric': metric,
                'message': f"Missing score in output: {metric}"
            })

    # Check for extra scores in output
    for metric in out_scores:
        if metric not in ref_scores:
            differences.append({
                'type': 'extra_score',
                'metric': metric,
                'message': f"Extra score in output not in reference: {metric}"
            })

    # Compare matching scores
    for metric in ref_scores:
        if metric in out_scores:
            ref_val = ref_scores[metric]
            out_val = out_scores[metric]

            # Use both relative and absolute tolerance
            # Pass if within either tolerance
            abs_diff = abs(ref_val - out_val)
            rel_diff = abs_diff / abs(ref_val) if ref_val != 0 else abs_diff

            if abs_diff > abs_tolerance and rel_diff > rel_tolerance:
                differences.append({
                    'type': 'score_mismatch',
                    'metric': metric,
                    'ref_value': ref_val,
                    'out_value': out_val,
                    'abs_diff': abs_diff,
                    'rel_diff': rel_diff,
                    'message': f"Score mismatch for {metric}: ref={ref_val:.4f}, out={out_val:.4f} (diff={abs_diff:.4f}, {rel_diff*100:.2f}%)"
                })

    return True if not differences else differences


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

def compare_pdb_structures(
    ref_file: Union[str, Path],
    output_file: Union[str, Path],
    coord_threshold: float = 0.1,
    check_elements: bool = True,
    check_residues: bool = True,
    check_remarks: bool = True,
) -> Union[bool, list]:
    """
    Compare two PDB files using biotite to verify structural consistency.
    
    Checks that:
    1. REMARK lines are identical
    2. Both files have the same number of atoms
    3. Residue names match at each position
    4. Element types match at each position
    5. Coordinates are within the specified threshold
    
    Args:
        ref_file: Path to reference PDB file
        output_file: Path to output PDB file to compare
        coord_threshold: Maximum allowed distance (in Angstroms) between 
                        corresponding atoms. Default is 0.1 Angstroms.
        check_elements: Whether to verify element types match
        check_residues: Whether to verify residue names match
        check_remarks: Whether to verify REMARK lines are identical
        
    Returns:
        True if structures match within tolerances, or a list of differences
    """
    ref_path = Path(ref_file)
    out_path = Path(output_file)
    
    if not out_path.exists():
        pytest.fail(f"Output file not found: {output_file}")
    
    if not ref_path.exists():
        pytest.fail(f"Reference file not found: {ref_file}")
    
    differences = []
    
    # Parse PDB files using biotite
    try:
        ref_pdb = PDBFile.read(str(ref_path))
        ref_structure = ref_pdb.get_structure(model=1)
    except Exception as e:
        differences.append({
            'type': 'parse_error',
            'file': 'reference',
            'message': f"Failed to parse reference PDB: {str(e)}"
        })
        return differences
    
    try:
        out_pdb = PDBFile.read(str(out_path))
        out_structure = out_pdb.get_structure(model=1)
    except Exception as e:
        differences.append({
            'type': 'parse_error',
            'file': 'output',
            'message': f"Failed to parse output PDB: {str(e)}"
        })
        return differences
    
    # Check REMARK lines
    if check_remarks:
        ref_remarks = _extract_remark_lines(ref_path)
        out_remarks = _extract_remark_lines(out_path)
        
        if len(ref_remarks) != len(out_remarks):
            differences.append({
                'type': 'remark_count',
                'ref_count': len(ref_remarks),
                'out_count': len(out_remarks),
                'message': f"REMARK line count mismatch: reference has {len(ref_remarks)}, output has {len(out_remarks)}"
            })
        else:
            for i, (ref_remark, out_remark) in enumerate(zip(ref_remarks, out_remarks)):
                if ref_remark != out_remark:
                    differences.append({
                        'type': 'remark_mismatch',
                        'line_index': i,
                        'ref_remark': ref_remark,
                        'out_remark': out_remark,
                        'message': f"REMARK mismatch at line {i}: ref='{ref_remark}', out='{out_remark}'"
                    })
    
    # Check atom counts
    ref_atom_count = len(ref_structure)
    out_atom_count = len(out_structure)
    
    if ref_atom_count != out_atom_count:
        differences.append({
            'type': 'atom_count',
            'message': f"Atom count mismatch: reference has {ref_atom_count}, output has {out_atom_count}"
        })
        return differences
    
    # Compare atom-by-atom
    for i in range(ref_atom_count):
        ref_atom = ref_structure[i]
        out_atom = out_structure[i]
        
        # Check residue names
        if check_residues and ref_atom.res_name != out_atom.res_name:
            differences.append({
                'type': 'residue_mismatch',
                'atom_index': i,
                'ref_residue': ref_atom.res_name,
                'out_residue': out_atom.res_name,
                'chain': ref_atom.chain_id,
                'res_id': ref_atom.res_id,
                'message': f"Residue mismatch at atom {i}: ref={ref_atom.res_name}, out={out_atom.res_name}"
            })
        
        # Check element types
        if check_elements and ref_atom.element != out_atom.element:
            differences.append({
                'type': 'element_mismatch',
                'atom_index': i,
                'ref_element': ref_atom.element,
                'out_element': out_atom.element,
                'chain': ref_atom.chain_id,
                'res_id': ref_atom.res_id,
                'message': f"Element mismatch at atom {i}: ref={ref_atom.element}, out={out_atom.element}"
            })
        
        # Check coordinates
        ref_coord = ref_atom.coord
        out_coord = out_atom.coord
        distance = np.linalg.norm(ref_coord - out_coord)
        
        if distance > coord_threshold:
            differences.append({
                'type': 'coord_deviation',
                'atom_index': i,
                'atom_name': ref_atom.atom_name,
                'chain': ref_atom.chain_id,
                'res_id': ref_atom.res_id,
                'res_name': ref_atom.res_name,
                'ref_coord': ref_coord.tolist(),
                'out_coord': out_coord.tolist(),
                'distance': float(distance),
                'message': f"Coordinate deviation at atom {i} ({ref_atom.atom_name} in {ref_atom.res_name}{ref_atom.res_id}): distance={distance:.4f}Å > threshold={coord_threshold}Å"
            })
    
    return True if not differences else differences


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

        # Let's only compare lines that begin with REMARK PDBinfo-LABEL:
        ref_lines = [line for line in ref_lines if line.startswith('REMARK PDBinfo-LABEL:')]
        out_lines = [line for line in out_lines if line.startswith('REMARK PDBinfo-LABEL:')]
        
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