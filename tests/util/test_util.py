#!/usr/bin/env python3

"""
Test suite for utility scripts.

This test suite tests the antibody format conversion utility (chothia2HLT.py)
and ensures it correctly processes antibody and nanobody structures.
"""

import os
from pathlib import Path

import pytest

from .util_test_utils import (compare_files, create_test_report, run_command,
                             verify_hlt_format)

# Test script configurations
SCRIPT_CONFIGS = {
    "antibody_hlt_conversion.sh": {
        "description": "Convert antibody PDB files from Chothia to HLT format",
        "output_files": [
            "antibody_HLT.pdb",
            "nanobody_HLT.pdb"
        ]
    }
}

# Path to test scripts
TEST_SCRIPT_DIR = "tests/util/scripts"


@pytest.mark.parametrize("script_name", list(SCRIPT_CONFIGS.keys()))
def test_util_script(script_name, clean_output_dir, output_dir, ref_dir):
    """
    Test utility scripts against reference outputs.
    
    Args:
        script_name: Name of the script to test
        clean_output_dir: Fixture that cleans the output directory
        output_dir: Path to output directory
        ref_dir: Path to reference directory
    """
    script_path = f"{TEST_SCRIPT_DIR}/{script_name}"
    description = SCRIPT_CONFIGS[script_name]["description"]
    
    # Run the script with the output directory as an argument
    print(f"Running {script_name}: {description}")
    run_command(f"bash {script_path} {output_dir}")
    
    # Get expected output files from the configuration
    expected_files = [(
        os.path.join(ref_dir, output_file),
        os.path.join(output_dir, output_file)
    ) for output_file in SCRIPT_CONFIGS[script_name]["output_files"]]
    
    # Verify each output file follows HLT format requirements
    for _, output_path in expected_files:
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
        validation = verify_hlt_format(output_path)
        assert validation['valid'], f"HLT format validation failed for {output_path}: {validation['issues']}"
    
    # Compare with reference files if they exist
    all_differences = {}
    for ref_file, output_file in expected_files:
        if os.path.exists(ref_file):
            result = compare_files(ref_file, output_file)
            
            # If result is True, files match; if it's a list, there are differences
            if result is not True:
                # Store the differences for reporting
                output_filename = os.path.basename(output_file)
                all_differences[output_filename] = result
                
                # Format the first few differences for display
                diff_message = "\n".join(
                    [f"Line {d.get('line')}: Expected '{d.get('ref', '')}' but got '{d.get('out', '')}" if 'ref' in d and 'out' in d
                     else f"Line {d.get('line')}: {d.get('message', 'Difference found')}" 
                     for d in result[:5]]
                )
                if len(result) > 5:
                    diff_message += f"\n... and {len(result) - 5} more differences"
                
                pytest.fail(f"Differences found in {output_file}:\n{diff_message}")
        else:
            print(f"Reference file not found: {ref_file}. Cannot compare outputs.")


def test_all_scripts_as_suite(clean_output_dir, output_dir, ref_dir):
    """
    Run all scripts in sequence and create a summary report.
    
    This test will run all scripts and create a comprehensive report
    rather than failing on the first error.
    
    Args:
        clean_output_dir: Fixture that cleans the output directory
        output_dir: Path to output directory
        ref_dir: Path to reference directory
    """
    results = {}
    
    for script_name in SCRIPT_CONFIGS.keys():
        script_path = f"{TEST_SCRIPT_DIR}/{script_name}"
        description = SCRIPT_CONFIGS[script_name]["description"]
        print(f"Running {script_name}: {description}")
        
        # Run the script with the output directory as an argument
        try:
            run_command(f"bash {script_path} {output_dir}")
            success = True
            details = []
        except Exception as e:
            success = False
            details = [f"Script execution failed: {str(e)}"]
        
        # Verify HLT format
        if success:
            expected_files = [
                os.path.join(output_dir, output_file)
                for output_file in SCRIPT_CONFIGS[script_name]["output_files"]
            ]
            
            for output_file in expected_files:
                if not os.path.exists(output_file):
                    success = False
                    details.append(f"File not found: {output_file}")
                    continue
                
                validation = verify_hlt_format(output_file)
                if not validation['valid']:
                    success = False
                    details.append(f"HLT format validation failed for {output_file}: {validation['issues']}")
        
        # Check output files
        if success and os.path.exists(ref_dir):
            expected_files = [(
                os.path.join(ref_dir, output_file),
                os.path.join(output_dir, output_file)
            ) for output_file in SCRIPT_CONFIGS[script_name]["output_files"]]
            
            file_differences = {}
            for ref_file, output_file in expected_files:
                if not os.path.exists(output_file):
                    file_differences[os.path.basename(output_file)] = [{
                        'message': 'File not found'
                    }]
                    success = False
                    details.append(f"File not found: {output_file}")
                    continue
                
                if not os.path.exists(ref_file):
                    details.append(f"Reference file not found: {ref_file}")
                    continue
                
                result = compare_files(ref_file, output_file)
                if result is not True:
                    output_filename = os.path.basename(output_file)
                    file_differences[output_filename] = result
                    success = False
                    details.append(f"Differences in {output_filename}: {len(result)} differences found")
        
        results[script_name] = {
            'passed': success,
            'details': details
        }
    
    # Create a report
    create_test_report(results, "tests/util/test_report.txt")
    
    # The test should pass if all scripts passed
    all_passed = all(result['passed'] for result in results.values())
    if not all_passed:
        failed_scripts = [script for script, result in results.items() if not result['passed']]
        pytest.fail(f"Tests failed for scripts: {', '.join(failed_scripts)}. See tests/util/test_report.txt for details.")


def test_specific_hlt_features():
    """Test specific features of the HLT conversion script."""
    # This test will be implemented when we have test HLT reference files
    # It will check for specific requirements of HLT format:
    # - Chain naming (H, L, T)
    # - Chain ordering (H, L, T)
    # - CDR loop annotation
    # - Residue numbering
    pass


if __name__ == "__main__":
    pytest.main(["-v", "tests/util/test_util.py"])