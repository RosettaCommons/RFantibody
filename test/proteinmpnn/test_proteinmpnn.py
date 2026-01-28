#!/usr/bin/env python3

"""
Test suite for ProteinMPNN model validation.

This test suite runs the ProteinMPNN example scripts and compares their outputs
with reference outputs to validate that the model is correctly set up.
"""

import os
from pathlib import Path

import pytest

from test.util.util_test_utils import (
    compare_files,
    compare_pdb_structures,
    create_test_report,
    run_command,
)

# Test script configurations
SCRIPT_CONFIGS = {
    "ab_seq_design.sh": {
        "description": "Design antibody sequences with ProteinMPNN",
        "output_files": [
            "ab_des_0_dldesign_0.pdb",
            "ab_des_0_dldesign_1.pdb"
        ]
    }
}

# Path to test scripts
TEST_SCRIPT_DIR = "test/proteinmpnn/scripts"


@pytest.mark.parametrize("script_name", list(SCRIPT_CONFIGS.keys()))
def test_proteinmpnn_script(script_name, clean_output_dir, output_dir, ref_dir):
    """
    Test ProteinMPNN scripts against reference outputs.
    
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
    
    # Check if all expected files exist
    for ref_path, output_path in expected_files:
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
    
    # Compare each output file with reference
    all_differences = {}
    for ref_file, output_file in expected_files:
        # Use biotite-based comparison for PDB files
        if output_file.endswith('.pdb'):
            result = compare_pdb_structures(ref_file, output_file)
        else:
            result = compare_files(ref_file, output_file)
        
        # If result is True, files match; if it's a list, there are differences
        if result is not True:
            # Store the differences for reporting
            output_filename = os.path.basename(output_file)
            all_differences[output_filename] = result
            
            # Format the first few differences for display
            diff_message = "\n".join(
                [d.get('message', 'Difference found') for d in result[:5]]
            )
            if len(result) > 5:
                diff_message += f"\n... and {len(result) - 5} more differences"
            
            pytest.fail(f"Differences found in {output_file}:\n{diff_message}")


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
        
        # Check output files
        if success:
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
                
                # Use biotite-based comparison for PDB files
                if output_file.endswith('.pdb'):
                    result = compare_pdb_structures(ref_file, output_file)
                else:
                    result = compare_files(ref_file, output_file)
                
                if result is not True:
                    output_filename = os.path.basename(output_file)
                    file_differences[output_filename] = result
                    success = False
                    details.append(f"Differences in {output_filename}: {len(result)} differences found")
                    
                    # Format the first few differences for display
                    diff_message = "\n".join(
                        [d.get('message', 'Difference found') for d in result[:5]]
                    )
                    if len(result) > 5:
                        diff_message += f"\n... and {len(result) - 5} more differences"
                    details.append(diff_message)
        
        results[script_name] = {
            'passed': success,
            'details': details
        }
    
    # Create a report
    create_test_report(results, "test/proteinmpnn/test_report.txt")
    
    # The test should pass if all scripts passed
    all_passed = all(result['passed'] for result in results.values())
    if not all_passed:
        failed_scripts = [script for script, result in results.items() if not result['passed']]
        # Collect all failure details
        all_details = []
        for script, result in results.items():
            if not result['passed']:
                all_details.append(f"\n{script}:")
                all_details.extend([f"  {d}" for d in result['details']])
        fail_details = "\n".join(all_details)
        pytest.fail(f"Tests failed for scripts: {', '.join(failed_scripts)}\n{fail_details}\nSee test/proteinmpnn/test_report.txt for full details.")


if __name__ == "__main__":
    pytest.main(["-v", "test/proteinmpnn/test_proteinmpnn.py"])