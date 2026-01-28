#!/usr/bin/env python3

"""
Test suite for RFDiffusion model validation.

This test suite runs the RFDiffusion example scripts as separate parametrized tests
and compares their outputs with reference outputs to validate that the model is correctly set up.
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
    "antibody_pdbdesign.sh": {
        "description": "Design an antibody using a PDB framework",
        "output_files": [
            "ab_des_0.pdb",
            "ab_des_1.pdb"
        ]
    },
    "antibody_qvdesign.sh": {
        "description": "Design an antibody using quiver output format",
        "output_files": [
            "ab_designs.qv"
        ]
    },
    "nanobody_pdbdesign.sh": {
        "description": "Design a nanobody using a PDB framework",
        "output_files": [
            "nb_des_0.pdb",
            "nb_des_1.pdb"
        ]
    },
    "nanobody_qvdesign.sh": {
        "description": "Design a nanobody using quiver output format",
        "output_files": [
            "nb_designs.qv"
        ]
    }
}

# Path to test scripts
TEST_SCRIPT_DIR = "test/rfdiffusion/scripts"


@pytest.mark.parametrize("script_name", SCRIPT_CONFIGS.keys())
def test_rfdiffusion_script(script_name, clean_output_dir, output_dir, ref_dir):
    """
    Run an individual RFDiffusion script and verify its output.
    
    This test is parametrized to run each script separately for better isolation
    and more detailed reporting on failures.
    
    Args:
        script_name: The name of the script to test
        clean_output_dir: Fixture that cleans the output directory
        output_dir: Path to output directory
        ref_dir: Path to reference directory
    """
    script_path = f"{TEST_SCRIPT_DIR}/{script_name}"
    description = SCRIPT_CONFIGS[script_name]["description"]
    print(f"\n\nTesting: {script_name} - {description}")
    
    # Create a report file for this specific test
    report_file = f"test/rfdiffusion/reports/{os.path.splitext(script_name)[0]}_report.txt"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    results = {}
    
    # Run the script with the output directory as an argument
    try:
        print(f"Running command: bash {script_path} {output_dir}")
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
            
            # Use biotite-based comparison for PDB files, generic comparison for others
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
    
    # Create a detailed report for this specific test
    create_test_report(results, report_file)
    
    # Log detailed information about GPU and test environment in the report
    with open(report_file, 'a') as f:
        f.write("\nTest Environment:\n")
        f.write("----------------\n")
        import torch
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            f.write(f"GPU: {gpu_info.name}\n")
            f.write(f"Reference directory: {ref_dir}\n")
            f.write(f"Output directory: {output_dir}\n")
    
    # The test should pass if the script passed
    if not success:
        # Format failure message with details
        fail_details = "\n".join(details)
        pytest.fail(f"Test failed for script: {script_name}\n{fail_details}\nSee {report_file} for full details.")


if __name__ == "__main__":
    pytest.main(["-v", "test/rfdiffusion/test_rfdiffusion.py"])