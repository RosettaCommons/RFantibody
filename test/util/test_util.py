#!/usr/bin/env python3

"""
Test suite for utility scripts.

This test suite tests the antibody format conversion utility (chothia2HLT.py)
and ensures it correctly processes antibody and nanobody structures.
"""

import os
import torch
from pathlib import Path

import pytest

from .util_test_utils import (compare_files, compare_structures,
                             create_test_report, run_command,
                             verify_hlt_format)

SCRIPT_CONFIGS = {
    "antibody_hlt_conversion.sh": {
        "description": "Convert antibody PDB files from Chothia to HLT format",
        "output": "antibody_HLT.pdb",
        "ref": "hu-4D5-8_Fv.pdb"
    },
    "nanobody_hlt_conversion.sh": {
        "description": "Convert nanobody PDB files from Chothia to HLT format",
        "output": "nanobody_HLT.pdb",
        "ref": "h-NbBCII10.pdb"
    }
}


# Path to test scripts
TEST_SCRIPT_DIR = "test/util/scripts"


@pytest.mark.parametrize("script_name", SCRIPT_CONFIGS.keys())
def test_util_script(script_name, clean_output_dir, output_dir, ref_dir):
    """
    Run an individual utility script and verify its output.
    
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
    report_file = f"test/util/reports/{os.path.splitext(script_name)[0]}_report.txt"
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
    
    # Verify HLT format
    if success:
        output_file_name = SCRIPT_CONFIGS[script_name]["output"]
        output_file = os.path.join(output_dir, output_file_name)
            
        if not os.path.exists(output_file):
            success = False
            details.append(f"File not found: {output_file}")
        else:
            validation = verify_hlt_format(output_file)
            if not validation['valid']:
                success = False
                details.append(f"HLT format validation failed for {output_file}: {validation['issues']}")
    
    # Check output files
    if success:
        file_differences = {}
        
        output_file_name = SCRIPT_CONFIGS[script_name]["output"]
        output_file = os.path.join(output_dir, output_file_name)
        
        if not os.path.exists(output_file):
            file_differences[output_file_name] = [{
                'message': 'File not found'
            }]
            success = False
            details.append(f"File not found: {output_file}")
        else:
            # Get the corresponding reference file
            ref_file_name = SCRIPT_CONFIGS[script_name]["ref"]
            ref_file = os.path.join(ref_dir, ref_file_name)
            
            if not os.path.exists(ref_file):
                details.append(f"Reference file not found: {ref_file}")
            else:
                # Compare files
                text_result = compare_files(ref_file, output_file)
                if text_result is not True:
                    file_differences[output_file_name] = text_result
                    success = False
                    details.append(f"Text differences in {output_file_name}: {len(text_result)} differences found")
                
                # Check structure
                struct_result = compare_structures(ref_file, output_file)
                if struct_result is not True:
                    success = False
                    details.append(f"Structure differences in {output_file_name}")
    
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
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_properties(0)
            f.write(f"GPU: {gpu_info.name}\n")
        f.write(f"Reference directory: {ref_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    # The test should pass if the script passed
    if not success:
        pytest.fail(f"Test failed for script: {script_name}. See {report_file} for details.")


if __name__ == "__main__":
    pytest.main(["-v", "test/util/test_util.py"])