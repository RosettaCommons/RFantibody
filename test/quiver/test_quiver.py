#!/usr/bin/env python3

"""
Test suite for Quiver CLI commands.

Adapted from https://github.com/nrbennet/quiver test suite.
Tests the accuracy of the Quiver CLI by ensuring correct PDB handling
and that no PDB lines are lost during manipulation.
"""

import glob
import math
import os
import subprocess
import uuid

import pandas as pd
import pytest


def run_cmd(cmd, cwd=None):
    """Run a shell command and return result."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nStderr: {result.stderr}\nStdout: {result.stdout}")
    return result.stdout


def compare_pdb_files(file1, file2):
    """Compare two PDB files line by line."""
    with open(file1, 'r') as f:
        lines1 = [line.rstrip() for line in f.readlines()]
    with open(file2, 'r') as f:
        lines2 = [line.rstrip() for line in f.readlines()]
    return lines1 == lines2


class TestQuiverCLI:
    """Test suite for Quiver CLI commands."""

    def test_qvfrompdbs_and_qvextract(self, input_dir, work_dir):
        """
        Test that we can create a Quiver file from PDBs and extract them back.
        The extracted PDB files should be identical to the originals.
        """
        # Create quiver file from PDBs
        run_cmd(f'uv run qvfrompdbs {input_dir}/*.pdb > test.qv', cwd=work_dir)

        # Extract the PDB files
        run_cmd(f'uv run qvextract test.qv', cwd=work_dir)

        # Compare extracted files to originals
        for pdb_file in glob.glob(f'{work_dir}/*.pdb'):
            filename = os.path.basename(pdb_file)
            original = os.path.join(input_dir, filename)
            assert compare_pdb_files(pdb_file, original), \
                f"Extracted file {filename} does not match original"

    def test_qvls(self, input_dir, work_dir):
        """
        Test that qvls returns the correct list of tags for a Quiver file.
        """
        # Create quiver file
        run_cmd(f'uv run qvfrompdbs {input_dir}/*.pdb > test.qv', cwd=work_dir)

        # Run qvls
        output = run_cmd(f'uv run qvls test.qv', cwd=work_dir)
        tags = [line.strip() for line in output.strip().split('\n') if line.strip()]

        # Get expected tags from PDB filenames
        pdbs = glob.glob(f'{input_dir}/*.pdb')
        expected_tags = [os.path.basename(pdb)[:-4] for pdb in pdbs]

        # Check all PDB files are listed
        for tag in expected_tags:
            assert tag in tags, f"Tag {tag} not found in qvls output"

        assert len(tags) == len(expected_tags), \
            f"Tag count mismatch: got {len(tags)}, expected {len(expected_tags)}"

    def test_qvextractspecific(self, input_dir, work_dir):
        """
        Test that qvextractspecific extracts only the requested tags.
        """
        # Create quiver file
        run_cmd(f'uv run qvfrompdbs {input_dir}/*.pdb > test.qv', cwd=work_dir)

        # Get 5 random tags
        output = run_cmd(f'uv run qvls test.qv', cwd=work_dir)
        all_tags = [line.strip() for line in output.strip().split('\n') if line.strip()]

        import random
        random.seed(42)
        selected_tags = random.sample(all_tags, min(5, len(all_tags)))

        # Write tags to file and extract
        tags_file = os.path.join(work_dir, 'tags.txt')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(selected_tags))

        run_cmd(f'cat tags.txt | uv run qvextractspecific test.qv', cwd=work_dir)

        # Check only selected tags were extracted
        extracted_pdbs = glob.glob(f'{work_dir}/*.pdb')
        extracted_tags = [os.path.basename(pdb)[:-4] for pdb in extracted_pdbs]

        assert set(selected_tags) == set(extracted_tags), \
            f"Extracted tags don't match: got {extracted_tags}, expected {selected_tags}"

        # Verify content matches originals
        for tag in selected_tags:
            extracted = os.path.join(work_dir, f'{tag}.pdb')
            original = os.path.join(input_dir, f'{tag}.pdb')
            assert compare_pdb_files(extracted, original), \
                f"Extracted {tag}.pdb does not match original"

    def test_qvslice(self, input_dir, work_dir):
        """
        Test that qvslice creates a new Quiver with only the requested tags.
        """
        # Create quiver file
        run_cmd(f'uv run qvfrompdbs {input_dir}/*.pdb > test.qv', cwd=work_dir)

        # Get 5 random tags
        output = run_cmd(f'uv run qvls test.qv', cwd=work_dir)
        all_tags = [line.strip() for line in output.strip().split('\n') if line.strip()]

        import random
        random.seed(42)
        selected_tags = random.sample(all_tags, min(5, len(all_tags)))

        # Write tags and slice
        tags_file = os.path.join(work_dir, 'tags.txt')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(selected_tags))

        run_cmd(f'cat tags.txt | uv run qvslice test.qv > sliced.qv', cwd=work_dir)

        # Extract from sliced quiver
        run_cmd(f'uv run qvextract sliced.qv', cwd=work_dir)

        # Check only selected tags were in sliced quiver
        extracted_pdbs = glob.glob(f'{work_dir}/*.pdb')
        extracted_tags = [os.path.basename(pdb)[:-4] for pdb in extracted_pdbs]

        assert set(selected_tags) == set(extracted_tags), \
            f"Sliced tags don't match: got {extracted_tags}, expected {selected_tags}"

        # Verify content matches originals
        for tag in selected_tags:
            extracted = os.path.join(work_dir, f'{tag}.pdb')
            original = os.path.join(input_dir, f'{tag}.pdb')
            assert compare_pdb_files(extracted, original), \
                f"Sliced {tag}.pdb does not match original"

    def test_qvsplit(self, input_dir, work_dir):
        """
        Test that qvsplit correctly splits a Quiver file into multiple files.

        Verifies:
        1. Correct number of output files
        2. Total tag count matches original
        3. All original PDBs are present across split files
        """
        # Create quiver file
        run_cmd(f'uv run qvfrompdbs {input_dir}/*.pdb > test.qv', cwd=work_dir)

        # Create split directory
        split_dir = os.path.join(work_dir, 'split')
        os.makedirs(split_dir)

        # Split with 3 tags per file
        run_cmd(f'uv run qvsplit {work_dir}/test.qv 3 -o {split_dir}', cwd=work_dir)

        # Count original PDBs
        num_pdbs = len(glob.glob(f'{input_dir}/*.pdb'))
        expected_files = math.ceil(num_pdbs / 3)

        # Check number of split files
        split_files = glob.glob(f'{split_dir}/*.qv')
        assert len(split_files) == expected_files, \
            f"Expected {expected_files} split files, got {len(split_files)}"

        # Count total tags across all split files
        total_tags = 0
        for qv_file in split_files:
            with open(qv_file, 'r') as f:
                tag_count = sum(1 for line in f if line.startswith('QV_TAG'))
            total_tags += tag_count
            # Each file should have at most 3 tags
            assert tag_count <= 3, \
                f"File {qv_file} has {tag_count} tags, expected at most 3"

        assert total_tags == num_pdbs, \
            f"Total tag count {total_tags} doesn't match original {num_pdbs}"

        # Extract all and verify all PDBs present
        for qv_file in split_files:
            run_cmd(f'uv run qvextract {qv_file}', cwd=split_dir)

        extracted_pdbs = glob.glob(f'{split_dir}/*.pdb')
        extracted_tags = set(os.path.basename(pdb)[:-4] for pdb in extracted_pdbs)

        original_tags = set(os.path.basename(pdb)[:-4] for pdb in glob.glob(f'{input_dir}/*.pdb'))

        assert extracted_tags == original_tags, \
            f"Tags mismatch after split: missing {original_tags - extracted_tags}"

        # Verify content matches originals
        for pdb in extracted_pdbs:
            filename = os.path.basename(pdb)
            original = os.path.join(input_dir, filename)
            assert compare_pdb_files(pdb, original), \
                f"Split {filename} does not match original"

    def test_qvrename(self, input_dir, work_dir):
        """
        Test that qvrename correctly renames Quiver entries.

        Verifies:
        1. Tags are renamed correctly
        2. PDB content is unchanged
        3. Score lines are also renamed (if present)
        """
        # Use the scored quiver file
        scored_qv = os.path.join(input_dir, 'designs_scored.qv')

        # Get original tags
        output = run_cmd(f'uv run qvls {scored_qv}', cwd=work_dir)
        original_tags = [line.strip() for line in output.strip().split('\n') if line.strip()]

        # Generate new random tags
        new_tags = [str(uuid.uuid4()) for _ in original_tags]

        # Write new tags and rename
        with open(os.path.join(work_dir, 'newtags.txt'), 'w') as f:
            f.write('\n'.join(new_tags))

        run_cmd(f'cat newtags.txt | uv run qvrename {scored_qv} > renamed.qv', cwd=work_dir)

        # Extract original PDBs
        os.makedirs(os.path.join(work_dir, 'original'), exist_ok=True)
        run_cmd(f'uv run qvextract {scored_qv} -o {work_dir}/original', cwd=work_dir)

        # Extract renamed PDBs
        run_cmd(f'uv run qvextract renamed.qv', cwd=work_dir)

        # Verify renamed tags exist and content matches
        for i, (old_tag, new_tag) in enumerate(zip(original_tags, new_tags)):
            renamed_pdb = os.path.join(work_dir, f'{new_tag}.pdb')
            original_pdb = os.path.join(work_dir, 'original', f'{old_tag}.pdb')

            assert os.path.exists(renamed_pdb), f"Renamed PDB {new_tag}.pdb not found"
            assert compare_pdb_files(renamed_pdb, original_pdb), \
                f"Content mismatch between {old_tag} and {new_tag}"

        # Test score lines are renamed
        # qvscorefile writes to a .sc file with same name as input
        run_cmd(f'uv run qvscorefile {scored_qv}', cwd=work_dir)
        run_cmd(f'uv run qvscorefile renamed.qv', cwd=work_dir)

        # Read the generated .sc files
        orig_sc = os.path.join(input_dir, 'designs_scored.sc')
        renamed_sc = os.path.join(work_dir, 'renamed.sc')
        orig_df = pd.read_csv(orig_sc, sep='\t')
        new_df = pd.read_csv(renamed_sc, sep='\t')

        # Verify scores match (except tag column)
        for i, (old_tag, new_tag) in enumerate(zip(original_tags, new_tags)):
            old_row = orig_df[orig_df['tag'] == old_tag]
            new_row = new_df[new_df['tag'] == new_tag]

            if not old_row.empty and not new_row.empty:
                for col in orig_df.columns:
                    if col != 'tag':
                        assert old_row[col].values[0] == new_row[col].values[0], \
                            f"Score mismatch for {col} between {old_tag} and {new_tag}"

    def test_qvscorefile(self, input_dir, work_dir):
        """
        Test that qvscorefile extracts scores correctly to a .sc file.
        """
        scored_qv = os.path.join(input_dir, 'designs_scored.qv')

        # Extract scores (writes to .sc file with same name as input)
        run_cmd(f'uv run qvscorefile {scored_qv}', cwd=work_dir)

        # Read the generated .sc file
        sc_file = os.path.join(input_dir, 'designs_scored.sc')
        assert os.path.exists(sc_file), f"Score file {sc_file} was not created"

        df = pd.read_csv(sc_file, sep='\t')

        # Should have a 'tag' column
        assert 'tag' in df.columns, "Score file missing 'tag' column"

        # Should have multiple rows
        assert len(df) > 0, "Score file is empty"

        # Tags should match qvls output
        tags_output = run_cmd(f'uv run qvls {scored_qv}', cwd=work_dir)
        expected_tags = [line.strip() for line in tags_output.strip().split('\n') if line.strip()]

        assert set(df['tag'].tolist()) == set(expected_tags), \
            "Score file tags don't match qvls output"
