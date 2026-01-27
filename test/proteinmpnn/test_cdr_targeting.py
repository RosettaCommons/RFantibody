#!/usr/bin/env python3

"""
Test suite for verifying CDR targeting in ProteinMPNN.

This test validates that the CDR indexing is correct by:
1. Loading an input PDB with known CDR annotations
2. Running ProteinMPNN targeting specific CDRs
3. Verifying that only the targeted CDR residues were mutated
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rfantibody.util.pose import Pose


class TestCDRIndexing:
    """Test that CDR indices are correctly computed and used."""

    @pytest.fixture
    def test_input_pdb(self):
        """Path to the test input PDB with CDR annotations."""
        return Path(__file__).parent / "inputs_for_test" / "ab_des_0.pdb"

    def test_cdr_dict_indices_are_1_indexed(self, test_input_pdb):
        """
        Test that cdr_dict contains 1-indexed global positions.

        The REMARK PDBinfo-LABEL lines use 1-indexed global residue numbers.
        cdr_dict should contain these same 1-indexed positions.
        """
        pose = Pose.from_pdb(str(test_input_pdb))

        # Parse expected CDR positions from REMARK lines directly
        expected_cdrs = {
            'H1': [], 'H2': [], 'H3': [],
            'L1': [], 'L2': [], 'L3': []
        }

        with open(test_input_pdb, 'r') as f:
            for line in f:
                if line.startswith('REMARK PDBinfo-LABEL'):
                    parts = line.strip().split()
                    # Format: REMARK PDBinfo-LABEL: <resnum> <cdr>
                    resnum = int(parts[2])
                    cdr = parts[3].upper()
                    if cdr in expected_cdrs:
                        expected_cdrs[cdr].append(resnum)

        # Sort and compare
        for cdr in expected_cdrs:
            expected_cdrs[cdr] = sorted(expected_cdrs[cdr])
            actual = sorted(pose.cdr_dict[cdr])
            assert actual == expected_cdrs[cdr], (
                f"CDR {cdr} indices mismatch:\n"
                f"  Expected (from REMARK): {expected_cdrs[cdr]}\n"
                f"  Actual (from cdr_dict): {actual}"
            )

    def test_cdr_positions_match_chain_residues(self, test_input_pdb):
        """
        Test that CDR positions correctly correspond to chain residues.

        H chain CDRs (H1, H2, H3) should have indices within the H chain range.
        L chain CDRs (L1, L2, L3) should have indices within the L chain range.
        """
        pose = Pose.from_pdb(str(test_input_pdb))

        # Get chain lengths
        len_H = np.sum(pose.chain == 'H')
        len_L = np.sum(pose.chain == 'L')

        # H chain CDRs should be in range [1, len_H]
        for cdr in ['H1', 'H2', 'H3']:
            for idx in pose.cdr_dict[cdr]:
                assert 1 <= idx <= len_H, (
                    f"CDR {cdr} has index {idx} outside H chain range [1, {len_H}]"
                )

        # L chain CDRs should be in range [len_H + 1, len_H + len_L]
        for cdr in ['L1', 'L2', 'L3']:
            for idx in pose.cdr_dict[cdr]:
                assert len_H + 1 <= idx <= len_H + len_L, (
                    f"CDR {cdr} has index {idx} outside L chain range "
                    f"[{len_H + 1}, {len_H + len_L}]"
                )


class TestProteinMPNNCDRTargeting:
    """
    Test that ProteinMPNN correctly targets only specified CDRs.

    These tests run ProteinMPNN with specific CDR selections and verify
    that only those CDRs are designed (mutated).
    """

    @pytest.fixture
    def test_input_pdb(self):
        """Path to the test input PDB with CDR annotations."""
        return Path(__file__).parent / "inputs_for_test" / "ab_des_0.pdb"

    def _run_proteinmpnn(self, input_pdb: Path, output_dir: Path, loops: str) -> Path:
        """
        Run ProteinMPNN targeting specific loops.

        Uses --allow-x and --omit-aas to force designed positions to be UNK,
        making it trivially clear which positions were targeted.

        Args:
            input_pdb: Path to input PDB file
            output_dir: Directory for output files
            loops: Comma-separated list of CDRs to design (e.g., "H3")

        Returns:
            Path to the output PDB file
        """
        # Create a temporary input directory with just this PDB
        input_dir = output_dir / "input"
        input_dir.mkdir(exist_ok=True)

        # Copy the input PDB to the temp input directory
        shutil.copy(input_pdb, input_dir / input_pdb.name)

        # Omit all amino acids except X, and allow X output
        # This forces designed positions to be UNK in the output
        all_aas_except_x = "ACDEFGHIKLMNPQRSTVWY"

        # Run ProteinMPNN via CLI
        cmd = [
            "uv", "run", "proteinmpnn",
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--loops", loops,
            "--seqs-per-struct", "1",
            "--temperature", "0.1",
            "--augment-eps", "0",
            "--omit-aas", all_aas_except_x,
            "--allow-x",
            "--deterministic",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"ProteinMPNN failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Find the output file
        output_files = list(output_dir.glob("*_dldesign_*.pdb"))
        assert len(output_files) > 0, (
            f"No output files found in {output_dir}\n"
            f"Contents: {list(output_dir.iterdir())}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        return output_files[0]

    def _get_designed_positions(self, output_pdb: Path) -> set:
        """
        Find positions that were designed (marked as UNK) in the output PDB.

        Since we run ProteinMPNN with --allow-x and --omit-aas excluding all
        real amino acids, designed positions will be 'UNK' in the output.

        Args:
            output_pdb: Path to output PDB

        Returns:
            Set of 1-indexed global positions that were designed (have UNK residue)
        """
        output_pose = Pose.from_pdb(str(output_pdb))

        designed = set()
        for i, res in enumerate(output_pose.seq):
            if res == 'UNK':
                designed.add(i + 1)  # Convert to 1-indexed

        return designed

    def _get_cdr_positions_from_remarks(self, pdb_path: Path, cdrs: list) -> set:
        """
        Parse CDR positions directly from REMARK PDBinfo-LABEL lines.

        This is the ground truth - these positions come directly from the PDB
        file and are independent of cdr_dict parsing. This ensures the test
        catches any indexing bugs in cdr_dict.

        Args:
            pdb_path: Path to PDB file
            cdrs: List of CDR names to get positions for (e.g., ['H3'] or ['H1', 'L3'])

        Returns:
            Set of 1-indexed global positions for the requested CDRs
        """
        positions = set()
        cdrs_upper = [c.upper() for c in cdrs]

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('REMARK PDBinfo-LABEL'):
                    parts = line.strip().split()
                    # Format: REMARK PDBinfo-LABEL: <resnum> <cdr>
                    resnum = int(parts[2])
                    cdr = parts[3].upper()
                    if cdr in cdrs_upper:
                        positions.add(resnum)

        return positions

    def test_h3_only_targeting(self, test_input_pdb):
        """
        Test that targeting H3 designs exactly the H3 residues.

        Uses --allow-x to make designed positions UNK, making it trivially
        clear which positions were targeted. Compares against REMARK lines
        (ground truth) to catch any indexing bugs in cdr_dict.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_pdb = self._run_proteinmpnn(test_input_pdb, output_dir, "H3")

            # Get expected H3 positions from REMARK lines (ground truth)
            expected_h3_positions = self._get_cdr_positions_from_remarks(test_input_pdb, ['H3'])

            # Get positions that were designed (marked as UNK)
            designed_positions = self._get_designed_positions(output_pdb)

            # Designed positions should exactly match H3 positions from REMARK lines
            assert designed_positions == expected_h3_positions, (
                f"Designed positions don't match H3 from REMARK lines:\n"
                f"  Expected H3 positions (from REMARK): {sorted(expected_h3_positions)}\n"
                f"  Actual designed (UNK) positions: {sorted(designed_positions)}\n"
                f"  Missing: {sorted(expected_h3_positions - designed_positions)}\n"
                f"  Extra: {sorted(designed_positions - expected_h3_positions)}"
            )

            print(f"H3 targeting test passed:")
            print(f"  H3 positions (from REMARK): {sorted(expected_h3_positions)}")
            print(f"  Designed (UNK) positions: {sorted(designed_positions)}")

    def test_l1_only_targeting(self, test_input_pdb):
        """
        Test that targeting L1 designs exactly the L1 residues.

        Uses --allow-x to make designed positions UNK, making it trivially
        clear which positions were targeted. Compares against REMARK lines
        (ground truth) to catch any indexing bugs in cdr_dict.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_pdb = self._run_proteinmpnn(test_input_pdb, output_dir, "L1")

            # Get expected L1 positions from REMARK lines (ground truth)
            expected_l1_positions = self._get_cdr_positions_from_remarks(test_input_pdb, ['L1'])

            # Get positions that were designed (marked as UNK)
            designed_positions = self._get_designed_positions(output_pdb)

            # Designed positions should exactly match L1 positions from REMARK lines
            assert designed_positions == expected_l1_positions, (
                f"Designed positions don't match L1 from REMARK lines:\n"
                f"  Expected L1 positions (from REMARK): {sorted(expected_l1_positions)}\n"
                f"  Actual designed (UNK) positions: {sorted(designed_positions)}\n"
                f"  Missing: {sorted(expected_l1_positions - designed_positions)}\n"
                f"  Extra: {sorted(designed_positions - expected_l1_positions)}"
            )

            print(f"L1 targeting test passed:")
            print(f"  L1 positions (from REMARK): {sorted(expected_l1_positions)}")
            print(f"  Designed (UNK) positions: {sorted(designed_positions)}")

    def test_multiple_cdr_targeting(self, test_input_pdb):
        """
        Test that targeting H1,L3 designs exactly those CDR residues.

        Uses --allow-x to make designed positions UNK, making it trivially
        clear which positions were targeted. Compares against REMARK lines
        (ground truth) to catch any indexing bugs in cdr_dict.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_pdb = self._run_proteinmpnn(test_input_pdb, output_dir, "H1,L3")

            # Get expected positions from REMARK lines (ground truth)
            expected_positions = self._get_cdr_positions_from_remarks(test_input_pdb, ['H1', 'L3'])

            # Get positions that were designed (marked as UNK)
            designed_positions = self._get_designed_positions(output_pdb)

            # Designed positions should exactly match H1+L3 positions from REMARK lines
            assert designed_positions == expected_positions, (
                f"Designed positions don't match H1+L3 from REMARK lines:\n"
                f"  Expected positions (from REMARK): {sorted(expected_positions)}\n"
                f"  Actual designed (UNK) positions: {sorted(designed_positions)}\n"
                f"  Missing: {sorted(expected_positions - designed_positions)}\n"
                f"  Extra: {sorted(designed_positions - expected_positions)}"
            )

            print(f"H1+L3 targeting test passed:")
            print(f"  H1+L3 positions (from REMARK): {sorted(expected_positions)}")
            print(f"  Designed (UNK) positions: {sorted(designed_positions)}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
