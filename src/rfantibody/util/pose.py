from dataclasses import dataclass
from typing import List

import numpy as np

from rfantibody.rf2.modules.parsers import get_cdr_masks_from_remarks, parse_pdblines
from rfantibody.rfdiffusion.chemical import aa2num, num2aa
from rfantibody.util.io import ab_write_pdblines


def range1(n):
    return range(1, n+1)


@dataclass
class Pose():

    atoms: np.ndarray # [L, 3, 3] tensor of backbone atom coordinates
    seq: np.ndarray # [L] tensor of amino acid residues in 3 letter format
    chain: np.ndarray # [L] tensor of chain identifiers

    cdr_dict: dict[str, list[int]] # dictionary of CDR indices, with indices starting at 1


    @classmethod
    def from_pdb(cls, pdbfile: str) -> 'Pose':
        '''
        Load a pdb file into a Pose object

        Args:
            pdbfile:
                The path to the pdb file to load
        '''
        
        with open(pdbfile, 'r') as f:
            pdblines = f.readlines()

        return cls.from_pdblines(pdblines)


    @classmethod
    def from_pdblines(cls, pdblines: List[str]) -> 'Pose':
        '''
        Create a Pose object from a list of pdb lines

        Args:
            pdblines:
                A list of pdb lines to parse
        '''
        
        seq, pdb_idx, xyz = parse_pdblines(pdblines)

        # Parse to a backbone xyz tensor
        bb_xyz = xyz[:, :4, :] # [L, 4, 3]

        # Convert the sequence from numbers to 3 letter amino acids
        seq = np.array([num2aa[i] for i in seq])

        # Get the chain identifiers, pdb_idx is a list of tuples (chain, resnum)
        chains = np.array([i[0] for i in pdb_idx])

        cdr_masks = get_cdr_masks_from_remarks(pdb_idx, pdblines)

        # Now turn the cdr_masks into a dict of cdr indices
        cdr_dict = {
            'H1': [],
            'H2': [],
            'H3': [],
            'L1': [],
            'L2': [],
            'L3': [],
        }

        for cdr, mask in cdr_masks.items():
            # Convert to 1-indexed as documented in cdr_dict
            cdr_dict[cdr] = [i + 1 for i in np.where(mask)[0].tolist()]

        return cls(
            atoms=bb_xyz,
            seq=seq,
            chain=chains,
            cdr_dict=cdr_dict,
        )


    def assert_HLT(self) -> bool:
        """
        Check if the pose is currently in HLT order.

        Returns:
            True if the pose is in HLT order, False otherwise.
        """

        # We will collect the consecutive chains in the pose

        # Find the indices where the value changes
        change_indices = np.where(np.diff(self.chain) != 0)[0] + 1

        # Include the first element as it is always unique in this context
        unique_indices = np.insert(change_indices, 0, 0)

        # Get the consecutive unique chains
        unique_chains = self.chain[unique_indices]

        # Check two things about these chains:
        # 1. The chains must be unique ie. there are no dis-contiguous chains
        # 2. The chains must be in the order H, L, T. Here, either H or L but not both can be missing,
        #    but T must be present

        # Check 1
        if np.unique(unique_chains).size != unique_chains.size:
            return False
        
        # Check 2
        if 'T' not in unique_chains:
            return False
        
        if 'H' in unique_chains and 'L' in unique_chains:
            return unique_chains == np.array(['H', 'L', 'T'])
        
        if 'H' in unique_chains and 'L' not in unique_chains:
            return unique_chains == np.array(['H', 'T'])
        
        if 'H' not in unique_chains and 'L' in unique_chains:
            return unique_chains == np.array(['L', 'T'])
        
        # If we get here something has gone wrong
        raise Exception(f'Unsupported combination of chains: {unique_chains} provided')


    def mutate_residue(
            self,
            chain: str,
            residx: int,
            newres: str,
    ) -> None:
        '''
        Mutate a residue in a pose

        Args:
            chain:
                The chain identifier of the residue to mutate

            residx:
                The zero-indexed residue index of the residue to mutate

            newres:
                The new 3 letter residue name to assign to the specified residue
        '''
        
        # Assert that the residue index is within the bounds of the chain
        assert self.chain[residx] == chain, 'Residue index is not in the specified chain'

        # Assert that the new residue is a valid amino acid
        assert newres in num2aa, 'Invalid amino acid'

        # Assign the new residue to the sequence
        self.seq[residx] = newres


    def dump_pdb(self, pdbfile: str) -> None:
        '''
        Dump a Pose object to a pdb file

        Args:
            pdbfile:
                The path to the pdb file to write
        '''
        
        pdblines = self.to_pdblines()

        with open(pdbfile, 'w') as f:
            f.writelines(pdblines)


    def to_pdblines(self) -> List[str]:
        '''
        Convert a pose to a list of pdb lines

        Returns:
            A list of pdb lines representing the pose
        '''

        # Convert the sequence back to numbers
        seq = np.array([aa2num[i] for i in self.seq])

        pdblines = ab_write_pdblines(
            atoms=self.atoms,
            seq=seq,
            chain_idx=self.chain,
            loop_map=self.cdr_dict,
            num2aa=num2aa,
        )

        return pdblines

