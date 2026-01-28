import os

import numpy as np

from rfantibody.proteinmpnn.util_protein_mpnn import aa_1_3
from rfantibody.util.pose import Pose


class SampleFeatures():
    '''
    This is a struct which keeps all the features related to a single sample together
    '''

    def __init__(
        self,
        pose: Pose,
        tag: str,
    ) -> None:

        self.pose = pose
        self.tag = os.path.basename(tag).split('.')[0]

    
    def loop_string2fixed_res(
        self,
        loop_string: str,
    ) -> None:
        """
        Given a loop string, create a dict of residues which should be designed by ProteinMPNN

        The dict is keyed by chain and the values are lists of residue indices. The lists are:
        - 1-indexed
        - Indexed relative to the start of the chain
        """

        # First do some sanity checks

        if loop_string == '':
            raise Exception('Received empty loop string. Skipping example')

        desloops = [l.upper() for l in loop_string.split(',')]

        # Assert correct chain ordering with T (target) last for interface design
        chains = np.unique(self.pose.chain).tolist()
        valid_orderings = [['H', 'L', 'T'], ['H', 'T']]
        assert chains in valid_orderings, (
            f"Invalid chain ordering: {chains}. "
            f"Expected one of {valid_orderings}. "
            "Chains must be in H-L-T order with T (target) last for interface design."
        )

        fixed_res = {}

        nchains = self.pose.chain.size
        
        if nchains <= 1:
            raise Exception('Too few chains detected. Skipping')

        # Now, we will make a fixed res dictionary for each chain that indicates which residues in each
        # chain should NOT be designed by ProteinMPNN

        # Determine the length of the H chain
        lenH = np.where(self.pose.chain == 'H')[0].size
        lenL = np.where(self.pose.chain == 'L')[0].size

        # Now we will parse the H chain loops (if any)
        loopH = []
        for loop in desloops:
            if 'H' in loop:
                loopH += self.pose.cdr_dict[loop]

        # Then we will parse the L chain loops (if any)
        loopL = []
        for loop in desloops:
            if 'L' in loop:
                loopL += self.pose.cdr_dict[loop]

        # Assert that we found at least one residue to design
        assert len(loopH) + len(loopL) > 0, (
            f"No designable residues found for requested loops: {desloops}. "
            "Check that the PDB has REMARK PDBinfo-LABEL lines defining CDR regions."
        )

        # We must now invert these "designable" residue lists into "fixed" residue lists
        idxH = list(range(1, lenH + 1))
        for res in loopH:
            idxH.remove(res)

        print(f'loopH: {loopH}')
        print(f'loopL: {loopL}')

        idxL = list(range(lenH + 1, lenH + lenL + 1))
        for res in loopL:
            idxL.remove(res)

        if 'H' in self.pose.chain:
            fixed_res['H'] = idxH

        if 'L' in self.pose.chain:
            fixed_res['L'] = [i - lenH for i in idxL] # We must subtract lenH to make the L chain 1-indexed

        if 'T' in self.pose.chain:
            lenT = np.where(self.pose.chain == 'T')[0].size

            idxT = list(range(1, lenT + 1))

            fixed_res['T'] = idxT

        # Final assignment
        self.fixed_res = fixed_res
        self.chains = np.unique(self.pose.chain).tolist()
            
    
    def thread_mpnn_seq(self, binder_seq: str) -> None:
        '''
        Thread the binder sequence onto the pose being designed
        '''

        for resi, mut_to in enumerate(binder_seq):
            name3 = aa_1_3[mut_to]

            self.pose.mutate_residue(
                chain = self.pose.chain[resi],
                residx = resi,
                newres = name3,
            )