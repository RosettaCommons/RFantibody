import numpy as np
import os
import sys
import torch
from icecream import ic
from rfantibody.rfdiffusion.parsers import chothia_pdb_parser, HLT_pdb_parser

from rfantibody.rfdiffusion.chemical import INIT_CRDS
import random
from rfantibody.rfdiffusion.util import Dotdict
from rfantibody.rfdiffusion.inference.utils import parse_pdb

def idx2int(idx:str) -> int:
    '''
    Given the string of a residue index that may contain letters, strip the letters
    and return an int of the residue index
    '''
    return int(''.join([i for i in idx if i.isdigit()]))

class AbPose():
    '''A class to convert between different pdb formats
    Currently supported:
        - chothia
        - remarked, HLT
        - contig input

    The core objects in an AbPose are:
        - xyz: a numpy array of xyz coordinates
        - seq: a numpy array of the amino acid sequence (integer)
        - pdb_idx: a list of the pdb index tuples.
            NB these are in the form [(H/L/T, one-indexed number)]
        - loop_dict: a dictionary of loop indices
        - idx: numpy array of the pdb_idx
        - hotspot_dict: a dictionary of hotspot residues
        '''

    def __init__(self):
        '''Initialise'''
        self.H=Dotdict()
        self.L=Dotdict()
        self.T=Dotdict()
        self.cdr_names=['L1','L2','L3','H1','H2','H3']
        self.H_names=['H1','H2','H3']
        self.L_names=['L1','L2','L3']
        self.cdr_chothia={
            'L1':(24,34),'L2':(50,56),'L3':(89,97),'H1':(26,32),'H2':(52,56),'H3':(95,102)
            }

    def has_H(self) -> bool:
        return self.H.xyz is not None

    def has_L(self) -> bool:
        return self.L.xyz is not None

    def has_T(self) -> bool:
        return self.T.xyz is not None

    def binder_len(self) -> int:
        '''
        Calculate and return the length of the antibody binder in the pose. This function
        purposefully does not cache the binder_len so that changes in the binder_len at
        runtime are supported.

        '''

        binder_len = 0
        if self.has_H():
            binder_len += self.H.seq.shape[0]
        if self.has_L():
            binder_len += self.L.seq.shape[0]

        return binder_len

    def length(self) -> int:
        '''
        Calculate and return the full length of the pose. This function also does not
        cache the length for the same reason as in the binder_len function.

        '''

        total_len = self.binder_len()

        if self.has_T():
            total_len += self.T.seq.shape[0]

        return total_len

    def parse_hotspots(self, hotspot_res: str) -> torch.Tensor:
        '''
        Parse a hotspot mask from a string of hotspot residues

        '''
        
        hotspot_idx = torch.zeros(self.length()).bool()
        if hotspot_res is None:
            ic("WARNING! No hotspot residues were provided to the model at inference time")
            return hotspot_idx

        assert all([i[0].isalpha() for i in hotspot_res]), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"
        hotspots = [(i[0],int(i[1:])) for i in hotspot_res]
        
        binderlen = self.binder_len()

        for idx,res in enumerate(self.T.pdb_idx):
            query = (res[0], idx2int(res[1]))
            if query in hotspots:
                ic(f'Using {res} as a hotspot')
                hotspot_idx[idx + binderlen] = True
        
        return hotspot_idx

    def chothia_assign_loops(self, pose: dict) -> dict:
        '''
        Generate a dictionary of one-hot np arrays which indicate which residue positions correspond
        to which chain.
        '''

        cdr_loops = ['L1','L2','L3','H1','H2','H3'] 
        loop_masks = {loop: np.zeros(self.length()).astype(bool) for loop in cdr_loops}

        for i in range(pose['pdb_idx'].shape[0]):
            if pose['pdb_idx'][i,0] == 'T': continue

            if pose['pdb_idx'][i,0] == 'H':
                for l in ['H1','H2','H3']:
                    if idx2int(pose['pdb_idx'][i,1]) in range(self.cdr_chothia[l][0], self.cdr_chothia[l][1]+1):
                        loop_masks[l][i] = True

            if pose['pdb_idx'][i,0] == 'L':
                for l in ['L1','L2','L3']:
                    if idx2int(pose['pdb_idx'][i,1]) in range(self.cdr_chothia[l][0], self.cdr_chothia[l][1]+1):
                        loop_masks[l][i] = True

        return loop_masks

    def combine_loop_masks(self) -> dict:
        '''
        Collect the chains in HLT order and create a dictionary mapping a loop name to
        a boolean mask of shape L which indicates the residues which are in that loop
        '''

        loop_masks = {l:[] for l in self.cdr_names}
        if self.has_H():
            for l in self.cdr_names: loop_masks[l].append(self.H.loop_masks[l])
        if self.has_L():
            for l in self.cdr_names: loop_masks[l].append(self.L.loop_masks[l])
        if self.has_T():
            for l in self.cdr_names: loop_masks[l].append(self.T.loop_masks[l])

        np_loop_masks = {l: np.concatenate(loop_masks[l], axis=0) for l in self.cdr_names}

        return np_loop_masks


    def design_loops2dict(self, design_loops:list) -> dict:

        design_loops_dict = {}
        for entry in design_loops:
            splits = entry.split(':')
            l = splits[0]
            if l.upper() in self.cdr_names:
                design_loops_dict[l] = splits[1]

        return design_loops_dict


    def parse_design_mask(self, design_loops_list:list) -> torch.Tensor:
        '''
        Create a design mask given a dictionary which maps loop names to the desired
        loop lengths. A loop will be designed if it is present in the keys of this
        dictionary, whether or not a set of lengths is specified.
        
        '''

        design_loops_dict = self.design_loops2dict(design_loops_list)

        design_loops = [key for key in design_loops_dict if key in self.cdr_names]
        
        running_design_mask = torch.zeros(self.length())

        loop_masks = self.combine_loop_masks()

        for l in design_loops:
            running_design_mask += loop_masks[l]

        design_mask = torch.where(running_design_mask > 0, True, False)

        return design_mask

    def framework_from_HLT(self, pdb_path: str) -> None:
        '''
        Parse an AbPose from an HLT-formatted PDB file. The chains in this file
        must be in the set {H,L} or this function will fail with an Exception.
        Automatic chain detection may be implemented at a later time.

        These functions will be useful so that users do not have to put their ab and target in the same file

        '''
        chain_order=['H','L']
        pose = HLT_pdb_parser(pdb_path)

        # convert pdb_idx to np.array
        pose['pdb_idx']=np.array(pose['pdb_idx'])

        # Only use H,L chains and only select chains which are present in the input pdb
        present_chains = np.unique(pose['pdb_idx'][:,0]).tolist()
        chains = [chain for chain in chain_order if chain in present_chains]

        assert(len(chains)>0), f'Did not find any chains labeled H or L in the pdb provided'

        mask_dict=Dotdict()
        L = pose['xyz'].shape[0]
        for chain in chains:
            mask_dict[chain] = np.array([
                True if i[0] == chain else False for i in pose['pdb_idx']
            ])
        
        # Add xyz coordinates to each chain object (self.H, self.L)
        # Concatenates target chains into one array
        item_dict={'H':self.H, 'L':self.L}
        for chain in chains:
            item_dict[chain].xyz=pose['xyz'][mask_dict[chain]]
            item_dict[chain].mask=pose['mask'][mask_dict[chain]]
            item_dict[chain].seq=pose['seq'][mask_dict[chain]]
            item_dict[chain].pdb_idx=pose['pdb_idx'][mask_dict[chain]]
        
        # Determine residue positions of the loops
        for chain in chains:
            loop_masks={l: pose['loop_masks'][l][mask_dict[chain]] for l in self.cdr_names}
            item_dict[chain].loop_masks=loop_masks         

    def target_from_HLT(self, pdb_path: str) -> None:
        '''
        Parse an AbPose from an HLT-formatted PDB file. All chains in this file will
        be assumed to be target chains and will be relabelled as chain T in the output

        These functions will be useful so that users do not have to put their ab and target in the same file
        '''
        pose = HLT_pdb_parser(pdb_path)

        # convert pdb_idx to np.array
        pose['pdb_idx']=np.array(pose['pdb_idx'])

        self.T.xyz=pose['xyz']
        self.T.mask=pose['mask']
        self.T.seq=pose['seq']
        self.T.pdb_idx=pose['pdb_idx']
        
        # Determine residue positions of the loops
        for l in self.cdr_names:
            assert(pose['loop_masks'][l].sum() == 0), f'The target protein should not contain any designed loops'

        loop_masks={l: pose['loop_masks'][l] for l in self.cdr_names}
        self.T.loop_masks=loop_masks         

    def from_HLT(self, pdb_path: str) -> None:
        '''
        Parse an AbPose from an HLT-formatted PDB file. This file must contain both
        the target and an antibody docked to it.

        '''
        chain_order=['H','L','T']
        pose = HLT_pdb_parser(pdb_path)

        # convert pdb_idx to np.array
        pose['pdb_idx']=np.array(pose['pdb_idx'])

        # Only use H,L,T chains and only select chains which are present in the input pdb
        present_chains = np.unique(pose['pdb_idx'][:,0]).tolist()
        chains = [chain for chain in chain_order if chain in present_chains]

        assert(len(chains)>0), f'Did not find any chains labeled H, L, or T in the pdb provided'

        mask_dict=Dotdict()
        L = pose['xyz'].shape[0]
        for chain in chains:
            mask_dict[chain] = np.array([
                True if i[0] == chain else False for i in pose['pdb_idx']
            ])
        
        # Add xyz coordinates to each chain object (self.H, self.L, self.T)
        # Concatenates target chains into one array
        item_dict={'H':self.H, 'L':self.L, 'T':self.T}
        for chain in chains:
            item_dict[chain].xyz=pose['xyz'][mask_dict[chain]]
            item_dict[chain].mask=pose['mask'][mask_dict[chain]]
            item_dict[chain].seq=pose['seq'][mask_dict[chain]]
            item_dict[chain].pdb_idx=pose['pdb_idx'][mask_dict[chain]]
        
        # Determine residue positions of the loops
        for chain in chains:
            loop_masks={l: pose['loop_masks'][l][mask_dict[chain]] for l in self.cdr_names}
            item_dict[chain].loop_masks=loop_masks         

    def from_contig(self, pdb_path: str, contig_string: str) -> None:
        raise NotImplementedError()

    def from_chothia(self, pdb_path: str, chain_dict: dict) -> None:
        '''Initialises a pose object from a chothia pdb file'''
        chains={'H':['H'],'L':['L'],'T':['T']}
        pose=chothia_pdb_parser(pdb_path, chains=chains)

        # convert pdb_idx and cdr_bool to np.array
        pose['pdb_idx']=np.array(pose['pdb_idx'])
        pose['cdr_bool']=np.array(pose['cdr_bool'])

        # Make masks for each chain in chain_dict
        # np boolean array of the same length as pdb['xyz']
        # True if the first item in each pdb_idx tuple is the chain in the chain_dict
        # First, check that all chains are present in the pdb_idx first item
        assert set(chain_dict.values()).issubset(set(np.unique([i[0] for i in pose['pdb_idx']]))),\
            "Not all chains in chain_dict are present in pdb_idx"
        L = pose['xyz'].shape[0]
        mask_dict=Dotdict()

        for chain in chain_dict:

            chain_mask=np.full(L, False)
            if not type(chain_dict[chain]) == tuple:
                chain_dict[chain]=tuple(chain_dict[chain])
            for ch in chain_dict[chain]:
                mask_ch = np.array([
                    True if i[0] == ch else False for i in pose['pdb_idx']
                ])
                chain_mask = np.logical_or(chain_mask, mask_ch)

            mask_dict[chain]=chain_mask

        # get integer index
        #pose['idx'] = convert_chothia_idx(pose['idx']).numpy()

        # Add xyz coordinates to each chain object (self.H, self.L, self.T)
        # Concatenates target chains into one array
        item_dict={'H':self.H, 'L':self.L, 'T':self.T}
        for chain in chain_dict:
            item_dict[chain].xyz=pose['xyz'][mask_dict[chain]]
            item_dict[chain].mask=pose['mask'][mask_dict[chain]]
            item_dict[chain].seq=pose['seq'][mask_dict[chain]]
            item_dict[chain].pdb_idx=pose['pdb_idx'][mask_dict[chain]]
            item_dict[chain].cdr_bool=pose['cdr_bool'][mask_dict[chain]]
            #item_dict[chain].idx=pose['idx'][mask_dict[chain]]

        # Determine residue positions of the loops
        pose['loop_masks'] = self.chothia_assign_loops(pose)
        for chain in chain_dict:
            loop_masks={l: pose['loop_masks'][l][mask_dict[chain]] for l in self.cdr_chothia}
            item_dict[chain].loop_masks=loop_masks
    
    def get_interchain_mask(self) -> torch.Tensor:
        '''
        '''
        L = self.length()

        interchain_mask = torch.zeros(L,L).bool()

        if self.has_H() ^ self.has_L():
            # If the pose has only H or only L chain then there are no interchain contacts
            return interchain_mask

        Hlen = self.H.xyz.shape[0]
        Llen = self.L.xyz.shape[0]
       
        H_mask = torch.zeros(L,L).bool()
        L_mask = torch.zeros(L,L).bool()
        T_mask = torch.zeros(L,L).bool()

        H_mask[:Hlen,:Hlen] = 1
        L_mask[Hlen:Hlen+Llen,Hlen:Hlen+Llen] = 1
        T_mask[self.binder_len():, self.binder_len():] = 1

        upper_interchain = ~T_mask *  H_mask * ~L_mask
        lower_interchain = ~T_mask * ~H_mask *  L_mask

        interchain_mask[upper_interchain] = True
        interchain_mask[lower_interchain] = True

        return interchain_mask

    def get_loop_map(self) -> dict:
        '''
        Return a dictionary mapping loop names (eg. 'H1') to a list of the 1-indexed
        residue positions of the residues which make up that loop
    
        '''
        loop_masks = self.combine_loop_masks()
        loop_map = {l: (np.nonzero(loop_masks[l])[0] + 1).tolist() for l in self.cdr_names}

        return loop_map

    def get_chain_idx(self) -> np.array:
        '''
        Parse the current pose to an ordered list of the chain letters of each residue position.
        This is used for writing antibodies to disk

        '''
        chains = []
        if self.has_H():
            chains.append(np.array(['H'] * self.H.xyz.shape[0]))
        if self.has_L():
            chains.append(np.array(['L'] * self.L.xyz.shape[0]))
        if self.has_T():
            chains.append(np.array(['T'] * self.T.xyz.shape[0]))

        assert(len(chains)>0)
        chain_idx = np.concatenate(chains, axis=0)

        return chain_idx


    def slice_features(self, features, loop_masks, slice_mask):
        '''
        '''

        out_features   = {key: features[key][slice_mask] for key in features if not key == 'loop_masks'}
        out_loop_masks = {key: loop_masks[key][slice_mask] for key in loop_masks}

        return out_features, out_loop_masks


    def expand_loop(self, features, loop_masks, curr_loop, length_difference, og_loop_len):
        '''
        '''

        # Scale of noise added to randomly initialized coordinates
        random_noise = 5.0

        out_features = {}

        # Will need to generate each feature independently

        # xyz
        extra_xyz = INIT_CRDS[None].repeat(length_difference, 1, 1) + torch.rand(length_difference, 1, 3)*random_noise # [padded_L,27,3]
        out_features['xyz'] = np.concatenate([extra_xyz.numpy(), features['xyz'][loop_masks[curr_loop]]], axis=0)

        # atom_mask
        extra_mask = np.zeros((length_difference,27)).astype(bool)
        extra_mask[:,:3] = True
        out_features['mask'] = np.concatenate([extra_mask, features['mask'][loop_masks[curr_loop]]], axis=0) # [padded_L,27]

        # seq
        extra_seq = np.ones(length_difference).astype(int) * 21 # Make all of these loops mask
        out_features['seq'] = np.concatenate([extra_seq, features['seq'][loop_masks[curr_loop]]], axis=0) # [padded_L]

        # pdb_idx
        extra_pdb_idx = np.array([('Z',0)] * length_difference) # Making a spoofed pdb_idx since we will not use this
        out_features['pdb_idx'] = np.concatenate([extra_pdb_idx, features['pdb_idx'][loop_masks[curr_loop]]], axis=0) # [padded_L]

        out_loops = {}
        for loop in loop_masks:
            if loop == curr_loop:
                # Matching loops get a mask of all True
                out_loops[loop] = np.ones(length_difference + og_loop_len).astype(bool)
            else:
                # Non-matching loops just get a mask of all False
                out_loops[loop] = np.zeros(length_difference + og_loop_len).astype(bool)

        return out_features, out_loops
    
    
    def update_features(self, features, loop_masks, loop_names, sampled_lengths):
        '''
        '''
        
        loop_idx  = 0
        seg_start = 0
        seg_end   = 0
        in_loop   = False

        for n in loop_names: assert(loop_masks[n].any()), f'No residues found for loop {n}. Running with a subset of loops from one chain is not currently supported'

        segments = {
                     'features'   : [],
                     'loop_masks' : []
                   }

        # Now that we know all of our loops are present, we can just go through in order with the loop_idx
        # Using a while loop so we have the ability to jump around in the index
        idx = 0
        while idx < features['xyz'].shape[0]:
            if loop_idx < len(loop_names) and loop_masks[loop_names[loop_idx]][idx]:
                in_loop = True

                # We have just entered a loop from outside of a loop
                # Dump a segment
                slice_mask = np.zeros(features['xyz'].shape[0]).astype(bool)
                slice_mask[seg_start:seg_end] = True
                seg = self.slice_features(features, loop_masks, slice_mask) # This will loop through all features in the real code
                
                segments['features'].append(seg[0])
                segments['loop_masks'].append(seg[1])
                
                # Determine how to segment the current loop
                og_loop_len = loop_masks[loop_names[loop_idx]].sum()

                if (loop_names[loop_idx] not in sampled_lengths) or (og_loop_len == sampled_lengths[loop_names[loop_idx]]):
                    ##################################################
                    # This loop is going to remain the same size
                    ##################################################

                    seg = self.slice_features(features, loop_masks, loop_masks[loop_names[loop_idx]])
                    
                    segments['features'].append(seg[0])
                    segments['loop_masks'].append(seg[1])

                # If we have gotten here then the current loop is changing in length
                elif sampled_lengths[loop_names[loop_idx]] < og_loop_len:
                    ##################################################
                    # Shrink the size of this loop
                    ##################################################
                    length_difference = og_loop_len - sampled_lengths[loop_names[loop_idx]]

                    slice_mask = np.copy(loop_masks[loop_names[loop_idx]])
                    slice_mask[idx:idx+length_difference] = False # Remove elements from the N terminus of the loop

                    seg = self.slice_features(features, loop_masks, slice_mask)
                                
                    segments['features'].append(seg[0])
                    segments['loop_masks'].append(seg[1])

                elif sampled_lengths[loop_names[loop_idx]] > og_loop_len:
                    ##################################################
                    # Expand the size of this loop
                    ##################################################
                    length_difference = sampled_lengths[loop_names[loop_idx]] - og_loop_len

                    expanded_loop = self.expand_loop(features, loop_masks, loop_names[loop_idx], length_difference, og_loop_len)

                    segments['features'].append(expanded_loop[0])
                    segments['loop_masks'].append(expanded_loop[1])

                else:
                    # Unreachable code
                    raise Exception('Something strange has occured in the loop length modification code')

                # Advance the index past this current loop
                idx += og_loop_len

                # We have parsed this loop, advance the loop index to the next loop
                # Index-out-of-bounds catching is performed in the main if statement of this loop
                loop_idx += 1

                # Setup the segment tracking variables at the new index
                seg_start = idx
                seg_end   = idx

            else:   
                # We are not in a loop
                seg_end = idx + 1
                idx += 1

                if seg_end >= features['xyz'].shape[0]:
                    # We have reached the last residue, dump the final segment
                    slice_mask = np.zeros(features['xyz'].shape[0]).astype(bool)
                    slice_mask[seg_start:] = True
                    
                    seg = self.slice_features(features, loop_masks, slice_mask) # This will loop through all features in the real code
                    
                    segments['features'].append(seg[0])
                    segments['loop_masks'].append(seg[1])
        
        ret_pose = Dotdict()
        
        for key in features:
            if key == 'loop_masks': continue
            feature_segments = [i[key] for i in segments['features']]
            ret_pose[key] = np.concatenate(feature_segments, axis=0)
        
        ret_pose.loop_masks = {}
        for loop in loop_masks:
            loop_segments = [i[loop] for i in segments['loop_masks']]
            ret_pose.loop_masks[loop] = np.concatenate(loop_segments, axis=0)
        
        return ret_pose


    def adjust_loop_lengths(self, design_loops_list: list) -> None:
        '''
        Take a dictionary which maps a loop name to a list of allowed lengths which that loop can be. The loop is then expanded
        or contracted to contain only that many residues. The AbPose object is then updated to reflect the new loop lengths.

        Implementation notes:
        - If a loop is the same size as the length requested, it will be left unaltered
        - If a loop must be contracted to achieve the requested size, it will have residues removed (starting from the N terminus)
          until the desired length is reached
        - If a loop must be expanded to achieve the requested size, the added residues will be initialized at the origin, new
          residues will be added at the N terminus of the loop

        '''

        design_loops = self.design_loops2dict(design_loops_list)

        ### 1. Sample a set of loop lengths according to design_loops
        ######################################################################
        sampled_lengths = {}

        for loop in design_loops:

            ### Input Checking
            if loop.upper() not in self.cdr_names:
                ic(f'Unknown loop name: {loop} encountered in antibody.design_loops. Skipping this loop and proceeding.')
                continue
            if loop.upper()[0] == 'L' and not self.has_L():
                ic(f'Design of L chain loop {loop} requested for an example with only an H chain. Skipping this loop and proceeding.')
                continue
            if loop.upper()[0] == 'H' and not self.has_H():
                ic(f'Design of H chain loop {loop} requested for an example with only an H chain. Skipping this loop and proceeding.')
                continue
            ### End Input Checking

            ### Parse the length specifications from the loop entry

            if len(design_loops[loop]) == 0:
                # Loops with no lengths specified will be designed but will not have their length adjusted
                print(f'Designing loop {loop} and not adjusting length')
                continue

            choices = []
            splits = design_loops[loop].split('|')
            for s in splits:
                if '-' in s:
                    specs = s.split('-')
                    assert(len(specs) == 2)
                    for spec in range(int(specs[0]), int(specs[1])+1):
                        choices.append(spec)
                else:
                    choices.append(int(s))

            ### Randomly sample an allowed length for this loop
            sampled_lengths[loop.upper()] = random.choice(choices)

        ### 2. Adjust the size of the H and L chains of the pose
        ######################################################################

        if self.has_H:
            H_in_sampled_len = len([i for i in sampled_lengths if i[0] == 'H']) > 0
            if H_in_sampled_len:
                self.H = self.update_features(self.H, self.H.loop_masks, self.H_names, sampled_lengths)

        if self.has_L:
            L_in_sampled_len = len([i for i in sampled_lengths if i[0] == 'L']) > 0
            if L_in_sampled_len:
                self.L = self.update_features(self.L, self.L.loop_masks, self.L_names, sampled_lengths)

        # T chain will never be designed so we do not need to update that chain


    def to_diffusion_inputs(self) -> dict:
        inputs = Dotdict()

        xyzs  = []
        seqs  = []
        masks = []

        if self.has_H():
            xyzs.append(self.H.xyz)
            seqs.append(self.H.seq)
            masks.append(self.H.mask)
        if self.has_L():
            xyzs.append(self.L.xyz)
            seqs.append(self.L.seq)
            masks.append(self.L.mask)
        if self.has_T():
            xyzs.append(self.T.xyz)
            seqs.append(self.T.seq)
            masks.append(self.T.mask)

        inputs.xyz_true  = torch.from_numpy(np.concatenate(xyzs, axis=0))
        inputs.seq_true  = torch.from_numpy(np.concatenate(seqs, axis=0))
        inputs.atom_mask = torch.from_numpy(np.concatenate(masks, axis=0))

        return inputs
        

