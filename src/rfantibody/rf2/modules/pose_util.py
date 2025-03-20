from __future__ import annotations

import glob
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from rfantibody.rf2.modules import parsers, util
from rfantibody.rf2.modules.util import Dotdict


@dataclass
class Pose:
    """
    Represents a pose in a molecular structure.

    Attributes:
        xyz (torch.Tensor[float]): The coordinates of the atoms in the pose.
        seq (torch.Tensor[int]): The sequence of the pose.
        atom_mask (torch.Tensor[bool]): The mask indicating which atoms are present in the pose.
        cdrs (CDR[str, torch.Tensor[int]]): The CDRs (Complementarity Determining Regions) of the pose.
        hotspots (torch.Tensor[bool]): The hotspot residues on the target.
        idx: The residue index of the pose.
        chain_dict: a dictionary containing H, L, T boolean masks
    """
    xyz: torch.Tensor # L,3/14/27,3
    seq: torch.Tensor # L
    atom_mask: torch.Tensor # L,3/14/27
    cdrs: CDR
    idx: torch.Tensor # L
    chain_dict: OrderedDict[str, torch.Tensor]

    # optional argument. If not provided, automatically generated
    hotspots: torch.Tensor = field(default=None, init=True)

    @property
    def length(self) -> int:
        return len(self.xyz)
    
    @property
    def antibody_length(self) -> int:
        return sum(self.antibody_mask)
    
    @property
    def target_length(self) -> int:
        return sum(self.target_mask)
    
    @property
    def n_atoms(self) -> int:
        return self.xyz.shape[1]

    def __post_init__(self):
        """
        Calculates hotspots if not provided originally (i.e. if you don't want specific hotspots)
        Verifies that the shape of each tensor attribute matches the expected dimensions.
        """
        if self.hotspots is None:
            self.hotspots = util.get_cdr_hotspots(self)

        attributes_to_check = {'seq': self.seq, 'atom_mask': self.atom_mask, 
                               'hotspots': self.hotspots, 'idx': self.idx}
        attributes_to_check.update(zip(self.cdrs.cdr_names(), self.cdrs.cdrs))

        for attr_name, tensor in attributes_to_check.items():
            if tensor.shape[0] != self.length:
                raise ValueError(f"{attr_name} length ({tensor.shape[0]}) does not match pose length ({self.length})")
        if not self.n_atoms in [3,14,27]: # without/with hydrogens
            raise ValueError("xyz coordinates are wrong shape. Need either with hydrogens (27 atoms) or without (14 atoms), or without sidechains (3 atoms)")
        if self.hotspots[self.antibody_mask].sum() != 0:
            raise ValueError("Antibody chains do not have hotspots. These are defined on the target")
        if any([value[self.target_mask].sum() != 0 for value in self.cdrs.cdrs]):
            raise ValueError("Target chains do not have CDR residues")

    @property
    def xyz27(self) -> torch.Tensor:
        """
        Return xyz coordinates with hydrogens (or padding)
        """
        if self.n_atoms == 27:
            return self.xyz
        else:
            npad=27-self.n_atoms
            return F.pad(self.xyz, (0, 0, 0, npad), "constant", 0)
        
    @property 
    def xyz14(self) -> torch.Tensor:
        """
        Returns xyz coordinates without hydrogens
        """
        if self.n_atoms == 3:
            return F.pad(self.xyz, (0, 0, 0, 11), "constant", 0)
        return self.xyz[:,:14]
    
    @property
    def mask27(self) -> torch.Tensor:
        """
        Returns mask with hydrogens
        """
        if self.atom_mask.shape[1] == 27:
            return self.atom_mask
        else:
            return F.pad(self.atom_mask, (0, 0, 0, 13), "constant", False)
    
    @property
    def mask14(self) -> torch.Tensor:
        """
        Returns mask without hydrogens
        """
        return self.atom_mask[:,:14]
    
    @property
    def chains_present(self) -> list:
        """
        Returns which chains are present in the Pose
        """
        return [ch for ch, mask in self.chain_dict.items() if mask.sum() > 0]

    @property
    def target_mask(self) -> torch.Tensor:
        return self.chain_dict['T']
    
    @property
    def antibody_mask(self) -> torch.Tensor:
        return ~self.target_mask
    
    @property
    def framework_mask(self) -> torch.Tensor:
        return ~self.target_mask & ~self.cdrs.mask_1d
    
    @property
    def pdb_idx(self) -> list:
        """
        Makes pdb idx from chain_dict and idx
        """
        chains=['H'] * sum(self.chain_dict['H']) + ['L'] * sum(self.chain_dict['L']) + ['T'] * sum(self.chain_dict['T'])
        return list(zip(chains, self.idx.tolist()))
    
    def pdblines(self, Bfacts=None) -> list:
        """
        Gets the pdblines from the pose
        """
        return util.get_pdblines(self, Bfacts=Bfacts)
    
    @property
    def same_chain(self) -> torch.Tensor:
        """
        Returns a mask for the same chain.
        Note both antibody chains go on same chain
        """
        same_chain = torch.zeros((self.length, self.length)).long()
        same_chain[:self.target_length, :self.target_length] = 1
        same_chain[self.target_length:, self.target_length:] = 1
        return same_chain

@dataclass
class CDR:
    """
    Dataclass for keeping CDR masks
    H1: torch.Tensor[bool]
    H2: torch.Tensor[bool]
    H3: torch.Tensor[bool]
    L1: torch.Tensor[bool]
    L2: torch.Tensor[bool]
    L3: torch.Tensor[bool]
    """

    H1: torch.Tensor
    H2: torch.Tensor
    H3: torch.Tensor
    L1: torch.Tensor
    L2: torch.Tensor
    L3: torch.Tensor

    def __post_init__(self):
        """
        Verifies that lengths are correct and that all CDRs are non-overlapping,
        and internally contiguous.
        """
        # Check non overlapping
        if not self.total_cdr_length == self.mask_1d.sum():
            raise ValueError("CDR definitions are overlapping")
        # Check contiguous
        for i in self.cdrs:
            if not self.is_contiguous(i):
                raise ValueError("CDRs are discontiguous. This doesn't make sense")
        
        # Check all cdrs are the same length and are tensors
        for i in self.cdrs:
            if i.shape[0] != self.length:
                raise ValueError("Masks for all 6 CDRs must all be provided (even when they are all False) and must be the same length")
            if not type(i) == torch.Tensor or i.dtype != torch.bool:
                raise TypeError("CDR masks must be torch boolean tensors")

    @staticmethod
    def is_contiguous(mask: torch.Tensor) -> bool:
        """
        Checks the True section in a CDR mask (i.e., the CDR) is contiguous
        """
        transitions = torch.diff(mask, prepend=torch.tensor([False]), append=torch.tensor([False]))
        loop_starts = torch.sum(transitions == 1)//2 #as counts both starts and ends
        return loop_starts <= 1
    
    @property
    def cdrs(self) -> list:
        return [self.H1, self.H2, self.H3, self.L1, self.L2, self.L3]
    
    @staticmethod
    def cdr_names() -> list:
        return ['H1','H2','H3','L1','L2','L3']
    
    @property
    def cdrs_present(self) -> list:
        return [self.cdr_names[idx] for idx, i in enumerate(self.cdrs) if i.sum() > 0]

    @property
    def total_cdr_length(self) -> int:
        # Sum all attributes dynamically
        return int(sum([i.sum() for i in self.cdrs]))

    @property
    def mask_1d(self) -> torch.Tensor:
        """
        Returns a combined 1D tensor of all masks
        """       
        return torch.any(torch.stack(self.cdrs), dim=0)  
    
    @property
    def length(self) -> int:
        return self.H1.shape[0]

def combine_chains(*, H=None, L=None, T=None) -> Pose:
    """
    Combine heavy, light, and template chains into a single pose
    Output is in HLT order
    Inputs:
        H: Pose
        L: Pose
        T: Pose
    Outputs:
        Pose object
    """
    chains = [i for i in [H, L, T] if i is not None]
    attributes = ['xyz', 'seq', 'atom_mask', 'hotspots', 'idx']
    combined_attrs = {attr: torch.cat([getattr(i, attr) for i in chains], dim=0) for attr in attributes}
    cdrs = concatenate_cdr_tensors(H, L, T)
    chain_dict = OrderedDict({ch: torch.cat([getattr(i, 'chain_dict')[ch] for i in chains if i is not None], dim=0) for ch in ['H', 'L', 'T']})
    return Pose(**combined_attrs, cdrs=cdrs, chain_dict=chain_dict)

def concatenate_cdr_tensors(H=None, L=None, T=None) -> dict:
    """
    Concatenates the torch tensor values in the cdr objects of H and L.
    Inputs:
        H: Pose
        L: Pose
        T: Pose
    Returns:
        Concatenated CDR object
    """
    chains=[H,L,T]
    concatenated = {cdr: torch.cat([getattr(chain, cdr) for chain in chains if chain is not None]) for cdr in ['H1','H2','H3','L1','L2','L3']}
    return CDR(**concatenated)

def get_chain(pose: Pose, chain: str) -> Pose:
    """
    Grabs a chain from a combined pose
    """
    if chain not in pose.chains_present:
        raise ValueError(f"{chain} chain is not present in this pose")
    mask=pose.chain_dict[chain]
    return masked_pose(pose, mask)

def masked_pose(pose: Pose, mask: torch.Tensor) -> Pose:
    """
    Takes a pose and a mask, and returns a new pose with the mask applied
    """
    attributes = ['xyz', 'seq', 'atom_mask', 'hotspots', 'idx']
    split_attrs = {attr: getattr(pose, attr)[mask] for attr in attributes}
    cdrs=CDR(**{ch: getattr(pose.cdrs, ch)[mask] for ch in pose.cdrs.cdr_names()})
    chain_dict={ch: pose.chain_dict[ch][mask] for ch in ['H', 'L', 'T']}
    return Pose(**split_attrs, cdrs=cdrs, chain_dict=chain_dict)

def pose_from_remarked(pdb_path: str) -> Pose:
    """
    Builds a pose object from a pdb file in HLT remarked format
    """
    pdbf=parsers.parse_HLT_remarked(pdb_path)
    return parsed_to_pose(pdbf)

def pose_from_pdblines(pdblines: list[str]) -> Pose:
    """
    Builds a pose from pre-parsed pdblines (i.e. from a quiver file)
    """
    pdbf=parsers.parse_HLT_lines(pdblines)
    return parsed_to_pose(pdbf)

def parsed_to_pose(pdbf: Dotdict) -> Pose:
    """
    Takes a parsed pdb object and returns a Pose object
    """
    if not set([i[0] for i in pdbf.pdb_idx]).issubset(['H','L','T']):
        raise ValueError("There must only be H, L and T chains in the pdb file")
    chain_dict=masks_from_pdb_idx(pdbf.pdb_idx)
    idx=torch.Tensor([int(i[1]) for i in pdbf.pdb_idx])
    cdrs=CDR(**pdbf.cdr_masks)
    return Pose(xyz=pdbf.xyz, seq=pdbf.seq, atom_mask=pdbf.atom_mask, cdrs=cdrs, idx=idx, chain_dict=chain_dict)

def pose_from_RF_output(output: dict, input_pose: Pose) -> Pose:
    """
    Builds a pose from the output of the RF
    """
    return Pose(xyz=output['xyz_prev'][0], seq=input_pose.seq, atom_mask=input_pose.atom_mask, cdrs=input_pose.cdrs, idx=input_pose.idx, chain_dict=input_pose.chain_dict)

def masks_from_pdb_idx(pdb_idx: list) -> dict[str, torch.Tensor]:
    """
    Makes a dictionary of boolean masks from a pdb_idx list
    """
    return OrderedDict({ch: torch.Tensor([i[0] == ch for i in pdb_idx]).bool() for ch in ['H','L','T']})

def pose_to_remarked_pdblines(pose: Pose, Bfacts=None, metrics=None) -> list:
    """
    Writes a pose object to a pdb file
    pose: Pose object
    outf: Path to output file
    """
    pdblines=pose.pdblines(Bfacts=Bfacts)

    # add loops
    for loop in pose.cdrs.cdr_names():
        for resi in pose.idx[getattr(pose.cdrs, loop)]:
            pdblines.append("REMARK PDBinfo-LABEL:%5s %s\n"%(int(resi.item()), loop))
    
    # add metrics, if provided
    if metrics is not None:
        for k,v in metrics.items():
            if torch.is_tensor(v):
                v=v.mean()
            pdblines.append(f"SCORE {k}: {v:.2f}\n")

    return pdblines

def pdblines_to_pdb(pdblines: list, outf: str) -> None:
    """
    Writes a list of pdblines to a pdb file
    """
    outf=f'{outf}.pdb' if outf[-4:] != '.pdb' else outf
    with open(outf,'w') as f:
        f.writelines(pdblines)

def pose_generator(conf: HydraConfig) -> tuple[Pose, str]:
    """
    Generator that yields a pose object for each of the inputs drawn from the config
    Input:
        conf: The input_file config from the hydra config
            conf.input.quiver: path to quiver file; generator outputs each pose in turn
            conf.input.pdb: path to pdb file; generator outputs a single pose
            conf.input.pdb_dir: path to directory of pdb files; generator outputs each pose in turn
    Output:
        Pose object
        tag associated with the Pose
    """
    if sum(1 for value in conf.input.values() if value is not None) != 1:
        raise ValueError("Must provide exactly one of quiver, remarked_pdb, or pdb_dir")
    
    if conf.input.quiver is not None:
        quiver=Quiver(conf.quiver, mode='r')
        tags=quiver.get_tags()

        for tag in tags:
            pdblines=quiver.get_pdblines(tag)
            yield pose_from_pdblines(pdblines), tag

    elif conf.input.pdb is not None:
        if not util.check_if_remarked(conf.input.pdb):
            raise NotImplementedError(f"Only HLT remarked pdb files are currently supported. {conf.input.pdb} is not remarked")
        yield pose_from_remarked(conf.input.pdb), os.path.splitext(os.path.basename(conf.input.pdb))[0]
    
    elif conf.input.pdb_dir is not None:
        for file in glob.glob(os.path.join(conf.input.pdb_dir, '*.pdb')):
            if not util.check_if_remarked(file):
                raise NotImplementedError(f"Only HLT remarked pdb files are currently supported. {file} is not remarked")
            yield pose_from_remarked(file), os.path.splitext(os.path.basename(file))[0]
