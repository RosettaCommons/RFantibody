from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from rfantibody.rf2.modules import util
from rfantibody.rf2.modules.util import Dotdict, check_for_nans, to_device
from rfantibody.rf2.network import kinematics
from rfantibody.rf2.network.util_module import XYZConverter


class Preprocess:
    """
    Class for handling conversion of a pose to RF2 inputs.
    """

    def __init__(self, pose_to_input: callable, conf: HydraConfig) -> None:
        """
        pose_to_input: function that takes a Pose object and returns a Dotdict dictionary of the core intermediate values
        """
        self.pose_to_input=pose_to_input
        self.conf=conf
        self.xyz_converter = XYZConverter()

    @to_device
    def __call__(self, pose: "Pose") -> tuple: 
        """
        Preprocessing function for RF2

        Inputs:

            Pose object. This is used to generate intermediate inputs.
            This is useful for separating training/inference.

            Inputs (Dotdict) containing:

                seq: torch.tensor [L]
                    integer representation of sequence

                xyz_t: torch.tensor [T, L, 27, 3]. 

                    NB T=1 or 2 (antibody alone or antibody and target)

                idx: torch.tensor [L]. 200 residue offset between chains

                atom_mask: torch.tensor [T, L, 14]    

                xyz_true: torch.tensor [L, 27, 3]

        Outputs: 
            RF2 network_inputs:

                msa_masked: torch.tensor [1, 1, L, 48]

                msa_full: torch.tensor [1, 1, L, 25]

                seq: torch.tensor [1,L,22]

                xyz_prev: torch.tensor [L, 27, 3]
                    the first template structure
                    In our case, this is always the target

                idx_pdb: torch.tensor [L]
                
                t1d: torch.tensor [T, L, 23]
                    22 + 1, added for the hotspot layer

                t2d: torch.tensor [T, L, L, 44]
                    made by xyz_to_t2d function (kinematics.py)

                xyz_t: torch.tensor [T, L, 27, 3]
                
                    xyz coords of the templates

                alpha_t: torch.tensor [T,L,10*3]
            
                mask_t: torch.tensor [T, L, 27] atom_mask
            
                mask_prev: [1, L, 27] atom_mask[0]
        """

        inputs=self.pose_to_input(pose, self.conf)

        check_inputs(inputs)

        T,L = inputs.xyz_t.shape[:2]

        # xyzs
        seed = getattr(self.conf.inference, 'seed', None) if hasattr(self.conf, 'inference') else None
        xyz_t, xyz_true, xyz_prev=get_xyzs(inputs, seed=seed)
        
        # seq/MSA
        seq = inputs.seq
        msa_masked, msa_full, mask_msa, msa = make_msa(seq, L)

        # t1d
        t1d=make_t1d(pose, seq[None], inputs.hotspots, L, T)

        # t2d
        t2d, mask_t_2d = make_t2d(inputs, xyz_t, pose)

        # alpha_t
        alpha_t = make_alpha_t(xyz_t, seq[None], t1d, inputs.mask_t, L, self.xyz_converter)

        # same_chain
        same_chain=pose.same_chain

        mask_prev, mask_recycle = make_template_masks(inputs, same_chain)

        network_input = {}
        network_input['msa_latent'] = msa_masked
        network_input['msa_full'] = msa_full
        network_input['seq'] = seq
        network_input['idx'] = inputs.idx
        network_input['t1d'] = t1d
        network_input['t2d'] = t2d
        network_input['xyz_t'] = xyz_t
        network_input['alpha_t'] = alpha_t
        network_input['mask_t'] = mask_t_2d
        network_input['same_chain'] = pose.same_chain

        return network_input, xyz_prev, mask_recycle, xyz_true, inputs.mask_true, msa, mask_msa, inputs.unclamp, inputs.negative, inputs.idx

def check_inputs(inputs: Dotdict) -> None:
    """
    Checks inputs
    """
    if torch.max(inputs.seq) > 21:
        raise ValueError(f'token in sequence is > 21, max is {torch.max(inputs.seq)}')

@check_for_nans
def get_xyzs(inputs: Dotdict, seed: int = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gets xyzs from inputs

    Args:
        inputs: Input dictionary containing xyz_t, mask_t, xyz_true
        seed: Random seed for xyz initialization (optional, for deterministic mode)
    """
    # Set seed right before xyz initialization to ensure reproducibility
    if seed is not None:
        torch.manual_seed(seed)
    xyz_t=util.get_init_xyz(inputs.xyz_t, inputs.mask_t)
    xyz_true = torch.nan_to_num(inputs.xyz_true)
    xyz_prev = xyz_t[0].clone()
    return xyz_t, xyz_true, xyz_prev

@check_for_nans    
def make_msa(seq: torch.Tensor, L: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make msa inputs from single sequence (not true MSA)
    """
    seq_onehot=F.one_hot(seq, num_classes=21) # [1, L, 21]
    # msa_masked
    msa_masked = torch.zeros((1,L,48))
    msa_masked[:,:,:21] = seq_onehot
    msa_masked[:,:,22:43] = seq_onehot

    # msa_full
    msa_full = torch.zeros((1,L,25))
    msa_full[:,:,:21] = seq_onehot

    # mask_msa
    mask_msa = torch.ones_like(msa_full[:,:,0]) # (T,L)

    # msa
    msa = seq.clone() 

    return msa_masked, msa_full, mask_msa, msa

@check_for_nans
def make_t1d(pose: "Pose", seq: torch.Tensor, hotspots: torch.Tensor, L: int, T: int) -> torch.Tensor:
    """
    Makes a t1d.
        Masks antibody sequence and sets confidence to 0
        Sets confidence to 1 for target sequence
        Adds hotspots
    """
    seq_onehot=F.one_hot(seq, num_classes=21) # [1, L, 21]
    t1d = torch.zeros((T, L, 23))
    t1d[:,:,:21] = seq_onehot.repeat(T,1,1) # [T, L, 23]

    # Mask antibody part
    t1d[:,pose.antibody_mask,:21] = 0
    t1d[:,pose.antibody_mask,21] = 1
    
    # template confidence
    t1d[:,pose.target_mask,21] = 1
    t1d[:,pose.antibody_mask,21]=0

    # add pre-masked hotspots
    t1d[0,:,22]=hotspots

    return t1d

@check_for_nans
def make_t2d(inputs: Dotdict, xyz_t: torch.Tensor, pose: "Pose") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Makes t2d, and the corresponding mask_t_2d
    """
    mask_t=inputs.mask_t
    mask_t_2d = mask_t[None,:,:,:3].all(dim=-1) # (B, T, L)
    mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
    assert mask_t_2d.shape[1] == 1

    # With T-H-L ordering, antibody is at the END, so mask last antibody_length residues
    # (Previously masked first antibody_length, which was correct for H-L-T ordering)
    mask_t_2d[0,0,-pose.antibody_length:,:] = False
    mask_t_2d[0,0,:,-pose.antibody_length:] = False

    t2d = kinematics.xyz_to_t2d(xyz_t.unsqueeze(0), mask_t_2d) # [B,T,L,L,44], this function requires a batch dimension
    t2d = t2d.squeeze(0) # [T,L,L,44]
    return t2d, mask_t_2d

@check_for_nans
def make_alpha_t(xyz_t: torch.Tensor, seq: torch.Tensor, t1d: torch.Tensor, mask_t: torch.Tensor, L: int, xyz_converter: XYZConverter) -> torch.Tensor:
    """
    Generates alpha_t
    """
    ### alpha_t ###
    ###############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = xyz_converter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(-1,L,10,2)
    alpha_mask = alpha_mask.reshape(-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 30)
    return alpha_t

@check_for_nans
def make_template_masks(inputs: Dotdict, same_chain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates mask_prev, mask_recycle
    """
    mask_prev    = inputs.mask_t[:1].clone() # (1,L,27)
    mask_recycle = mask_prev[:,:,:3].all(dim=-1)
    mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B,L,L)
    mask_recycle = same_chain.float()*mask_recycle.float()
    mask_recycle = mask_recycle.squeeze(0) # (L,L)
    return mask_prev, mask_recycle

def pose_to_inference_RFinput(pose: "Pose", conf: HydraConfig) -> Dotdict:
    """
    Generates input to RF from a Pose object.
    The idea is that anything config-related will be done here.
    """
    seq=pose.seq
    xyz_true=torch.clone(pose.xyz)
    mask_true=torch.clone(pose.atom_mask)
    xyz_t=torch.clone(pose.xyz)[None]
    mask_t=torch.clone(pose.atom_mask)[None]

    # mask the antibody
    xyz_t[:,pose.antibody_mask] = float('nan')
    mask_t[:,pose.antibody_mask] = False
    # mask target sidechains
    xyz_t[:,pose.target_mask,3:] = float('nan')
    mask_t[:,pose.target_mask,3:] = False

    # idx
    idx=make_RF_idx(pose)

    # hotspot masking
    hotspots=util.mask_hotspots(pose, conf.inference.hotspot_show_proportion)

    return Dotdict({
        'seq':seq,
        'xyz_t':xyz_t,
        'mask_t':mask_t,
        'idx': idx,
        'hotspots':hotspots,
        'xyz_true':xyz_true,
        'mask_true':mask_true,
        'negative': False,
        'unclamp': False
        })

def make_RF_idx(pose) -> torch.Tensor:
    """
    Makes the idx feature to RF, adding a 200 residue offset between chains.
    Convention (matching rf_antibody):
    - Target (T) chain: starts at 0
    - Antibody chains (H, L): always get 200-residue buffer, even when first

    Examples:
    - With target (T=251, H=121, L=106): T:0-250, H:451-571, L:772-877
    - Without target (H=121, L=106): H:200-320, L:521-626
    """
    idx=[]
    lastidx = -1  # Start at -1 so first antibody chain offset calculation works

    for i, ch in enumerate(pose.chains_present):
        chain_mask = pose.chain_dict[ch]
        chain_length = chain_mask.sum().item()

        # Create sequential indices starting from 0 for this chain
        chain_idx = torch.arange(chain_length, dtype=torch.int64)

        # Add offset based on chain type:
        # - Target (T) starts at 0
        # - Antibody chains (H, L) always get previous_end + 1 + 200
        if ch == 'T':
            # Target starts at 0, no offset
            pass
        else:
            # Antibody chains: add offset from previous chain + 200 buffer
            chain_idx = chain_idx + (lastidx + 1) + 200

        lastidx = chain_idx[-1].item()
        idx.append(chain_idx)

    return torch.cat(idx)
