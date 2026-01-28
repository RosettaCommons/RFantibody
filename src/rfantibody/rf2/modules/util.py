from __future__ import annotations

import functools
import glob
import os

import numpy as np
import torch

import rfantibody.rf2.network.util as RF2_util
from rfantibody.rf2.network.chemical import INIT_CRDS, aa2long, num2aa
from rfantibody.util.quiver import Quiver


def get_pdblines(pose: "Pose", Bfacts=None) -> list:
    """
    Get pdblines from Pose object
    """
    pdblines=[]
    ctr=1
    seq=pose.seq.cpu()
    atoms=pose.xyz.cpu()
    pdb_idx=pose.pdb_idx
    if Bfacts==None:
        Bfacts=torch.zeros(pose.length)
    for i,s in enumerate(seq):
        natoms = atoms.shape[-2]
        if (natoms!=3 and natoms!=14 and natoms!=27):
            raise ValueError("Wrong number of atoms in xyz coordinates (must be 3, 14 or 27)")
        
        atms = aa2long[s]
        # his protonation state hack
        if (s==8 and torch.linalg.norm( atoms[i,9,:]-atoms[i,5,:] ) < 1.7):
            atms = (
                " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                  None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                " HD1",  None,  None,  None,  None,  None,  None) # his_d

        for j,atm_j in enumerate(atms):
            if (j<natoms and atm_j is not None and not torch.isnan(atoms[i,j,:]).any()):
                pdblines.append("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s],
                    pdb_idx[i][0], pdb_idx[i][1], atoms[i,j,0], atoms[i,j,1], atoms[i,j,2],
                    1.0, Bfacts[i] ) )
                ctr += 1
    pdblines.append("TER\n")
    return pdblines

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

def random_round(number):
    """
    Randomly rounds up or down
    """
    return np.floor(number).astype(int) + np.random.randint(0, 2)
       
def mask_hotspots(pose: "Pose", show_proportion: float) -> torch.Tensor[bool]:
    """
    Takes a pose object, grabs the hotspots and masks them according to the config.
    """
    if not (0 <= show_proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1")
    hotspots=torch.clone(pose.hotspots)

    if hotspots.sum() == 0:
        # If there are no hotspots, return the original mask
        print("No interface residues found, not using hotspots")
        return hotspots

    true_indices = np.where(hotspots)[0]
    # Calculate how many to HIDE (inverse of show_proportion)
    hide_proportion = 1 - show_proportion
    num_to_hide = min(random_round(hide_proportion * len(true_indices)), len(true_indices))
    if num_to_hide > 0:
        indices_to_convert = np.random.choice(
            true_indices,
            size=num_to_hide,
            replace=False,
        )
        hotspots[indices_to_convert] = False
    return hotspots

def get_init_xyz(xyz_t, mask_t, random_noise=5.0):
    """
    NRB moved this out of the class to allow for more broad access
    by other parts of the code

    Replaces old function for getting initial coordinates
    Input:
        - xyz_t: torch.tensor. Initial coordinates, with nans where coordinates
            are missing [T, L, 27, 3]
    """
    T,L = xyz_t.shape[:2]
    xyz_t_init=INIT_CRDS.reshape(1,1,27,3).repeat(T, L, 1, 1) + torch.rand(T, L, 1, 3)*random_noise
    xyz_t=torch.where(xyz_t.isnan(), xyz_t_init, xyz_t)
    for i in range(xyz_t.shape[0]):
        xyz_t[i] = RF2_util.center_and_realign_missing(xyz_t[i], mask_t[i])
    return xyz_t

def check_for_nans(func):
    """
    Decorator to check for NaNs in the output of a function
    """
    @functools.wraps(func)
    def wrapper_check_for_nans(*args, **kwargs):
        result = func(*args, **kwargs)

        # Function to check for NaNs in a tensor
        def check_tensor(tensor):
            if torch.isnan(tensor).any():
                raise ValueError(f"NaN values detected in output of {func.__name__}")

        # Check if the result is a single tensor
        if isinstance(result, torch.Tensor):
            check_tensor(result)
        # Check if the result is a tuple or list of tensors
        elif isinstance(result, (tuple, list)):
            for tensor in result:
                if isinstance(tensor, torch.Tensor):
                    check_tensor(tensor)
        else:
            raise ValueError("Function did not return a tensor or a tuple/list of tensors")

        return result

    return wrapper_check_for_nans

def to_device(func):
    @functools.wraps(func)
    def wrapper_to_device(*args, **kwargs):
        # Extract the device from kwargs or set default
        device = kwargs.pop('device', 'cpu')

        def move_to_device(item):
            # Move tensors to the specified device
            if isinstance(item, torch.Tensor):
                return item.to(device, non_blocking=True)
            # Recursively apply to items in lists or tuples
            elif isinstance(item, (list, tuple)):
                return type(item)(move_to_device(x) for x in item)
            # Recursively apply to values in dictionaries
            elif isinstance(item, dict):
                return {k: move_to_device(v) for k, v in item.items()}
            # Return the item unchanged if not a tensor, list, tuple, or dict
            return item

        result = func(*args, **kwargs)
        return move_to_device(result)

    return wrapper_to_device

def check_if_remarked(pdb_path: str) -> bool:
    """
    Checks if a pdb file is in remarked_HLT format
    """
    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('REMARK'):
                return True
    return False

def get_done_list(conf: HydraConfig) -> list:
    """
    Parses config to work out output format.
    Appends done tags to the list
    """
    if conf.output.pdb_dir is not None:
        strip=lambda x: os.path.splitext(os.path.basename(x))[0]
        return [strip(i)[:-5] for i in glob.glob(f'{conf.output.pdb_dir}/*_best.pdb')]
    elif conf.output.quiver is not None:
        qv=Quiver(f'{conf.output.quiver}', mode='r')
        return qv.tags
    else:
        raise ValueError('Must specify output.pdb_dir or output.quiver')

def get_cdr_hotspots(pose: "Pose", cutoff=8) -> torch.Tensor[bool]:
    """
    Function to get hotspots on the target chain that are
    within some distance of the target (i.e. are contacting)
    inputs:
        Pose, with:
        - xyz: Concatenated target & binder coordinates [L, 14/27, 3]
        - atom_mask: Atom mask. True where atom is present, False 
            otherwise [L, 14/27]
        - target_mask: True where target, False where binder [L]
        - loop_mask: True where CDR loop, False where not [L]
        cutoff:
            minimum Cb-Cb distance to be a "hotspot"
    output:
        - hotspot_mask: True where hotspot (on target), False where not [L]
    """
    if pose.target_length == 0:
        return torch.zeros(pose.length, dtype=torch.bool)
    if pose.cdrs.mask_1d.sum() == 0:
        raise ValueError("Pose has no CDR residues")
    
    # generate Cb
    xyz=pose.xyz.clone()
    Cb = RF2_util.get_Cb(xyz[None])[0]
    xyz[~pose.atom_mask[:,4],4] = Cb[~pose.atom_mask[:,4]]

    # get distances
    dist=torch.cdist(xyz[pose.target_mask,4], xyz[pose.cdrs.mask_1d,4])
    dist=torch.min(dist, dim=1).values

    hotspots = torch.zeros(pose.length, dtype=torch.bool)
    # torch.where doesn't work on cpu
    convert=lambda a: torch.from_numpy(np.array(a)).bool()
    hotspots[pose.target_mask] = convert([True if i < cutoff else False for i in dist])
    return hotspots
