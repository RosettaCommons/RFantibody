from __future__ import annotations

import glob
import os
import subprocess
import uuid

import torch

from rfantibody.rf2.modules import parsers
from rfantibody.rf2.modules import pose_util as pu

script_dir=os.path.dirname(os.path.abspath(__file__))
usalign = f'{script_dir}/../../../../include/USalign/USalign'

def tmalign_to_subset(pose1: "Pose", pose2: "Pose", subset: str) -> None:
    """
    Uses USalign implementation of TMalign to align the target of pose1 onto the target of pose2
    """
    if subset not in ['framework','target']:
        raise ValueError(f'subset must be "framework" or "target", not {subset}')

    if any([pose1.target_length == 0 or pose2.target_length == 0]):
        return None
    
    output1, subset_mask = dump_temporary_subset(pose1, subset=subset)
    output2, _ = dump_temporary_subset(pose2, subset=subset)

    # Run USalign, and capture the output
    tag = f'tmp_{uuid.uuid4()}'
    retlines = subprocess.run([usalign, output1, output2, '-o', tag], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')

    score = max(float(l.split(' ')[1].strip()) for l in retlines if l.startswith('TM-score='))
    #if score < 0.8:
    #    raise Exception(f'TM-score of {subset} alignment is too low for reliable alignment: {score:.3f}')
    print(f'TM-score of {subset} alignment: {score:.3f}')
    os.remove(output1)
    os.remove(output2)
    
    # get transform matrix
    with open(f'{tag}.pdb', 'r') as file:
        aligned_pdblines = [line for line in file if line.startswith('ATOM')]
    _, _, aligned_subset_xyz=parsers.parse_pdblines(aligned_pdblines)
    remove_temporary_files(tag)
    align_pose_to_subset(pose1, aligned_subset_xyz.to(pose1.xyz.device), subset_mask)

def dump_temporary_subset(pose: "Pose", subset: str) -> tuple[str, torch.Tensor]:
    """
    Dumps target to file
    Outputs the unique identified tag for that dumped pdb, and the mask corresponding to the subset
    """
    output=f'tmp_{uuid.uuid4()}.pdb'
    if subset == 'framework':
        subset_mask=pose.framework_mask
    elif subset == 'target':
        subset_mask=pose.target_mask
    else:
        raise ValueError(f'subset must be "framework" or "target", not {subset}')
    masked_pose=pu.masked_pose(pose, subset_mask)
    pu.pdblines_to_pdb(pu.pose_to_remarked_pdblines(masked_pose), output)
    return output, subset_mask

def align_pose_to_subset(pose_complex: "Pose", aligned_subset: torch.Tensor, mask: torch.Tensor) -> None:
    """
    Aligns a Pose (target + antibodies) to a subset of pre-aligned residues.
    This function aligns e.g. the whole complex to a pre-aligned target using SVD for optimal rotation.
    Or can align to an aligned framework.
    Inputs:
        pose_complex: Pose object, unaligned
        aligned_subset: xyz tensor parsed from US align output, of a subset of residues in the Pose object
        mask: boolean mask relating residues in the Pose to the aligned subset.
    """
    if not mask.sum() == len(aligned_subset):
        raise ValueError("Mask for the pose does not correctly index to the aligned xyz coordinates")
    # Extract coordinates and compute means
    xyz1 = pose_complex.xyz[:, 1].clone()
    xyz2 = aligned_subset[:, 1].clone()
    xyz1_mean = xyz1[mask].mean(dim=0)
    xyz2_mean = xyz2.mean(dim=0)
    
    # Center the coordinates around the mean
    xyz1_centered = xyz1 - xyz1_mean
    xyz2_centered = xyz2 - xyz2_mean
    
    # Calculate the covariance matrix for SVD
    C = xyz1_centered[mask].T @ xyz2_centered

    # Compute optimal rotation matrix using SVD
    U, S, Vt = torch.linalg.svd(C)  # Vt is V transposed

    # Ensure right-handed coordinate system in the rotation matrix
    d = torch.ones(3, device=pose_complex.xyz.device, dtype=pose_complex.xyz.dtype)
    d[-1] = torch.sign(torch.linalg.det(U) * torch.linalg.det(Vt))
    D = torch.diag(d)

    # Calculate the rotation matrix
    R = U @ D @ Vt

    # Apply transformation to the full set of coordinates
    xyz_full = pose_complex.xyz.clone()
    xyz_full -= xyz1_mean  # Center the complex
    xyz_full = xyz_full @ R  # Rotate
    xyz_full += xyz2_mean  # Translate back
    pose_complex.xyz = xyz_full

def calc_prealigned_rmsd(pose1: "Pose", pose2: "Pose", mask: torch.Tensor[bool]) -> float:
    """
    Takes two prealigned poses and calculates RMSD in the mask region
    """
    device=mask.device
    xyz1 = pose1.xyz[:, 1].clone().to(device)
    xyz2 = pose2.xyz[:, 1].clone().to(device)
    xyz1 = xyz1[mask]
    xyz2 = xyz2[mask]
    return torch.sqrt(((xyz1 - xyz2) ** 2).sum() / mask.sum()).item()

def remove_temporary_files(tag: str) -> None:
    """
    Removes the output files from USalign
    """
    for fp in glob.glob(f'./{tag}*'):
        os.remove(fp)
