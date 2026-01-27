from __future__ import annotations

import biotite.structure as struc
import torch


def align_to_subset(pose1: "Pose", pose2: "Pose", subset: str) -> bool:
    """
    Aligns pose1 to pose2 based on a subset (target or framework).
    Uses Kabsch superposition on Cα atoms if lengths match, otherwise skips alignment.

    Args:
        pose1: Pose to be aligned (modified in place)
        pose2: Reference pose
        subset: Either 'target' or 'framework'

    Returns:
        bool: True if alignment was performed, False if skipped due to length mismatch
    """
    if subset not in ['framework','target']:
        raise ValueError(f'subset must be "framework" or "target", not {subset}')

    # Get the subset mask
    if subset == 'framework':
        subset_mask1 = pose1.framework_mask
        subset_mask2 = pose2.framework_mask
    elif subset == 'target':
        subset_mask1 = pose1.target_mask
        subset_mask2 = pose2.target_mask

    # Check if no target/framework exists
    if subset_mask1.sum() == 0 or subset_mask2.sum() == 0:
        return False

    # Check if lengths match
    if subset_mask1.sum() != subset_mask2.sum():
        print(f'Warning: {subset} lengths differ between poses ({subset_mask1.sum().item()} vs {subset_mask2.sum().item()}). Skipping RMSD calculations.')
        return False

    # Convert to numpy for biotite
    # Extract all atoms (L, n_atoms, 3)
    xyz1_full = pose1.xyz.cpu().numpy()
    xyz2_full = pose2.xyz.cpu().numpy()

    # Extract Cα coordinates for alignment (L, 3)
    xyz1_ca = xyz1_full[:, 1, :]  # Cα is index 1
    xyz2_ca = xyz2_full[:, 1, :]

    # Get subset coordinates for alignment
    mask_np = subset_mask1.cpu().numpy()
    xyz1_subset = xyz1_ca[mask_np]
    xyz2_subset = xyz2_ca[mask_np]

    # Use biotite to get the superimposition transformation
    # superimpose returns (fitted, transformation)
    _, transformation = struc.superimpose(xyz1_subset, xyz2_subset)

    # Apply transformation to all atoms in pose1
    # Reshape for biotite: (L * n_atoms, 3)
    n_res, n_atoms, _ = xyz1_full.shape
    xyz1_flat = xyz1_full.reshape(-1, 3)

    # Apply transformation using the transformation object's apply method
    xyz1_aligned = transformation.apply(xyz1_flat)

    # Reshape back and convert to torch
    xyz1_aligned = xyz1_aligned.reshape(n_res, n_atoms, 3)
    pose1.xyz = torch.from_numpy(xyz1_aligned).to(pose1.xyz.device, dtype=pose1.xyz.dtype)

    return True


def calc_prealigned_rmsd(pose1: "Pose", pose2: "Pose", mask: torch.Tensor[bool]) -> float:
    """
    Calculates RMSD between two pre-aligned poses in the mask region.
    Uses Cα atoms for RMSD calculation.

    Args:
        pose1: First pose
        pose2: Second pose
        mask: Boolean mask indicating which residues to include

    Returns:
        float: RMSD in Angstroms
    """
    device = mask.device
    xyz1 = pose1.xyz[:, 1].clone().to(device)  # Cα atoms
    xyz2 = pose2.xyz[:, 1].clone().to(device)  # Cα atoms
    xyz1 = xyz1[mask]
    xyz2 = xyz2[mask]
    return torch.sqrt(((xyz1 - xyz2) ** 2).sum() / mask.sum()).item()
