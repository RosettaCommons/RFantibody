import torch

from rfantibody.rf2.modules.util import Dotdict
from rfantibody.rf2.network.chemical import INIT_CRDS, aa2long, aa2num

split_remark = lambda line: (line.split()[3][0], int(line.split()[2]))

def parse_HLT_remarked(pdb_path: str) -> dict:
    """
    New parser that also reads lines prefixed with 'REMARK PDBinfo-LABEL'
    and gets the cdr_bool from this
    Inputs:
        path to pdb file.
        Must be in H,L,T format.
        No other chains
    Outputs:
        pdb object
    """
    with open(pdb_path,'r') as f:
        lines=f.readlines()
    return parse_HLT_lines(lines)

def parse_HLT_lines(lines: list[str]) -> Dotdict:
    seq, pdb_idx, xyz = parse_pdblines(lines)
    if not all([i[0] in ['H','L','T'] for i in pdb_idx]):
        raise ValueError("Not all chains are H,L,T")
    cdr_masks=get_cdr_masks_from_remarks(pdb_idx, lines)
    atom_mask = torch.logical_not(torch.isnan(xyz[...,0]))

    out = Dotdict({'xyz':xyz, # cartesian coordinates, [Lx14]
            'atom_mask':atom_mask, # mask showing which atoms are present in the PDB file, [Lx27]
            'idx':torch.tensor([i[1] for i in pdb_idx], dtype=torch.int64), # residue numbers in the PDB file, [L]
            'seq':torch.tensor(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
            'cdr_masks': cdr_masks, # dict of loop masks, [L]
    })

    fix_any_duplicates(out)

    # Reorder chains to T-H-L (Target first, to match rf_antibody)
    reorder_chains_to_THL(out)

    return out

def parse_pdblines(lines: list[str]) -> tuple[torch.Tensor, list, torch.Tensor]:
    """
    Takes lines beginning with ATOM and parses to res, seq, pdb_idx, xyz
    """
    res = [(l[22:26], l[17:20]) for l in lines if l[:4] == "ATOM" and l[12:16].strip() == "CA"]
    seq = torch.tensor([aa2num[r[1]] if r[1] in aa2num else 20 for r in res])

    # Generating pdb_idx for indexing
    pdb_idx = [(l[21:22].strip(), int(l[22:26].strip())) for l in lines if l[:4] == "ATOM" and l[12:16].strip() == "CA"]

    # Creating a tensor for XYZ coordinates
    xyz = torch.full((len(res), 27, 3), float('nan'), dtype=torch.float32)

    # A dictionary to quickly find the index in pdb_idx (for efficiency)
    pdb_idx_lookup = {k: i for i, k in enumerate(pdb_idx)}

    for l in lines:
        if l[:4] == "ATOM":
            chain, resNo, atom, aa = l[21:22].strip(), int(l[22:26]), ' ' + l[12:16].strip().ljust(3), l[17:20]
            if (chain, resNo) in pdb_idx_lookup:
                idx = pdb_idx_lookup[(chain, resNo)]
                if aa in aa2num:  # Ensure aa is known
                    for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
                        if tgtatm is not None and tgtatm.strip() == atom.strip():  # Matching atom name
                            xyz[idx, i_atm, :] = torch.tensor([float(l[30:38]), float(l[38:46]), float(l[46:54])], dtype=torch.float32)
                            break

    return seq, pdb_idx, xyz

def get_cdr_masks_from_remarks(pdb_idx: list, lines: list[str]) -> dict:
    """
    Parses PDB remarks to boolean masks
    """
    cdr_pdb_idx=[]
    cdr_names = ['H1','H2','H3','L1','L2','L3']
    cdr_masks = {loop: torch.zeros(len(pdb_idx)).bool() for loop in cdr_names}
    for l in lines:
        if l.startswith('REMARK PDBinfo-LABEL'):
            l=l.strip()
            cdr_pdb_idx.append(split_remark(l))
            loop = l[27:29].upper() 
            if loop in cdr_names: 
                resi = int(l[21:26]) - 1 # Loop residues in HLT are 1-indexed
                cdr_masks[loop][resi] = True
    if torch.any(torch.stack(list(cdr_masks.values())), dim=0).sum() != len(cdr_pdb_idx):
        raise ValueError("Not all cdr residues found in file. Remark indexing is bad")
    return cdr_masks

def fix_any_duplicates(pdb: Dotdict) -> None:
    """
    In place fixing any duplicate residues.
    Occasionally there are duplicate residues in a pdb file.
    Only keep the first.
    """
    if len(pdb['pdb_idx']) == len(set(pdb['pdb_idx'])):
        return
    mask=torch.ones(len(pdb['pdb_idx']), dtype=bool)
    for i in range(len(pdb['pdb_idx'])-1):
        if pdb['pdb_idx'][i] == pdb['pdb_idx'][i+1]:
            mask[i+1]=False
    pdb['pdb_idx'] = [i for i,m in zip(pdb['pdb_idx'], mask) if m]
    pdb['seq'] = pdb['seq'][mask]
    pdb['atom_mask'] = pdb['atom_mask'][mask]
    pdb['xyz'] = pdb['xyz'][mask]
    pdb['idx'] = pdb['idx'][mask]
    pdb['cdr_masks'] = {k:v[mask] for k,v in pdb['cdr_masks'].items()}

def reorder_chains_to_THL(pdb: Dotdict) -> None:
    """
    Reorders chains from PDB file order (typically H-L-T) to T-H-L (Target first).
    This matches rf_antibody's chain ordering convention.
    Modifies the pdb dict in place.
    """
    # Find indices for each chain
    T_indices = [i for i, (chain, _) in enumerate(pdb['pdb_idx']) if chain == 'T']
    H_indices = [i for i, (chain, _) in enumerate(pdb['pdb_idx']) if chain == 'H']
    L_indices = [i for i, (chain, _) in enumerate(pdb['pdb_idx']) if chain == 'L']

    # If no target chain present, no reordering needed (pure antibody)
    if len(T_indices) == 0:
        return

    # Create new ordering: T, H, L
    new_order = T_indices + H_indices + L_indices
    new_order_tensor = torch.tensor(new_order, dtype=torch.long)

    # Reorder all fields
    pdb['pdb_idx'] = [pdb['pdb_idx'][i] for i in new_order]
    pdb['seq'] = pdb['seq'][new_order_tensor]
    pdb['xyz'] = pdb['xyz'][new_order_tensor]
    pdb['atom_mask'] = pdb['atom_mask'][new_order_tensor]
    pdb['idx'] = pdb['idx'][new_order_tensor]
    pdb['cdr_masks'] = {k: v[new_order_tensor] for k, v in pdb['cdr_masks'].items()}

