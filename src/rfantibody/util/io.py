from typing import List, Optional

import numpy as np
import torch

from rfantibody.rfdiffusion.chemical import aa2long, aa2num


def stamp_pdbline(
    prefix: str,
    ctr: int,
    atom_name: str,
    residue_name: str,
    chain: str,
    residue_idx: int,
    x_coord: float,
    y_coord: float,
    z_coord: float,
    occupancy: float,
    b_factor: float,
) -> str:
    return "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
        prefix,
        ctr,
        atom_name,
        residue_name,
        chain,
        residue_idx,
        x_coord,
        y_coord,
        z_coord,
        occupancy,
        b_factor,
    )


def ab_write_pdblines(
    atoms: np.ndarray,
    seq: np.ndarray,
    chain_idx: np.ndarray,
    num2aa: List[str],
    idx_pdb: Optional[np.ndarray] = None,
    bfacts: Optional[np.ndarray] = None,
    loop_map: dict[str, List[int]] = {},
) -> List[str]:
    """
    Given a set of atomic coordinates and a sequence, generate a list of PDB lines
    describing the structure.

    Args:
        atoms:
            A [L, N, 3] tensor of atomic coordinates, where N can be 1, 3, 4, 14, or 27
        seq:
            A [L] tensor of integer amino acid residues
        chain_idx:
            A [L] tensor of chain indices
        num2aa:
            The way to convert from residue numbers to amino acid residues
        idx_pdb:
            A [L] tensor of residue indices
        bfacts:
            A [L] tensor of B-factors
        loop_map:
            A dictionary mapping loop names to lists of residue indices
    """

    ctr = 1
    if bfacts is None:
        bfacts = torch.zeros(atoms.shape[0])
    if idx_pdb is None:
        # Default to 1-indexed residue numbers
        idx_pdb = 1 + torch.arange(atoms.shape[0])

    Bfacts = np.clip(
        bfacts,
        a_min = 0,
        a_max = 1,
    )

    pdblines = []
    for i in range(seq.shape[0]):
        chain = chain_idx[i]

        # If the input is a single set of atomic coordinates, assume it is a C-alpha trace
        if (len(atoms.shape)==2):
            
            pdblines.append(
                stamp_pdbline(
                    prefix="ATOM",
                    ctr=ctr,
                    atom_name=" CA ",
                    residue_name=num2aa[seq[i]],
                    chain=chain,
                    residue_idx=idx_pdb[i],
                    x_coord=atoms[i,0],
                    y_coord=atoms[i,1],
                    z_coord=atoms[i,2],
                    occupancy=1.0,
                    b_factor=Bfacts[i],
                )
            )

            ctr += 1

        # If the input is a set of atomic coordinates with 3 atoms per residue,
        # assume it is a backbone trace
        elif atoms.shape[1]==3:
            for j,atm_j in enumerate([" N  "," CA "," C  "]):
                
                pdblines.append(
                    stamp_pdbline(
                        prefix="ATOM",
                        ctr=ctr,
                        atom_name=atm_j,
                        residue_name=num2aa[seq[i]],
                        chain=chain,
                        residue_idx=idx_pdb[i],
                        x_coord=atoms[i,j,0],
                        y_coord=atoms[i,j,1],
                        z_coord=atoms[i,j,2],
                        occupancy=1.0,
                        b_factor=Bfacts[i],
                    )
                )

                ctr += 1

        # If the input is a set of atomic coordinates with 4 atoms per residue,
        # assume it is a backbone trace with an oxygen atom
        elif atoms.shape[1]==4:
            for j,atm_j in enumerate([" N  "," CA "," C  "," O  "]):

                pdblines.append(
                    stamp_pdbline(
                        prefix="ATOM",
                        ctr=ctr,
                        atom_name=atm_j,
                        residue_name=num2aa[seq[i]],
                        chain=chain,
                        residue_idx=idx_pdb[i],
                        x_coord=atoms[i,j,0],
                        y_coord=atoms[i,j,1],
                        z_coord=atoms[i,j,2],
                        occupancy=1.0,
                        b_factor=Bfacts[i],
                    )
                )
                
                ctr += 1
            
        # Otherwise, assume the input is a full atomic tensor with either 14 or 27 atoms per residue
        else:
            natoms = atoms.shape[1]

            assert(natoms==14 or natoms==27), "Invalid number of atoms per residue, must be 14 or 27"

            atms = aa2long[aa2num[seq[i]]]

            # his prot hack
            if (aa2num[seq[i]]==8 and torch.linalg.norm( atoms[i,9,:]-atoms[i,5,:] ) < 1.7):
                atms = (
                    " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                      None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                    " HD1",  None,  None,  None,  None,  None,  None) # his_d

            for j,atm_j in enumerate(atms):
                if (j<natoms and atm_j is not None):
                    pdblines.append(
                        stamp_pdbline(
                            prefix="ATOM",
                            ctr=ctr,
                            atom_name=atm_j,
                            residue_name=seq[i],
                            chain=chain,
                            residue_idx=idx_pdb[i],
                            x_coord=atoms[i,j,0],
                            y_coord=atoms[i,j,1],
                            z_coord=atoms[i,j,2],
                            occupancy=1.0,
                            b_factor=Bfacts[i],
                        )
                    )
                    ctr += 1


    # This may or may not be necessary between the coordinates and the REMARKS
    pdblines.append("TER\n")

    # Add in labels for loop locations in the output structure
    # NB: could also add in the hotspots labels as remarks here as well
    for loop in loop_map:
        for resi in loop_map[loop]:
            pdblines.append("REMARK PDBinfo-LABEL:%5s %s\n"%(resi, loop))

    return pdblines
    

def parse_HLT_lines(
    lines: List[str]
) -> dict:
    '''
    Args:
        lines:
            A list of lines from a PDB file

    Returns:
        A dictionary containing the following keys:

            xyz:
                np.ndarray, [L, 27, 3] tensor of atomic coordinates
            mask:
                np.ndarray, [L, 27] tensor of atom masks
            idx:
                np.ndarray, [L] tensor of residue indices
            seq:
                np.ndarray, [L] tensor of amino acid residues
            pdb_idx:
                np.ndarray, [L] tensor of pdb indices
            loop_masks:
                dict[str, np.ndarray], dictionary of loop masks 
    '''

    cdr_names = ['H1','H2','H3','L1','L2','L3']

    # indices of residues observed in the structure
    res = []
    for l in lines:
        if l[:4]!="ATOM": continue

        i = 0 if l[11] == ' ' else 1
        if l[12+i:16+i].strip()=="CA":
            res.append((l[22+i:26+i].strip(),l[17+i:20+1].strip(),l[21+i]))

    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [(l[21:22].strip(), l[22:26].strip()) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    loop_masks = {loop: np.zeros(len(res)).astype(bool) for loop in cdr_names}

    # 4 BB + 23 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        # Check for lines that begin with REMARK and parse them to their loop labels
        if l[:3] == 'TER': continue
        if l[:6] == "REMARK":
            loop = l[27:29].upper() 
            if loop in cdr_names: 
                resi = int(l[21:26]) - 1 # Loop residues in HLT are 1-indexed
                loop_masks[loop][resi] = True
            continue
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22].strip(), l[22:26].strip(), ' '+l[12:16].strip().ljust(3), l[17:20]
        if (chain,resNo) not in pdb_idx:
            continue
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    out = {
        'xyz':xyz, # cartesian coordinates, [Lx14]
        'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
        'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
        'seq':np.array(seq), # amino acid sequence, [L]
        'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
        'loop_masks' : loop_masks # dict of [L] masks indicating which residue belongs to which loop
    }

    return out