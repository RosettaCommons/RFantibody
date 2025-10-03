"""
This script converts antibody PDB files from Chothia format to HLT format.
HLT format requirements:
- Heavy chain is renamed to chain H
- Light chain is renamed to chain L
- Target chain(s) are renamed to chain T
- Chains are ordered as Heavy, Light, then Target
- CDR loops are annotated with REMARK statements at the end of the file
"""

import argparse
import numpy as np

from biotite.structure.io.pdb import PDBFile
from biotite.structure import array
from biotite.structure import residue_iter

protein_residues = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", 
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
    "THR", "TRP", "TYR", "VAL"
]

HCROP_DEF = 115
LCROP_DEF = 110

def parse_args():
    """Parse command line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - input_pdb: Path to input PDB file
            - heavy: Heavy chain ID in input file
            - light: Light chain ID in input file
            - target: Comma-separated list of target chain IDs
            - output: Optional output file path
    """
  
    
    parser = argparse.ArgumentParser(description='Convert Chothia-formatted PDB to HLT format')
    parser.add_argument('input_pdb', help='Input PDB file in Chothia format')
    parser.add_argument('--heavy', '-H', help='Heavy chain ID')
    parser.add_argument('--light', '-L', help='Light chain ID')
    parser.add_argument('--target', '-T', help='Target chain ID(s), comma-separated')
    parser.add_argument('--output', '-o', help='Output HLT file path')
    parser.add_argument('--whole_fab', '-w', action='store_true', help=f'Keep entire Fab region, not provided: False, cropping is used')
    parser.add_argument('--Hcrop', default=HCROP_DEF, type=int, help='Chothia residue number to crop to for heavy chain a ' + \
                        f'reasonable number is between 105 and 115, default: {HCROP_DEF}')
    parser.add_argument('--Lcrop', default=LCROP_DEF, type=int, help='Chothia residue number to crop to for light chain a ' + \
                        f'reasonable number is between 100 and 110, default: {LCROP_DEF}')
    parser.add_argument('--Tcrop', default="", help='Chothia residue numbers to use for target chains  ' + \
                        'format chain_id1:start_res_numb-end_res_numb,chain_id2:start_res_numb-end_res_numb... (like A:5-90, both start and end residue numbers are used)')
    
    args = parser.parse_args()

    if not (args.heavy or args.light or args.target):
        raise ValueError('Either heavy or light or target chain must be specified')



    return args

def get_Fv_ranges():
    """Define the residue ranges for each Fv loop according to Chothia numbering scheme.
    
    Returns:
        dict: Dictionary mapping Fv names to their residue ranges (start, end) inclusive
    """

    return {
        'H': (1, 102),
        'L': (1, 97)
    }

def get_cdr_ranges():
    """Define the residue ranges for each CDR loop according to Chothia numbering scheme.
    
    The Chothia numbering scheme is a standardized way to number antibody residues,
    making it possible to identify CDR loops based on residue numbers.
    
    Returns:
        dict: Dictionary mapping CDR names to their residue ranges (start, end) inclusive
    """
    return {
        'H': {
            'H1': (26, 32),  # Heavy chain CDR1: residues 26-32
            'H2': (52, 56),  # Heavy chain CDR2: residues 52-56
            'H3': (95, 102), # Heavy chain CDR3: residues 95-102
        },
        'L': {
            'L1': (24, 34),  # Light chain CDR1: residues 24-34
            'L2': (50, 56),  # Light chain CDR2: residues 50-56
            'L3': (89, 97),  # Light chain CDR3: residues 89-97
        },
    }

def convert_to_hlt(
    input_pdb,
    heavy_chain,
    light_chain,
    target_chains,
    whole_fab,
    Hcrop,
    Lcrop,
    Tcrop
):
    """Convert a Chothia-formatted PDB file to HLT format.
    
    Args:
        input_pdb (str): Path to input PDB file
        heavy_chain (str): Chain ID for heavy chain in input file
        light_chain (str): Chain ID for light chain in input file
        target_chains (list): List of chain IDs for target chains
        whole_fab (bool): Whether to keep entire Fab region
        Hcrop (int): Chothia residue number to crop to for heavy chain
        Lcrop (int): Chothia residue number to crop to for light chain
        Tcrop (int): Chothia start-end residue number to use
    
    Returns:
        tuple: (biotite.structure.Structure, dict)
            - Modified structure in HLT format
            - Dictionary mapping CDR names to lists of residue numbers
    """
    # Read input PDB file using biotite
    pdb_file = PDBFile.read(input_pdb)
    structure = pdb_file.get_structure(model=1)
 
        
    target_residD = {}
    if len(target_chains) > 0:
        if Tcrop != "":
            tL = Tcrop.strip().split(',')  # split for each target
            for t in tL:
                try:
                    tch = t.strip().split(':')  # splitting the chain id from the values
                 
                except:
                    print(f"Tcrop argument error: splitting string {t} to chain id and the residue ids was not successfull, ")
                    print(f"format for a chain should be chain_id:start_res_numb-end_res_numb")
                    print(f"Correct the arguments and try again, exiting")
                    exit()
                if len(tch) != 2:
                    print("Tcrop argument error: {t} could not be processed, format for a chain should be chain_id:start_res_numb-end_res_numb")
                    print(f"Correct the arguments and try again, exiting")
                    exit()
                if tch[0] not in target_chains:
                    print(f"Tcrop argument error: {tch[0]} chain not among target chains to process, it will be ignored")
                try:
                    tres = tch[1].strip().split('-')
                    if len(tres) != 2:
                        print(f"Tcrop argument error: splitting residue range {tres} for chain id {tch[0]} was not successfull, ")
                        print(f"format for a chain should be chain_id:start_res_numb-end_res_numb")
                        print(f"Correct the arguments and try again, exiting")
                        exit()
                    Tstart = int(tres[0])
                    Tend = int(tres[1])
                    target_residD[tch[0]] = [Tstart,Tend]
                except:
                    print(f"Tcrop argument error: extracting the starting and ending Chothia residue from {tres} for chain id {tch[0]} was not successfull, ")
                    print(f"format for a chain should be chain_id:start_res_numb-end_res_numb")
                    print(f"Correct the arguments and try again, exiting")
                    exit()
            for t in target_residD:
                print(f"For target chain {t} residues with Chothia number {target_residD[t][0]}-{target_residD[t][1]} will be used")    

    # Initialize new structure for HLT format
    atom_list = []   
    target_list = [] 
    
    # Subset the structure to only include protein chains
    protein_atom_list = []
    for atom in structure:
        if atom.res_name in protein_residues:
            protein_atom_list.append(atom)
 
    structure = array(protein_atom_list)

    # Initialize dictionary to track CDR loop residue numbers
    # These will be used to generate the REMARK statements
    cdr_residues = {
        'H1': [], 'H2': [], 'H3': [],
        'L1': [], 'L2': [], 'L3': []
    }
    
    # Process chains in HLT order
    current_residue = 1  # Track absolute residue numbering (1-indexed)
    cdr_ranges = get_cdr_ranges()
    
    # Process each chain type in order: H, L, T
    residue_counter = 1

    for chain_id in ['H', 'L', 'T']:
        orig_chain = None
        if chain_id == 'H':
            orig_chain = heavy_chain
        elif chain_id == 'L':
            orig_chain = light_chain
        else:  # Handle target chains (can be multiple)
            for t in target_chains:
                chain_mask = structure.chain_id == t
                if np.any(chain_mask):
                    
                    atoms = structure[chain_mask]

                    if t in target_residD:  # not all the residues should be used from this chain
                        for residue in residue_iter(atoms):
                            auth_res_num = np.unique(residue.res_id)[0]
                            if auth_res_num >= target_residD[t][0] and auth_res_num <= target_residD[t][1]:
                                # Rename chain to T
                                residue.chain_id = np.full(len(residue), chain_id)
                                target_list += residue

                    else:
                        # print('res at ', auth_res_num)
                        # Rename chain to T
                        atoms.chain_id = np.full(len(atoms), chain_id)
                        target_list += atoms
                        # print(atoms)
                                            
                    # Update residue counter
                    # TODO replace this unique with biotite's num residues function
                    current_residue += len(np.unique(atoms.res_id))
            
            
        # Get atoms for current chain among H and L
        chain_mask = structure.chain_id == orig_chain
        if np.any(chain_mask):
            atoms = structure[chain_mask]
            # Rename chain to H or L
            atoms.chain_id = np.full(len(atoms), chain_id)
        
            # Renumber residues to absolute numbering and identify CDR loop residues

            if chain_id in cdr_ranges:
                curr_ranges = cdr_ranges[chain_id]
                for residue in residue_iter(atoms):
                    auth_res_num = np.unique(residue.res_id)[0]

                    if not whole_fab:
                        if chain_id == 'H' and auth_res_num > Hcrop:
                            continue
                        elif chain_id == 'L' and auth_res_num > Lcrop:
                            continue
                    
                    for cdr, (start, end) in curr_ranges.items():
                        # Only process CDRs matching current chain
                        if start <= auth_res_num <= end:
                            # Convert to absolute residue number and store
                            cdr_residues[cdr].append(residue_counter)

                    # Assign the residue a new residue number
                    residue.res_id = np.full(len(residue), residue_counter)

                    # Remove insertion codes
                    residue.ins_code = np.full(len(residue), '')

                    atom_list += residue

                    residue_counter += 1

        if chain_id == 'T' and len(target_chains) >0:
            for residue in residue_iter(array(target_list)):
                # Assign the residue a new residue number
                residue.res_id = np.full(len(residue), residue_counter)

                # Remove insertion codes
                residue.ins_code = np.full(len(residue), '')
                             
                atom_list += residue

                residue_counter += 1

    # print("atom_list", atom_list, len(atom_list))
    return array(atom_list), cdr_residues

def main():
    """
    Main function to run the conversion process
    """

    # Parse command line arguments
    args = parse_args()
    target_chains = args.target.split(',') if args.target else []
 
    if (args.heavy != None or args.light != None):
        if args.whole_fab:
            print(f"Whole fab region is used for antibody, no cropping")
        else:
            if args.heavy != '':
                print(f"Heavy chain {args.heavy} will be cropped after {args.Hcrop}")
            if args.light != '':
                print(f"Light chain {args.light} will be cropped after {args.Lcrop}")

    # Generate output path if not specified
    output_path = args.output or args.input_pdb.replace('.pdb', '_HLT.pdb')
    
    # Convert structure to HLT format
    hlt_structure, cdr_residues = convert_to_hlt(
        args.input_pdb,
        args.heavy,
        args.light,
        target_chains,
        args.whole_fab,
        args.Hcrop,
        args.Lcrop,
        args.Tcrop
    )
    
    # Create new PDB file with converted structure
    pdb_file = PDBFile()
    pdb_file.set_structure(hlt_structure)
    
    # Write structure and CDR annotations
    with open(output_path, 'w') as f:
        pdb_file.write(f)
        # Add CDR annotations as REMARK statements
        # Format: REMARK PDBinfo-LABEL: <residue_number> <CDR_name>
        for cdr in sorted(cdr_residues.keys()):
            for res_num in sorted(cdr_residues[cdr]):
                f.write(f"REMARK PDBinfo-LABEL: {res_num:4d} {cdr}\n")

if __name__ == '__main__':
    main()
