import os
import re

import numpy as np

import string
import os,re
import random

import gzip

import pandas as pd
from rfantibody.rfdiffusion import util 
from rfantibody.rfdiffusion.chemical import aa2num, aa2long


to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue
            
        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file

def parse_pdb(filename, xyz27=False,seq=False):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, xyz27, seq)

#'''
def parse_pdb_lines(lines, xyz27, seq, get_aa=util.aa2num.get):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    # 4 BB + up to 10 SC atoms
    if xyz27:
        xyz = np.full((len(idx_s), 27, 3), np.nan, dtype=np.float32)
    else:
        xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[get_aa(aa)]):
            if tgtatm and tgtatm.strip() == atom.strip():
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0 
    if not seq:
        return xyz,mask,np.array(idx_s)
    else:
        return xyz,mask,np.array(idx_s),np.array(seq)

#'''

'''
def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    #idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    idx_s = [int(r[0]) for r in res]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s), np.array(seq)
'''


def parse_templates(item, params):

    # init FFindexDB of templates
    ### and extract template IDs
    ### present in the DB
    ffdb = FFindexDB(read_index(params['FFDB']+'_pdb.ffindex'),
                     read_data(params['FFDB']+'_pdb.ffdata'))
    #ffids = set([i.name for i in ffdb.index])

    # process tabulated hhsearch output to get
    # matched positions and positional scores
    infile = params['DIR']+'/hhr/'+item[-2:]+'/'+item+'.atab'
    hits = []
    for l in open(infile, "r").readlines():
        if l[0]=='>':
            key = l[1:].split()[0]
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # get per-hit statistics from an .hhr file
    # (!!! assume that .hhr and .atab have the same hits !!!)
    # [Probab, E-value, Score, Aligned_cols, 
    # Identities, Similarity, Sum_probs, Template_Neff]
    lines = open(infile[:-4]+'hhr', "r").readlines()
    pos = [i+1 for i,l in enumerate(lines) if l[0]=='>']
    for i,posi in enumerate(pos):
        hits[i].append([float(s) for s in re.sub('[=%]',' ',lines[posi]).split()[1::2]])
        
    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines(data))

    # process hits
    counter = 0
    xyz,qmap,mask,f0d,f1d,ids = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[6], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f0d.append(data[3])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[4][sel2])
        mask.append(data[5][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1

    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(np.bool)
    qmap = np.vstack(qmap).astype(np.long)
    f0d = np.vstack(f0d).astype(np.float32)
    f1d = np.vstack(f1d).astype(np.float32)
    ids = ids
        
    return xyz,mask,qmap,f0d,f1d,ids

def HLT_pdb_parser(path):
    '''
    inputs:
        - path | string of path to chothia labeled pdb
        - path to summary file (optional)
    outputs:
        - out | dictionary including, xyz, seq, pdb_idx, cdr_bool
    Adapted from Joe's chothia parser

    NB this function is able to handle files with repeated pdb residue indices
    '''

    cdr_names = ['H1','H2','H3','L1','L2','L3']

    with open(path,'r') as f:
        lines = f.readlines()

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

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
           'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
           'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
           'seq':np.array(seq), # amino acid sequence, [L]
           'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           'loop_masks' : loop_masks # dict of [L] masks indicating which residue belongs to which loop
           }
    return out

def chothia_pdb_parser(path, summary_path=None, chains=None, expanded_loop_def=False, rand_loop_ext=0):
    '''
    inputs:
        - path | string of path to chothia labeled pdb
        - path to summary file (optional)
    outputs:
        - out | dictionary including, xyz, seq, pdb_idx, cdr_bool
    Adapted from Jake's parser. cdr_bool is now for *all* Ab chains in a pdb file
    '''

    with open(path,'r') as f:
        lines = f.readlines()

    if chains is None:
        if summary_path is None:
            summary_path = '/databases/antibody_2023JAN13/sabdab_summary_all.tsv'
            if not os.path.exists(summary_path):
                summary_path = '/net/databases/antibody/sabdab_summary_all.tsv'

        df=pd.read_csv(summary_path, sep='\t')
        #load current path from sabdab summary file
        tmp = df[df['pdb']==path[-8:-4]]
        h_chains = tmp['Hchain'].tolist()
        l_chains = tmp['Lchain'].tolist()

        if h_chains == l_chains:
            print('Chains are the same')
            return None
    else:
        h_chains = chains['H']
        l_chains = chains['L']

    # indices of residues observed in the structure
    res = []
    for l in lines:
            i = 0 if l[11] == ' ' else 1
            if l[:4]=="ATOM" and l[12+i:16+i].strip()=="CA":
                res.append((l[22+i:27+i].strip(),l[17+i:20+1].strip(),l[21+i]))

    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [(l[21:22].strip(), l[22:27].strip()) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + 23 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22].strip(), l[22:27].strip(), ' '+l[12:16].strip().ljust(3), l[17:20]
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

    cdr_chothia={'L1':(24,34),'L2':(50,56),'L3':(89,97),'H1':(26,32),'H2':(52,56),'H3':(95,102)}
    if expanded_loop_def:
        # Expand the definition of CDR loops by 3 residues in each direction
        cdr_chothia={key: (cdr_chothia[key][0]-3, cdr_chothia[key][1]+3) for key in cdr_chothia}

    if rand_loop_ext > 0:
        # Randomly extend each loop start and end by uniform(0, rand_loop_ext) residues
        for key in cdr_chothia:
            rand_start = random.randint(0, rand_loop_ext)
            rand_end   = random.randint(0, rand_loop_ext)

            cdr_chothia[key] = (cdr_chothia[key][0] - rand_start, cdr_chothia[key][1] + rand_end)

    cdrs_bool = np.zeros(len(pdb_idx))
    for i,l in enumerate(pdb_idx):
        iscdr = False
        if l[1][-1].isalpha():
            idx_i = int(l[1][:-1])
        else:
            idx_i = int(l[1])

        if l[0] in l_chains:
            if idx_i in range(cdr_chothia['L1'][0],cdr_chothia['L1'][1]+1):
                iscdr=True
            elif idx_i in range(cdr_chothia['L2'][0],cdr_chothia['L2'][1]+1):
                iscdr=True
            elif idx_i in range(cdr_chothia['L3'][0],cdr_chothia['L3'][1]+1):
                iscdr=True
        elif l[0] in h_chains:
            if idx_i in range(cdr_chothia['H1'][0],cdr_chothia['H1'][1]+1):
                iscdr=True
            elif idx_i in range(cdr_chothia['H2'][0],cdr_chothia['H2'][1]+1):
                iscdr=True
            elif idx_i in range(cdr_chothia['H3'][0],cdr_chothia['H3'][1]+1):
                iscdr=True
        if iscdr:
            cdrs_bool[i] = 1

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           'cdr_bool' : [True if i==1 else False for i in cdrs_bool]
           }
    return out

def split_remark(line):
    '''split remark line into a tuple of chain, idx
    e.g. REMARK PDBinfo-LABEL:   26 H1 is split into (H, 26)
    '''
    chain = line.split()[3][0]
    idx = int(line.split()[2])
    return chain, idx

def parse_HLT_remarked(pdb_path, preserve_pdb_numbering=False):
    """
    New parser that also reads lines prefixed with 'REMARK PDBinfo-LABEL'
    and gets the cdr_bool from this
    Inputs:
        path to pdb file.
        Must be in H,L,T format.
        No other chains
    Outputs:
        pdb object in the same format as chothia_pdb_parser
    """
    lines = open(pdb_path,'r').readlines()
    return parse_HLT_lines(lines, preserve_pdb_numbering)

def parse_HLT_lines(lines, preserve_pdb_numbering=False):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 27, 3), np.nan, dtype=np.float32)
    cdr_pdb_idx=[]
    for l in lines:
        if l[:4] == "ATOM":
            chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
            idx = pdb_idx.index((chain,resNo))
            for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
                if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                    xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break
        elif l.startswith('REMARK PDBinfo-LABEL'):
            l=l.strip()
            cdr_pdb_idx.append(split_remark(l))
    
    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0 

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)
    
    pdb_idx = new_idx

    # Now get cdr_bool
    cdr_bool=[True if i in cdr_pdb_idx else False for i in pdb_idx]

    if not preserve_pdb_numbering:
        # set pdb_idx to 1 indexed
        chains=set(np.unique([i[0] for i in pdb_idx]))
        for ch in chains:
            sub=min([i[1] for i in pdb_idx if i[0]==ch])-1
            pdb_idx=[(i[0],i[1]-sub) if i[0] == ch else i for i in pdb_idx]

    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx27]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
            'cdr_bool' : cdr_bool # [L]
           }
    return out
