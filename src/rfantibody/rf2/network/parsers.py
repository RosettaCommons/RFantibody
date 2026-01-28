import gzip
import re
import string

import numpy as np
import torch

from rfantibody.rf2.network import util
from rfantibody.rf2.network.chemical import INIT_CRDS
from rfantibody.rf2.network.ffindex import *

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename, max_seq=5000):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)
    
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
        msa_i = line.translate(table)
        msas_i = msa_i.split('/')

        if (len(msa)==0):
            # first seq
            Ls = [len(x) for x in msas_i]
            msa = [[s] for s in msas_i]
        else:
            nchains = len(msas_i)
            isgood = all([ len(msa[i][0]) == len(msas_i[i]) for i in range(nchains) ])
            if isgood:
                for i in range(nchains):
                    msa[i].append(msas_i[i])
            else:
                print ("Len error", filename, len(msa[0]) )

        # sequence length
        L = sum(Ls)

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

        if (len(msa[0]) >= max_seq):
            break


    # concatenate
    msa = [np.array([list(s) for s in t], dtype='|S1').view(np.uint8) for t in msa]
    msa = np.concatenate(msa,axis=-1)

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)
    return msa,ins,Ls


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines)

#'''
def parse_pdb_lines(lines):

    # indices of residues observed in the structure
    idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]

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

    return xyz,mask,np.array(idx_s)

def parse_pdb_w_seq(filename, chnid=None):
    lines = open(filename,'r').readlines()
    return parse_pdb_lines_w_seq(lines)

def parse_pdb_lines_w_seq(lines, chnid=None):
    # indices of residues observed in the structure
    #idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    idx_s = [int(r[0]) for r in res]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 27, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chn, resNo, atom, aa = l[21], int(l[22:26]), l[12:16], l[17:20]
        if chnid is not None and chn != chain:
            continue

        idx = idx_s.index(resNo)
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    #xyz[np.isnan(xyz[...,0])] = 0.0

    return xyz,mask,np.array(idx_s), np.array(seq)


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

def parse_templates_raw(ffdb, hhr_fn, atab_fn, templ_to_use, max_templ=20):
    # process tabulated hhsearch output to get
    # matched positions and positional scores
    hits = []
    read_stat = False
    for l in open(atab_fn, "r").readlines():
        if l[0]=='>':
            read_stat = False
            if len(hits) == max_templ:
                break
            key = l[1:].split()[0]
            if len(templ_to_use) > 1:
                if key not in templ_to_use:
                    continue
            read_stat = True
            hits.append([key,[],[]])
        elif "score" in l or "dssp" in l:
            continue
        elif not read_stat:
            continue
        else:
            hi = l.split()[:5]+[0.0,0.0,0.0]
            hits[-1][1].append([int(hi[0]),int(hi[1])])
            hits[-1][2].append([float(hi[2]),float(hi[3]),float(hi[4])])

    # parse templates from FFDB
    for hi in hits:
        #if hi[0] not in ffids:
        #    continue
        entry = get_entry_by_name(hi[0], ffdb.index)
        if entry == None:
            print ("Failed to find %s in *_pdb.ffindex"%hi[0])
            continue
        data = read_entry_lines(entry, ffdb.data)
        hi += list(parse_pdb_lines_w_seq(data)) # (add four more items)

    # process hits
    counter = 0
    xyz,qmap,mask,f1d,ids,seq = [],[],[],[],[],[]
    for data in hits:
        if len(data)<7:
            continue
        #print ("Process %s..."%data[0])
        
        qi,ti = np.array(data[1]).T
        _,sel1,sel2 = np.intersect1d(ti, data[5], return_indices=True)
        ncol = sel1.shape[0]
        if ncol < 10:
            continue
        
        ids.append(data[0])
        f1d.append(np.array(data[2])[sel1])
        xyz.append(data[3][sel2])
        mask.append(data[4][sel2])
        seq.append(data[-1][sel2])
        qmap.append(np.stack([qi[sel1]-1,[counter]*ncol],axis=-1))
        counter += 1
    
    xyz = np.vstack(xyz).astype(np.float32)
    mask = np.vstack(mask).astype(np.float32)
    qmap = np.vstack(qmap).astype(np.int_)
    f1d = np.vstack(f1d).astype(np.float32)
    seq = np.hstack(seq).astype(np.int_)
    ids = ids

    return torch.from_numpy(xyz), torch.from_numpy(mask), torch.from_numpy(qmap), \
           torch.from_numpy(f1d), torch.from_numpy(seq), ids

def read_templates(qlen, ffdb, hhr_fn, atab_fn, templ_to_use=[], offset=0, n_templ=10, random_noise=5.0):
    xyz_t, mask, qmap, t1d, seq, ids = parse_templates_raw(ffdb, hhr_fn, atab_fn, templ_to_use, max_templ=max(n_templ, 20))
    npick = min(n_templ, len(ids))
    if npick < 1: # no templates
        xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npick,qlen,1,1) + torch.rand(npick,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((1, qlen), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((1,qlen,1)).float()), -1)
        return xyz, t1d

    sample = torch.arange(npick)
    #
    xyz = INIT_CRDS.reshape(1,1,27,3).repeat(npick,qlen,1,1) + torch.rand(npick,qlen,1,3)*random_noise
    mask_t = torch.full((npick,qlen,27),False) # True for valid atom, False for missing atom
    f1d = torch.full((npick, qlen), 20).long()
    f1d_val = torch.zeros((npick, qlen, 1)).float()
    #
    for i, nt in enumerate(sample):
        sel = torch.where(qmap[:,1] == nt)[0]
        pos = qmap[sel, 0] + offset
        xyz[i, pos] = xyz_t[sel]
        mask_t[i,pos] = mask[sel].bool()
        f1d[i, pos] = seq[sel]
        f1d_val[i,pos] = t1d[sel, 2].unsqueeze(-1)
        #
        xyz[i] = util.center_and_realign_missing(xyz[i], mask_t[i])
    
    f1d = torch.nn.functional.one_hot(f1d, num_classes=21).float()
    f1d = torch.cat((f1d, f1d_val), dim=-1)
    
    return xyz, f1d, mask_t

def read_template_pdb(qlen, templ_fn, align_conf=1.0):
    xyz = torch.full((qlen, 27, 3), np.nan).float()
    seq = torch.full((qlen,), 20).long() # all gaps
    
    with open(templ_fn) as fp:
        for l in fp:
            if l[:4] != "ATOM":
                continue
            
            resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]
            aa_idx = util.aa2num[aa] if aa in util.aa2num.keys() else 20
            #
            idx = resNo - 1
            for i_atm, tgtatm in enumerate(util.aa2long[aa_idx][:3]):
                if tgtatm == atom:
                    xyz[idx,i_atm,:] = torch.tensor([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                    break
            else:
                continue
            seq[idx] = aa_idx
    
    maskaa = torch.logical_not(torch.isnan(xyz[:,:,0])) # (qlen, 3)
    mask = maskaa[:,:3].all(dim=-1) # (qlen)
    conf = torch.where(mask, torch.tensor(align_conf).float(), torch.zeros(qlen, dtype=xyz.dtype))
    seq = torch.nn.functional.one_hot(seq, num_classes=21).float()
    t1d = torch.cat((seq, conf[:,None]), -1)
    xyz = xyz.nan_to_num()

    return xyz[None], t1d[None], maskaa[None]
