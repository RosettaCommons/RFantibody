import numpy as np
import torch


def MSABlockDeletion(msa, ins, nb=5):
    '''
    Down-sample given MSA by randomly delete blocks of sequences
    Input: MSA/Insertion having shape (N, L)
    output: new MSA/Insertion with block deletion (N', L)
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    # Get statistics from clustering results (clustering extra sequences with seed sequences)
    csum = torch.zeros(N_seq, N_res, data.shape[-1], device=data.device).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params={'MAXLAT': 128, 'MAXSEQ': 512}, p_mask=0.15, eps=1e-6, L_s=[]):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    N, L = msa.shape
    
    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=21)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Nclust sequences will be selected randomly as a seed MSA (aka latent MSA)
    # - First sequence is always query sequence
    # - the rest of sequences are selected randomly
    Nclust = min(N, params['MAXLAT'])
    
    sample = torch.randperm(N-1, device=msa.device)
    msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
    ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

    # 15% random masking 
    # - 10%: aa replaced with a uniformly sampled random amino acid
    # - 10%: aa replaced with an amino acid sampled from the MSA profile
    # - 10%: not replaced
    # - 70%: replaced with a special token ("mask")
    random_aa = torch.tensor([[0.05]*20 + [0.0]], device=msa.device)
    same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=21)
    probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
    probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
    
    sampler = torch.distributions.categorical.Categorical(probs=probs)
    mask_sample = sampler.sample()

    mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
    msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
    seq_out = msa_masked[0].clone()
   
    ## get extra sequenes
    if N - Nclust >= params['MAXSEQ']: # there are enough extra sequences
        Nextra = params['MAXSEQ']
        msa_extra = torch.cat((msa_masked[:1,:], msa[1:,:][sample[Nclust-1:]]), dim=0) 
        ins_extra = torch.cat((ins_clust[:1,:], ins[1:,:][sample[Nclust-1:]]), dim=0)
        extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
        extra_mask[0] = mask_pos[0]
    elif N - Nclust < 1: # no extra sequences, use all masked seed sequence as extra one
        Nextra = Nclust
        msa_extra = msa_masked.clone()
        ins_extra = ins_clust.clone()
        extra_mask = mask_pos.clone()
    else: # it has extra sequences, but not enough to maxseq. Use mixture of seed (except query) & extra
        Nextra = min(N, params['MAXSEQ'])
        msa_add = msa[1:,:][sample[Nclust-1:]]
        ins_add = ins[1:,:][sample[Nclust-1:]]
        mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
        msa_extra = torch.cat((msa_masked, msa_add), dim=0)
        ins_extra = torch.cat((ins_clust, ins_add), dim=0)
        extra_mask = torch.cat((mask_pos, mask_add), dim=0)
    N_extra_pool = msa_extra.shape[0]
    
    # 1. one_hot encoded aatype: msa_clust_onehot
    msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=22) # (N, L, 22)
    msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=22)
    
    # clustering (assign remaining sequences to their closest cluster by Hamming distance
    count_clust = torch.logical_and(~mask_pos, msa_clust != 20).float() # 20: index for gap, ignore both masked & gaps
    count_extra = torch.logical_and(~extra_mask, msa_extra != 20).float()
    # get number of identical tokens for each pair of sequences (extra vs seed)
    agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra_pool, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T) # (N_extra_pool, Nclust)
    assignment = torch.argmax(agreement, dim=-1) # map each extra seq to the closest seed seq

    # 2. cluster profile -- ignore masked token when calculate profiles
    count_extra = ~extra_mask # only consider non-masked tokens in extra seqs
    count_clust = ~mask_pos # only consider non-masked tokens in seed seqs
    msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
    msa_clust_profile += count_clust[:,:,None]*msa_clust_onehot
    count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L) # 
    count_profile += count_clust
    count_profile += eps
    msa_clust_profile /= count_profile[:,:,None]

    # 3. insertion statistics
    msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
    msa_clust_del += count_clust*ins_clust
    msa_clust_del /= count_profile
    ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
    msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
    ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
    
    # seed MSA features (one-hot aa, cluster profile, ins statistics, terminal info)
    msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

    # extra MSA features (one-hot aa, insertion, terminal info)
    ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
    msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)

    return seq_out, msa_clust, msa_seed, msa_extra, mask_pos

