import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from rfantibody.rf2.network.Attention_module import Attention
from rfantibody.rf2.network.Track_module import PairStr2Pair
from rfantibody.rf2.network.util import get_Cb
from rfantibody.rf2.network.util_module import (
    create_custom_forward,
    init_lecun_normal,
    rbf,
)


# Module contains classes and functions to generate initial embeddings
class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.d_out = d_model
    
    def forward(self, idx, stride, nc_cycle=False):
        B, L = idx.shape[:2]

        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)

        seqsep = torch.full((B,L,L),100, device=idx.device)
        seqsep[0] = idx[0,None,:] - idx[0,:,None] # (B, L, L)
        if nc_cycle:
            seqsep[0] = (seqsep[0] + L//2)%L - L//2

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        if STRIDE>0 and STRIDE<L:
            emb = torch.zeros((B,L,L,self.d_out), device=idx.device, dtype=self.emb.weight.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=idx.device)[:,None]
                for j in range((L-1)//STRIDE+1):
                    cols = torch.arange(j*STRIDE, min((j+1)*STRIDE, L), device=idx.device)[None,:]
                    ib = torch.bucketize(seqsep[:,rows,cols], bins).long()
                    emb[:,rows,cols] = self.emb(ib)
        else:
            ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
            emb = self.emb(ib) #(B, L, L, d_model)

        return emb

class MSA_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_init=22+22+2+2,
                 minpos=-32, maxpos=32, p_drop=0.1):
        super(MSA_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence -- used for MSA embedding
        self.emb_left = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_right = nn.Embedding(22, d_pair) # embedding for query sequence -- used for pair embedding
        self.emb_state = nn.Embedding(22, d_state)
        self.pos = PositionalEncoding2D(d_pair, minpos=minpos, maxpos=maxpos)

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        self.emb_q = init_lecun_normal(self.emb_q)
        self.emb_left = init_lecun_normal(self.emb_left)
        self.emb_right = init_lecun_normal(self.emb_right)
        self.emb_state = init_lecun_normal(self.emb_state)

        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, stride, nc_cycle=None):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #   - pair: Initial Pair embedding (B, L, L, d_pair)

        B, N, L = msa.shape[:3]

        # msa embedding 
        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        tmp = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = (msa + tmp.expand(-1, N, -1, -1)) # adding query embedding to MSA

        # pair embedding 
        left = self.emb_left(seq)[:,None] # (B, 1, L, d_pair)
        right = self.emb_right(seq)[:,:,None] # (B, L, 1, d_pair)
        pair = (left + right) # (B, L, L, d_pair)
        pair += self.pos(idx, stride, nc_cycle) # add relative position

        # state embedding
        state = self.emb_state(seq) #.repeat(oligo,1,1)

        return msa, pair, state

class Extra_emb(nn.Module):
    # Get initial seed MSA embedding
    def __init__(self, d_msa=256, d_init=22+1+2, p_drop=0.1):
        super(Extra_emb, self).__init__()
        self.emb = nn.Linear(d_init, d_msa) # embedding for general MSA
        self.emb_q = nn.Embedding(22, d_msa) # embedding for query sequence

        self.d_init = d_init
        self.d_msa = d_msa

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

    def forward(self, msa, seq, idx, oligo=1):
        # Inputs:
        #   - msa: Input MSA (B, N, L, d_init)
        #   - seq: Input Sequence (B, L)
        #   - idx: Residue index
        # Outputs:
        #   - msa: Initial MSA embedding (B, N, L, d_msa)
        #N = msa.shape[1] # number of sequenes in MSA

        B,N,L = msa.shape[:3]

        msa = self.emb(msa) # (B, N, L, d_model) # MSA embedding
        seq = self.emb_q(seq).unsqueeze(1) # (B, 1, L, d_model) -- query embedding
        msa = msa + seq.expand(-1, N, -1, -1) # adding query embedding to MSA

        return msa 

class TemplatePairStack(nn.Module):
    # process template pairwise features
    # use structure-biased attention
    def __init__(self, n_block=2, d_templ=64, n_head=4, d_hidden=16, d_t1d=22, d_state=32, p_drop=0.25):
        super(TemplatePairStack, self).__init__()
        self.n_block = n_block
        self.proj_t1d = nn.Linear(d_t1d, d_state)
        proc_s = [PairStr2Pair(d_pair=d_templ, n_head=n_head, d_hidden=d_hidden, d_state=d_state, p_drop=p_drop) for i in range(n_block)]
        self.block = nn.ModuleList(proc_s)
        self.norm = nn.LayerNorm(d_templ)
        self.reset_parameter()

        self.d_out = d_templ
    
    def reset_parameter(self):
        self.proj_t1d = init_lecun_normal(self.proj_t1d)
        nn.init.zeros_(self.proj_t1d.bias)

    def forward(self, templ, rbf_feat, t1d, strides, use_checkpoint=False, p2p_crop=-1):
        B, T, L = templ.shape[:3]

        templ = templ.reshape(B*T, L, L, -1)
        t1d = t1d.reshape(B*T, L, -1)
        state = self.proj_t1d(t1d)

        for i_block in range(self.n_block):
            if use_checkpoint:
                templ = checkpoint.checkpoint(create_custom_forward(self.block[i_block]), templ, rbf_feat, state, strides, p2p_crop, use_reentrant=True) 
            else:
                templ = self.block[i_block](templ, rbf_feat, state, strides, p2p_crop)

        # fd reduce memory in inference
        STRIDE = L
        if strides is not None and not self.training:
            STRIDE = strides['templ_pair']

        if STRIDE>0 and STRIDE<L:
            out = torch.zeros((B*T,L,L,self.d_out), device=templ.device, dtype=t1d.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))
                out[:,:,rows] = self.norm(templ[:,:,rows]).to(t1d.dtype)
        else:
            out = self.norm(templ)

        return out.reshape(B, T, L, L, -1)


class Templ_emb(nn.Module):
    # Get template embedding
    # Features are
    #   t2d:
    #   - 37 distogram bins + 6 orientations (43)
    #   - Mask (missing/unaligned) (1)
    #   t1d:
    #   - tiled AA sequence (20 standard aa + gap)
    #   - confidence (1)
    #   
    def __init__(self, d_t1d=21+1, d_t2d=43+1, d_tor=30, d_pair=128, d_state=32, 
                 n_block=2, d_templ=64,
                 n_head=4, d_hidden=16, p_drop=0.25):
        super(Templ_emb, self).__init__()
        # process 2D features
        self.emb = nn.Linear(d_t1d*2+d_t2d, d_templ)
        self.templ_stack = TemplatePairStack(n_block=n_block, d_templ=d_templ, n_head=n_head,
                                             d_hidden=d_hidden, p_drop=p_drop, d_t1d=d_t1d)
        
        self.attn = Attention(d_pair, d_templ, n_head, d_hidden, d_pair, p_drop=p_drop)
        
        # process torsion angles
        self.proj_t1d = nn.Linear(d_t1d+d_tor, d_templ)
        self.attn_tor = Attention(d_state, d_templ, n_head, d_hidden, d_state, p_drop=p_drop)

        #fd
        self.d_rbf = 64
        self.d_templ = d_templ

        self.reset_parameter()
    
    def reset_parameter(self):
        self.emb = init_lecun_normal(self.emb)
        nn.init.zeros_(self.emb.bias)

        nn.init.kaiming_normal_(self.proj_t1d.weight, nonlinearity='relu')
        nn.init.zeros_(self.proj_t1d.bias)
    
    def _get_templ_emb(self, t1d, t2d, stride):
        B, T, L, _ = t1d.shape

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        # Prepare 2D template features
        left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
        right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)
        #
        templ = torch.cat((t2d.to(t1d.device), left, right), -1) # (B, T, L, L, 88)
        if STRIDE>0 and STRIDE<L:
            out = torch.zeros((B,T,L,L,self.d_templ), device=t1d.device, dtype=t2d.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))
                out[:,:,rows] = self.emb(templ[:,:,rows])
        else:
            out = self.emb(templ)

        return out # Template templures (B, T, L, L, d_templ)

    def _get_templ_rbf(self, xyz_t, mask_t, stride):
        B, T, L = xyz_t.shape[:3]

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        # process each template features
        xyz_t = xyz_t.reshape(B*T, L, 3)
        mask_t = mask_t.reshape(B*T, L, L)

        rbf_feat = torch.zeros((B*T,L,L,self.d_rbf), device=xyz_t.device, dtype=xyz_t.dtype)
        for i in range((L-1)//STRIDE+1):
            rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))

            rbf_i = rbf( torch.cdist(xyz_t[:,rows], xyz_t) )
            #print (rbf_i.shape, mask_t[:,rows].shape)
            rbf_i *= mask_t[:,rows].unsqueeze(-1)

            rbf_feat[:,rows] = rbf_i.to(xyz_t.dtype)

        return rbf_feat
    
    def forward(
      self, t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, strides, 
      use_checkpoint=False, p2p_crop=-1, symmids=None
    ):
        # Input
        #   - t1d: 1D template info (B, T, L, 22)
        #   - t2d: 2D template info (B, T, L, L, 44)
        #   - alpha_t: torsion angle info (B, T, L, 30)
        #   - xyz_t: template CA coordinates (B, T, L, 3)
        #   - mask_t: is valid residue pair? (B, T, L, L)
        #   - pair: query pair features (B, L, L, d_pair)
        #   - state: query state features (B, L, d_state)
        B, T, L, _ = t1d.shape

        if strides is None or self.training:
            templ_emb_stripe = -1
            templ_attn_stripe = -1
        else:
            templ_emb_stripe = strides['templ_emb']
            templ_attn_stripe = strides['templ_attn']

        templ = self._get_templ_emb(t1d, t2d, templ_emb_stripe)
        rbf_feat = self._get_templ_rbf(xyz_t, mask_t, templ_emb_stripe)

        # process each template pair feature
        templ = self.templ_stack(
            templ, rbf_feat, t1d, strides,
            use_checkpoint=use_checkpoint, p2p_crop=p2p_crop
        ).to(pair.dtype) # (B, T, L,L, d_templ)

        # Prepare 1D template torsion angle features
        t1d = torch.cat((t1d, alpha_t), dim=-1) # (B, T, L, 22+30)
        t1d = self.proj_t1d(t1d)
        
        # mixing query state features to template state features
        state = state.reshape(B*L, 1, -1) # (B*L, 1, d_state)
        t1d = t1d.permute(0,2,1,3).reshape(B*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn_tor), state, t1d, t1d, templ_attn_stripe, use_reentrant=True)
            out = out.reshape(B, L, -1)
        else:
            out = self.attn_tor(state, t1d, t1d, templ_attn_stripe).reshape(B, L, -1)
        state = state.reshape(B, L, -1)
        if self.training:
            state = state + out
        else:
            state += out

        # mixing query pair features to template information (Template pointwise attention)
        pair = pair.reshape(B*L*L, 1, -1)
        templ = templ.permute(0, 2, 3, 1, 4).reshape(B*L*L, T, -1)
        if use_checkpoint:
            out = checkpoint.checkpoint(create_custom_forward(self.attn), pair, templ, templ, templ_attn_stripe, use_reentrant=True)
            out = out.reshape(B, L, L, -1)
        else:
            out = self.attn(pair, templ, templ, templ_attn_stripe).reshape(B, L, L, -1)
        #
        pair = pair.reshape(B, L, L, -1)
        if self.training:
            pair = pair + out
        else:
            pair += out

        return pair, state

class Recycling(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=32, d_rbf=64):
        super(Recycling, self).__init__()
        self.proj_dist = nn.Linear(d_rbf+d_state*2, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)

        self.d_pair = d_pair
        self.d_msa = d_msa
        self.d_rbf = d_rbf
        
        self.reset_parameter()

    def reset_parameter(self):
        #self.emb_rbf = init_lecun_normal(self.emb_rbf)
        #nn.init.zeros_(self.emb_rbf.bias)
        self.proj_dist = init_lecun_normal(self.proj_dist)
        nn.init.zeros_(self.proj_dist.bias)

    def forward(self, seq, msa, pair, state, xyz, stride, mask_recycle=None):
        B, L = msa.shape[:2]
        dtype = msa.dtype

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        state = self.norm_state(state)

        if STRIDE<L:
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))
                msa_i = msa[:,rows]
                msa[:,rows] = self.norm_msa(msa_i).to(dtype)

                pair_i = pair[:,rows]
                pair[:,rows] = self.norm_pair(pair_i).to(dtype)
        else:
            msa = self.norm_msa(msa).to(dtype)
            pair = self.norm_pair(pair).to(dtype)

        ## SYMM
        left = state.to(dtype).unsqueeze(2).expand(-1,-1,L,-1)
        right = state.to(dtype).unsqueeze(1).expand(-1,L,-1,-1)

        # recreate Cb given N,Ca,C
        Cb = get_Cb(xyz[:,:,:3])

        dist_CB = torch.zeros((B,L,L,self.d_rbf), device=msa.device, dtype=dtype)

        if STRIDE<L:
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))
                dist_CB[:,rows] = rbf(
                    torch.cdist(Cb[:,rows], Cb)
                ).to(dtype)
        else:
            dist_CB = rbf( torch.cdist( Cb, Cb ) ).to(dtype)

        if mask_recycle != None:
            dist_CB = mask_recycle[...,None].to(dtype)*dist_CB

        dist_CB = torch.cat((dist_CB, left, right), dim=-1)

        if STRIDE<L:
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L))
                pair[:,rows] += self.proj_dist(dist_CB[:,rows])
        else:
            pair += self.proj_dist(dist_CB)

        return msa, pair, state

