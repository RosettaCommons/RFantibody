import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from rfantibody.rf2.network.util_module import init_lecun_normal


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, r_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model*r_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_model*r_ff, d_model)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize linear layer right before ReLu: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear1.bias)

        # initialize linear layer right before residual connection: zero initialize
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, src, stride=0):
        # a bit hacky.  L is always the -2 dimension so stripe there
        L = src.shape[-2]
        dtype = src.dtype

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        if STRIDE>0 and STRIDE<L:
            out = torch.zeros_like(src)
            for i in range((L-1)//STRIDE+1):
                cols = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=src.device)
                out_i = self.norm(src[...,cols,:])
                out_i = self.linear2(self.dropout(F.relu(self.linear1(out_i))))
                out[...,cols,:] = out_i.to(dtype=dtype)
            src = out
        else:
            src = self.norm(src)
            src = self.linear2(self.dropout(F.relu(self.linear1(src))))

        return src

class Attention(nn.Module):
    # calculate multi-head attention
    def __init__(self, d_query, d_key, n_head, d_hidden, d_out, p_drop=0.1):
        super(Attention, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.d_out = d_out
        #
        self.to_q = nn.Linear(d_query, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_key, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_key, n_head*d_hidden, bias=False)
        #
        self.to_out = nn.Linear(n_head*d_hidden, d_out)
        self.scaling = 1/math.sqrt(d_hidden)
        #
        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, query, key, value, stride):
        B, Q = query.shape[:2]
        B, K = key.shape[:2]
        #

        ## fd this is only ever called w/ high B and low Q/K
        ## apply stride along B
        STRIDE = B
        if (not self.training and stride>0 and stride<B):
            STRIDE = stride

        if STRIDE>0 and STRIDE<B:
            out = torch.zeros((B,Q,self.d_out), device=query.device, dtype=query.dtype)
            for i in range((B-1)//STRIDE+1):
                batches = torch.arange(i*STRIDE, min((i+1)*STRIDE, B), device=query.device)

                query_i = self.to_q(query[batches]).reshape(-1, Q, self.h, self.dim)
                key_i = self.to_k(key[batches]).reshape(-1, K, self.h, self.dim)
                value_i = self.to_v(value[batches]).reshape(-1, K, self.h, self.dim)

                query_i = query_i * self.scaling
                attn = einsum('bqhd,bkhd->bhqk', query_i, key_i)
                attn = F.softmax(attn, dim=-1)
        
                out_i = einsum('bhqk,bkhd->bqhd', attn, value_i)
                out_i = out_i.reshape(-1, Q, self.h*self.dim)
        
                out[batches] = self.to_out(out_i).to(query.dtype)
        else:
            query = self.to_q(query).reshape(B, Q, self.h, self.dim)
            key = self.to_k(key).reshape(B, K, self.h, self.dim)
            value = self.to_v(value).reshape(B, K, self.h, self.dim)

            query = query * self.scaling
            attn = einsum('bqhd,bkhd->bhqk', query, key)
            attn = F.softmax(attn, dim=-1)

            out = einsum('bhqk,bkhd->bqhd', attn, value)
            out = out.reshape(B, Q, self.h*self.dim)
            out = self.to_out(out)

        return out

# MSA Attention (row/column) from AlphaFold architecture
class SequenceWeight(nn.Module):
    def __init__(self, d_msa, n_head, d_hidden, p_drop=0.1):
        super(SequenceWeight, self).__init__()
        self.h = n_head
        self.dim = d_hidden
        self.scale = 1.0 / math.sqrt(self.dim)

        self.to_query = nn.Linear(d_msa, n_head*d_hidden)
        self.to_key = nn.Linear(d_msa, n_head*d_hidden)
        self.dropout = nn.Dropout(p_drop)

        self.reset_parameter()
    
    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)

    def forward(self, msa, MSANorm, stride=0):
        B, N, L = msa.shape[:3]

        STRIDE = N
        if (not self.training and stride>0):
            STRIDE = stride

        tar_seq = MSANorm(msa[:,0])
        q = self.to_query(tar_seq).view(B, 1, L, self.h, self.dim)
        q = q * self.scale
        attn = torch.zeros((B,N,L,self.h,1), device=msa.device, dtype=msa.dtype)

        
        if STRIDE>0 and STRIDE<N:
            for i in range((N-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, N), device=msa.device)
                msa_i = MSANorm(msa[:,rows])
                k_i = self.to_key(msa_i).view(B, -1, L, self.h, self.dim)
                attn[:,rows] = einsum('bqihd,bkihd->bkihq', q, k_i)
        else:
            k = self.to_key( MSANorm(msa) ).view(B, N, L, self.h, self.dim)
            attn = einsum('bqihd,bkihd->bkihq', q, k)

        attn = F.softmax(attn, dim=1)
        return self.dropout(attn)

class MSARowAttentionWithBias(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_hidden=32):
        super(MSARowAttentionWithBias, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        #
        self.seq_weight = SequenceWeight(d_msa, n_head, d_hidden, p_drop=0.1)
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_pair, n_head, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.dim_out = d_msa

        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        
        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, pair, stride_n, stride_l):
        B, N, L = msa.shape[:3]

        # fd reduce memory in inference
        STRIDE_N, STRIDE_L = N, L
        if (not self.training and stride_n>0):
            STRIDE_N = stride_n
        if (not self.training and stride_l>0):
            STRIDE_L = stride_l

        seq_weight = self.seq_weight(msa, self.norm_msa, stride_l)

        if STRIDE_L < L or STRIDE_N < N:
            # low memory
            attn = torch.zeros((B,L,L,self.h), device=pair.device, dtype=pair.dtype)
            for i in range((N-1)//STRIDE_N+1):
                rows = torch.arange(i*STRIDE_N, min((i+1)*STRIDE_N, N), device=pair.device)

                msa_i = self.norm_msa(msa[:,rows])
                seq_weight_i = seq_weight[:,rows]
                query_i = self.to_q(msa_i).reshape(B, -1, L, self.h, self.dim)
                key_i = self.to_k(msa_i).reshape(B, -1, L, self.h, self.dim)

                key_i *= self.scaling
                query_i *= seq_weight_i.expand(-1, -1, -1, -1, self.dim)

                attn_i = einsum('bnihk,bnjhk->bijh', query_i, key_i)
                attn += attn_i

            for i in range((L-1)//STRIDE_L+1):
                rows = torch.arange(i*STRIDE_L, min((i+1)*STRIDE_L, L), device=pair.device)
                pair_i = self.norm_pair(pair[:,rows])
                attn[:,rows] += self.to_b(pair_i) # (B, STRIDE, L, h)

            attn = F.softmax(attn, dim=-2) # (B, L, L, h)

            out = torch.zeros((B,N,L,self.h*self.dim), device=pair.device, dtype=pair.dtype)
            for i in range((L-1)//STRIDE_L+1):
                slices = torch.arange(i*STRIDE_L, min((i+1)*STRIDE_L, L), device=pair.device) # rows in value, cols in out
                msa_i = self.norm_msa(msa[:,:,slices])
                value_i = self.to_v(msa_i).reshape(B, N, -1, self.h, self.dim)
                out += einsum(
                    'bijh,bnjhd->bnihd', attn[:,:,slices], value_i
                ).reshape(B, N, L, -1)

            outg = torch.zeros((B,N,L,self.dim_out), device=pair.device, dtype=pair.dtype)
            for i in range((L-1)//STRIDE_L+1):
                slices = torch.arange(i*STRIDE_L, min((i+1)*STRIDE_L, L), device=pair.device) # rows in value, cols in out
                msa_i = self.norm_msa(msa[:,:,slices])
                gate_i = torch.sigmoid(self.to_g(msa_i)) # (B, N, L, h*dim)
                out_i = out[:,:,slices] * gate_i
                outg[:,:,slices] = self.to_out( out_i ).to(dtype=pair.dtype)
            out = outg

        else:
            msa = self.norm_msa(msa)
            pair = self.norm_pair(pair)

            query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
            key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
            value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
            bias = self.to_b(pair) # (B, L, L, h)
            gate = torch.sigmoid(self.to_g(msa))

            query = query * seq_weight.expand(-1, -1, -1, -1, self.dim)
            key = key * self.scaling
            attn = einsum('bsqhd,bskhd->bqkh', query, key)
            attn = attn + bias
            attn = F.softmax(attn, dim=-2)

            out = einsum('bqkh,bskhd->bsqhd', attn, value).reshape(B, N, L, -1)
            out = gate * out

            out = self.to_out(out)

        return out


class MSAColAttention(nn.Module):
    def __init__(self, d_msa=256, n_head=8, d_hidden=32):
        super(MSAColAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.dim_out = d_msa
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, stride):
        B, N, L = msa.shape[:3]
        dtype = msa.dtype

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        if STRIDE > 0 and STRIDE < L:
            out = torch.zeros((B,N,L,self.dim_out), device=msa.device, dtype=msa.dtype)
            for i in range((L-1)//STRIDE+1):
                cols = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=msa.device)

                msa_i = self.norm_msa(msa[:,:,cols]).to(dtype=dtype)
                query_i = self.to_q(msa_i).reshape(B, N, -1, self.h, self.dim)
                key_i = self.to_k(msa_i).reshape(B, N, -1, self.h, self.dim)
                value_i = self.to_v(msa_i).reshape(B, N, -1, self.h, self.dim)
                gate_i = torch.sigmoid(self.to_g(msa_i))

                query_i = query_i * self.scaling
                attn_i = einsum('bqihd,bkihd->bihqk', query_i, key_i)
                attn_i = F.softmax(attn_i, dim=-1)

                out_i = einsum('bihqk,bkihd->bqihd', attn_i, value_i).reshape(B, N, -1, self.dim_out)
                out_i = gate_i * out_i
                out_i = self.to_out(out_i).to(dtype=msa.dtype)
                #
                out[:,:,cols] = out_i
        else:
            msa = self.norm_msa(msa)

            query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
            key = self.to_k(msa).reshape(B, N, L, self.h, self.dim)
            value = self.to_v(msa).reshape(B, N, L, self.h, self.dim)
            gate = torch.sigmoid(self.to_g(msa))

            query = query * self.scaling
            attn = einsum('bqihd,bkihd->bihqk', query, key)
            attn = F.softmax(attn, dim=-1)

            out = einsum('bihqk,bkihd->bqihd', attn, value).reshape(B, N, L, -1)
            out = gate * out

            out = self.to_out(out)

        return out

class MSAColGlobalAttention(nn.Module):
    def __init__(self, d_msa=64, n_head=8, d_hidden=8):
        super(MSAColGlobalAttention, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        #
        self.to_q = nn.Linear(d_msa, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_v = nn.Linear(d_msa, d_hidden, bias=False)
        self.to_g = nn.Linear(d_msa, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_msa)

        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, msa, stride):
        B, N, L = msa.shape[:3]

        #
        dtype = msa.dtype
        msa = self.norm_msa(msa).to(dtype=dtype)
        #
        query = self.to_q(msa).reshape(B, N, L, self.h, self.dim)
        query = query.mean(dim=1) # (B, L, h, dim)
        key = self.to_k(msa) # (B, N, L, dim)
        value = self.to_v(msa) # (B, N, L, dim)
        gate = torch.sigmoid(self.to_g(msa)) # (B, N, L, h*dim)
        #
        query = query * self.scaling
        attn = einsum('bihd,bkid->bihk', query, key) # (B, L, h, N)
        attn = F.softmax(attn, dim=-1)
        #
        out = einsum('bihk,bkid->bihd', attn, value).reshape(B, 1, L, -1) # (B, 1, L, h*dim)
        out = gate * out # (B, N, L, h*dim)
        #
        out = self.to_out(out)
        return out

# Instead of triangle attention, use Tied axail attention with bias from coordinates..?
class BiasedAxialAttention(nn.Module):
    def __init__(self, d_pair, d_bias, n_head, d_hidden, p_drop=0.1, is_row=True):
        super(BiasedAxialAttention, self).__init__()
        #
        self.is_row = is_row
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_bias = nn.LayerNorm(d_bias)

        self.to_q = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_k = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_v = nn.Linear(d_pair, n_head*d_hidden, bias=False)
        self.to_b = nn.Linear(d_bias, n_head, bias=False) 
        self.to_g = nn.Linear(d_pair, n_head*d_hidden)
        self.to_out = nn.Linear(n_head*d_hidden, d_pair)
        
        self.scaling = 1/math.sqrt(d_hidden)
        self.h = n_head
        self.dim = d_hidden
        self.dim_out = d_pair

        # initialize all parameters properly
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Glorot uniform / Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        # bias: normal distribution
        self.to_b = init_lecun_normal(self.to_b)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.to_g.weight)
        nn.init.ones_(self.to_g.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, pair, bias, stride):
        B, L = pair.shape[:2] # after subunit mask is applied

        # pair: (B, L, L, d_pair)
        if self.is_row:
            pair = pair.permute(0,2,1,3)
            bias = bias.permute(0,2,1,3)

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        if STRIDE > 0 and STRIDE < L:
            attn = torch.zeros((B,L,L,self.h), device=pair.device, dtype=pair.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=pair.device)

                pair_i = self.norm_pair(pair[:,rows])
                query = self.to_q(pair_i).reshape(B, -1, L, self.h, self.dim)
                query *= self.scaling
                key = self.to_k(pair_i).reshape(B, -1, L, self.h, self.dim)
                key = key / L # normalize for tied attention
                attn_i = einsum('bnihk,bnjhk->bijh', query, key)
                attn +=  attn_i 

            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=pair.device)
                bias_i = self.norm_bias(bias[:,rows]).to(dtype=bias.dtype)
                attn[:,rows] += self.to_b(bias_i) # (B, STRIDE, L, h)

            attn = F.softmax(attn, dim=-2) # (B, L, L, h)

            out = torch.zeros((B,L,L,self.dim_out), device=pair.device, dtype=pair.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=pair.device)

                pair_i = self.norm_pair(pair[:,rows])
                value_i = self.to_v(pair_i).reshape(B, -1, L, self.h, self.dim)

                for j in range((L-1)//STRIDE+1):
                    cols = torch.arange(j*STRIDE, min((j+1)*STRIDE, L), device=pair.device)
                    NC,NR = cols.shape[0], rows.shape[0]
                    out_ij = einsum('bijh,bnjhd->bnihd', attn[:,cols], value_i).reshape(B, NR, NC, -1)
                    gate_ij = torch.sigmoid(self.to_g(pair_i[:,:,cols]))
                    out[:,rows[:,None],cols[None,:]] = self.to_out( gate_ij*out_ij )
        else:
            pair = self.norm_pair(pair)
            bias = self.norm_bias(bias)

            query = self.to_q(pair).reshape(B, L, L, self.h, self.dim)
            key = self.to_k(pair).reshape(B, L, L, self.h, self.dim)
            value = self.to_v(pair).reshape(B, L, L, self.h, self.dim)
            bias = self.to_b(bias) # (B, L, L, h)
            gate = torch.sigmoid(self.to_g(pair)) # (B, L, L, h*dim) 
        
            query = query * self.scaling
            key = key / L # normalize for tied attention
            attn = einsum('bnihk,bnjhk->bijh', query, key) # tied attention
            attn = attn + bias
            attn = F.softmax(attn, dim=-2) # (B, L, L, h)
        
            out = einsum('bijh,bnjhd->bnihd', attn, value).reshape(B, L, L, -1)
            out = gate * out
        
            out = self.to_out(out)

        if self.is_row:
            out = out.permute(0,2,1,3)

        return out

class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair, d_hidden=128, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.left_proj = nn.Linear(d_pair, d_hidden)
        self.right_proj = nn.Linear(d_pair, d_hidden)
        self.left_gate = nn.Linear(d_pair, d_hidden)
        self.right_gate = nn.Linear(d_pair, d_hidden)
        #
        self.gate = nn.Linear(d_pair, d_pair)
        self.norm_out = nn.LayerNorm(d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_pair)

        self.d_hidden = d_hidden
        self.d_out = d_pair

        self.outgoing = outgoing
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal distribution for regular linear weights
        self.left_proj = init_lecun_normal(self.left_proj)
        self.right_proj = init_lecun_normal(self.right_proj)
        
        # Set Bias of Linear layers to zeros
        nn.init.zeros_(self.left_proj.bias)
        nn.init.zeros_(self.right_proj.bias)

        # gating: zero weights, one biases (mostly open gate at the begining)
        nn.init.zeros_(self.left_gate.weight)
        nn.init.ones_(self.left_gate.bias)
        
        nn.init.zeros_(self.right_gate.weight)
        nn.init.ones_(self.right_gate.bias)
        
        nn.init.zeros_(self.gate.weight)
        nn.init.ones_(self.gate.bias)

        # to_out: right before residual connection: zero initialize -- to make it sure residual operation is same to the Identity at the begining
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, pair, stride):
        B,L = pair.shape[:2]

        # fd reduce memory in inference
        STRIDE = L
        if (not self.training and stride>0):
            STRIDE = stride

        if STRIDE > 0 and STRIDE < L:
            out = torch.zeros((B,L,L,self.d_out), device=pair.device, dtype=pair.dtype)
            for i in range((L-1)//STRIDE+1):
                rows = torch.arange(i*STRIDE, min((i+1)*STRIDE, L), device=pair.device)

                if self.outgoing:
                    pair_i = self.norm(pair[:,rows,:]).to(dtype=pair.dtype)
                    left = self.left_proj(pair_i) # (B, L, L, d_h)
                    left_gate = torch.sigmoid(self.left_gate(pair_i))
                    left = left_gate * left
                else:
                    pair_i = self.norm(pair[:,:,rows]).to(dtype=pair.dtype)
                    left = self.left_proj(pair_i) # (B, L, L, d_h)
                    left_gate = torch.sigmoid(self.left_gate(pair_i))
                    left = left_gate * left

                for j in range((L-1)//STRIDE+1):
                  cols = torch.arange(j*STRIDE, min((j+1)*STRIDE, L), device=pair.device)

                  if self.outgoing:
                      pair_i = self.norm(pair[:,cols,:]).to(dtype=pair.dtype)
                      right = self.right_proj(pair_i) # (B, L, L, d_h)
                      right_gate = torch.sigmoid(self.right_gate(pair_i))
                      right = right_gate * right

                      out_ij = einsum('bikd,bjkd->bijd', left, right/float(L))
                  else:
                      pair_i = self.norm(pair[:,:,cols]).to(dtype=pair.dtype)
                      right = self.right_proj(pair_i) # (B, L, L, d_h)
                      right_gate = torch.sigmoid(self.right_gate(pair_i))
                      right = right_gate * right

                      out_ij = einsum('bkid,bkjd->bijd', left, right/float(L))

                  out_ij = self.norm_out(out_ij)
                  out_ij = self.out_proj(out_ij)

                  pair_ij = self.norm( pair[:,rows[:,None],cols[None,:]] ).to(dtype=pair.dtype)
                  gate = torch.sigmoid(self.gate(pair_ij)) # (B, L, L, d_pair)
                  out[:,rows[:,None],cols[None,:]] = (gate * out_ij)
        else:
            pair = self.norm(pair)

            left = self.left_proj(pair) # (B, L, L, d_h)
            left_gate = torch.sigmoid(self.left_gate(pair))
            left = left_gate * left
        
            right = self.right_proj(pair) # (B, L, L, d_h)
            right_gate = torch.sigmoid(self.right_gate(pair))
            right = right_gate * right
        
            if self.outgoing:
                out = einsum('bikd,bjkd->bijd', left, right/float(L))
            else:
                out = einsum('bkid,bkjd->bijd', left, right/float(L))
            out = self.norm_out(out)
            out = self.out_proj(out)

            gate = torch.sigmoid(self.gate(pair)) # (B, L, L, d_pair)
            out = gate * out

        return out