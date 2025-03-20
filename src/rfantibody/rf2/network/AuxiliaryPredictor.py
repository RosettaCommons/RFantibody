import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        #
        self.proj_symm = nn.Linear(n_feat, 37*2)
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:37].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, 21)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

class LDDTNetwork(nn.Module):
    def __init__(self, n_feat, d_hidden=128, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.norm = nn.LayerNorm(n_feat)
        self.linear_1 = nn.Linear(n_feat, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.proj = nn.Linear(d_hidden, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_2.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x):
        x = F.relu_(self.linear_2(F.relu_(self.linear_1(self.norm(x))))) 
        logits = self.proj(x) # (B, L, 50)

        return logits.permute(0,2,1)

class ExpResolvedNetwork(nn.Module):
    def __init__(self, d_msa, d_state, p_drop=0.1):
        super(ExpResolvedNetwork, self).__init__()
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_state = nn.LayerNorm(d_state)
        self.proj = nn.Linear(d_msa+d_state, 1)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, seq, state):
        B, L = seq.shape[:2]
        
        seq = self.norm_msa(seq)
        state = self.norm_state(state)
        feat = torch.cat((seq, state), dim=-1)
        logits = self.proj(feat)
        return logits.reshape(B, L)

class PAENetwork(nn.Module):
    def __init__(self, n_feat, n_bin_pae=64):
        super(PAENetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_pae)
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, L, 64)

        return logits.permute(0,3,1,2)

class BinderNetwork(nn.Module):
    def __init__(self, n_hidden=64, n_bin_pae=64):
        super(BinderNetwork, self).__init__()
        #self.proj = nn.Linear(n_bin_pae, n_hidden)
        #self.classify = torch.nn.Linear(2*n_hidden, 1)
        self.classify = torch.nn.Linear(n_bin_pae, 1)
        self.reset_parameter()

    def reset_parameter(self):
        #nn.init.zeros_(self.proj.weight)
        #nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.classify.weight)
        nn.init.zeros_(self.classify.bias)

    def forward(self, pae, same_chain):
        #logits = self.proj( pae.permute(0,2,3,1) )
        logits = pae.permute(0,2,3,1)
        #logits_intra = torch.mean( logits[same_chain==1], dim=0 )
        logits_inter = torch.mean( logits[same_chain==0], dim=0 ).nan_to_num() # all zeros if single chain
        #prob = torch.sigmoid( self.classify( torch.cat((logits_intra,logits_inter)) ) )
        prob = torch.sigmoid( self.classify( logits_inter ) )
        return prob

class v2_BinderNetwork(nn.Module):
    def __init__(self, d_pair=128, d_state=32, d_rbf=64):
        super(v2_BinderNetwork, self).__init__()

        self.rbf2attn = nn.Linear(d_rbf, 1)
        self.downsample = torch.nn.Linear(d_pair+2*d_state, 1)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.downsample.weight)
        nn.init.zeros_(self.downsample.bias)
        nn.init.zeros_(self.rbf2attn.weight)
        nn.init.zeros_(self.rbf2attn.bias)

    def forward(self, pair, rbf_feat, state, same_chain):
        B, L = pair.shape[:2]

        # 1. get attention map
        # pair: (B, L, L, d_pair)
        attn = self.rbf2attn(rbf_feat)

        # 2. get logits
        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)
        logits = self.downsample( torch.cat((pair, left, right), dim=-1) )

        # 3. dot product
        if (torch.sum(same_chain==0)==0):
            logits = logits.flatten()
            attn = attn.flatten()
        else:
            logits = logits[same_chain==0]
            attn = attn[same_chain==0]

        attn = F.softmax(attn, dim=-1)

        logits_inter = torch.mean( logits * attn, dim=0 ).nan_to_num() # all zeros if single chain

        prob = torch.sigmoid( logits_inter ).double()

        if (torch.sum(same_chain==0)==0):
            prob = prob.unsqueeze(0)

        return prob
