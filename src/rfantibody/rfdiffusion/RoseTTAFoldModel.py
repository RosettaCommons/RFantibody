import torch
import torch.nn as nn
from opt_einsum import contract as einsum

from rfantibody.rfdiffusion.AuxiliaryPredictor import (
    DistanceNetwork,
    ExpResolvedNetwork,
    LDDTNetwork,
    MaskedTokenNetwork,
)
from rfantibody.rfdiffusion.Embeddings import (
    Extra_emb,
    MSA_emb,
    Recycling,
    Templ_emb,
    Timestep_emb,
)
from rfantibody.rfdiffusion.Track_module import IterativeSimulator
from rfantibody.rfdiffusion.util import INIT_CRDS


class RoseTTAFoldModule(nn.Module):
    def __init__(self, 
                 n_extra_block, 
                 n_main_block, 
                 n_ref_block,
                 d_msa,
                 d_msa_full,
                 d_pair,
                 d_templ,
                 n_head_msa,
                 n_head_pair,
                 n_head_templ,
                 d_hidden,
                 d_hidden_templ,
                 p_drop,
                 d_t1d,
                 d_t2d,
                 d_time_emb, # total dims for input timestep emb
                 d_time_emb_proj, # size of projected timestep emb
                 T, # total timesteps (used in timestep emb
                 use_motif_timestep, # Whether to have a distinct emb for motif
                 freeze_track_motif, # Whether to freeze updates to motif in track
                 use_selfcond_emb, # Whether to use self-conditioning embedding
                 SE3_param_full={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 SE3_param_topk={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32},
                 input_seq_onehot=False,     # For continuous vs. discrete sequence - NRB,
                 ):

        super(RoseTTAFoldModule, self).__init__()

        self.freeze_track_motif = freeze_track_motif

        # Input Embeddings
        d_state = SE3_param_topk['l0_out_features']
        self.latent_emb = MSA_emb(d_msa=d_msa, d_pair=d_pair, d_state=d_state,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq - NRB
        self.full_emb = Extra_emb(d_msa=d_msa_full, d_init=25,
                p_drop=p_drop, input_seq_onehot=input_seq_onehot) # Allowed to take onehotseq - NRB

        self.templ_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)
        if use_selfcond_emb:
            # Make a second template embedding module just for self-conditioning
            self.selfcond_emb = Templ_emb(d_pair=d_pair, d_templ=d_templ, d_state=d_state,
                                           n_head=n_head_templ,
                                           d_hidden=d_hidden_templ, p_drop=0.25, d_t1d=d_t1d, d_t2d=d_t2d)

        # timestep embedder 
        if d_time_emb:
            print('NOTE: Using sinusoidal timestep embeddings of dim ',d_time_emb, ' projected to dim ',d_time_emb_proj)
            assert d_t1d >= 22 + d_time_emb_proj, 'timestep projection size doesn\'t fit into RF t1d projection layers'
            self.timestep_embedder = Timestep_emb(input_size=d_time_emb, 
                                                  output_size=d_time_emb_proj,
                                                  T=T,
                                                  use_motif_timestep=use_motif_timestep)


        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(n_extra_block=n_extra_block,
                                            n_main_block=n_main_block,
                                            n_ref_block=n_ref_block,
                                            d_msa=d_msa, d_msa_full=d_msa_full,
                                            d_pair=d_pair, d_hidden=d_hidden,
                                            n_head_msa=n_head_msa,
                                            n_head_pair=n_head_pair,
                                            SE3_param_full=SE3_param_full,
                                            SE3_param_topk=SE3_param_topk,
                                            p_drop=p_drop)
        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
       
        self.exp_pred = ExpResolvedNetwork(d_msa, d_state)

    def forward(self, msa_latent, msa_full, seq, xyz, idx, t,
                t1d=None, t2d=None, xyz_t=None, alpha_t=None,
                sc2d=None, xyz_sc=None,
                msa_prev=None, pair_prev=None, state_prev=None,
                return_raw=False, return_full=False, return_infer=False, return_w_msa_prev=False,
                use_checkpoint=False, motif_mask=None, i_cycle=None, n_cycle=None):

        B, N, L = msa_latent.shape[:3]
        # Get embeddings
        msa_latent, pair, state = self.latent_emb(msa_latent, seq, idx)
        msa_full = self.full_emb(msa_full, seq, idx)

        # Do recycling
        if msa_prev == None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
        if pair_prev == None:
            pair_prev = torch.zeros_like(pair)
        if state_prev == None:    
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(seq, msa_prev, pair_prev, xyz, state_prev)
        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle


        # Get timestep embedding (if using)
        if hasattr(self, 'timestep_embedder'):
            assert t is not None
            time_emb = self.timestep_embedder(L,t,motif_mask)
            n_tmpl = t1d.shape[1]
            t1d = torch.cat([t1d, time_emb[None,None,...].repeat(1,n_tmpl,1,1)], dim=-1)

        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, pair, state, use_checkpoint=use_checkpoint)

        # If active, add self-conditioning embedding
        if hasattr(self, 'selfcond_emb'):
            assert(sc2d is not None and xyz_sc is not None), \
                'Self-conditioning inputs not provided but self-cond embedding module is active'

            alpha_sc = torch.zeros_like(alpha_t)
            pair, state = self.selfcond_emb(t1d, sc2d, alpha_sc, xyz_sc, pair, state, use_checkpoint=use_checkpoint)
        
        # Predict coordinates from given inputs
        is_frozen_residue = motif_mask if self.freeze_track_motif else torch.zeros_like(motif_mask).bool()
        msa, pair, R, T, alpha_s, state = self.simulator(seq, msa_latent, msa_full, pair, xyz[:,:,:3],
                                                         state, idx, use_checkpoint=use_checkpoint,
                                                         motif_mask=is_frozen_residue)
        
        if return_raw:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            return msa[:,0], pair, xyz, state, alpha_s[-1]

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)
        
        # Predict LDDT
        lddt = self.lddt_pred(state)

        if return_infer:
            # get last structure
            xyz = einsum('bnij,bnaj->bnai', R[-1], xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T[-1].unsqueeze(-2)
            
            # get scalar plddt
            nbin = lddt.shape[1]
            bin_step = 1.0 / nbin
            lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=lddt.dtype, device=lddt.device)
            pred_lddt = nn.Softmax(dim=1)(lddt)
            pred_lddt = torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

            return msa[:,0], pair, xyz, state, alpha_s[-1], logits_aa.permute(0,2,1), pred_lddt

        #
        # predict distogram & orientograms
        logits = self.c6d_pred(pair)
        
        # predict experimentally resolved or not
        logits_exp = self.exp_pred(msa[:,0], state)
        
        # get all intermediate bb structures
        xyz = einsum('rbnij,bnaj->rbnai', R, xyz[:,:,:3]-xyz[:,:,1].unsqueeze(-2)) + T.unsqueeze(-2)

        # for msa_prev self-conditioning
        if return_w_msa_prev:
            return logits, logits_aa, logits_exp, xyz, alpha_s, lddt, msa[:,0], pair, state

        return logits, logits_aa, logits_exp, xyz, alpha_s, lddt
