import sys
import os
import time
from collections import namedtuple

import torch
import torch.nn as nn
import numpy as np

from rfantibody.rf2.network.parsers import parse_a3m, read_templates, read_template_pdb
from rfantibody.rf2.network.RoseTTAFoldModel  import RoseTTAFoldModule
from rfantibody.rf2.network import util
from rfantibody.rf2.network.ffindex import *
from rfantibody.rf2.network.featurizing import MSAFeaturize
from rfantibody.rf2.network.kinematics import xyz_to_t2d
from rfantibody.rf2.network.chemical import INIT_CRDS
from rfantibody.rf2.network.util_module import XYZConverter
from rfantibody.rf2.network.symmetry import symm_subunit_matrix, find_symm_subs

# suppress dgl warning w/ newest pytorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    default_model = os.path.dirname(__file__)+"/weights/RF2_jan24.pt"

    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold2NA")
    parser.add_argument("-inputs", help="R|Input data in format A:B:C, with\n"
         "   A = multiple sequence alignment file\n"
         "   B = hhpred hhr file\n"
         "   C = hhpred atab file\n"
         "Spaces seperate multiple inputs.  The last two arguments may be omitted\n",
         required=True, nargs='+')
    parser.add_argument("-db", help="HHpred database location", default=None)
    parser.add_argument("-prefix", default="S", type=str, help="Output file prefix [S]")
    parser.add_argument("-symm", default="C1", help="Symmetry group (Cn,Dn,T,O, or I).  If provided, 'input' should cover the asymmetric unit. [C1]")
    parser.add_argument("-model", default=default_model, help="Model weights. [weights/RF2_jan24.pt]")
    parser.add_argument("-n_recycles", default=3, type=int, help="Number of recycles to use [3].")
    parser.add_argument("-n_models", default=1, type=int, help="Number of models to predict [1].")
    parser.add_argument("-subcrop", default=-1, type=int, help="Subcrop pair-to-pair updates. A value of -1 means no subcropping. [-1]")
    parser.add_argument("-topk", default=1536, type=int, help="Limit number of residue-pair neighbors in structure updates. A value of -1 means no subcropping. [2048]")
    parser.add_argument("-low_vram", default=False, help="Offload some computations to CPU to allow larger systems in low VRAM. [False]", action='store_true')
    parser.add_argument("-nseqs", default=256, type=int, help="The number of MSA sequences to sample in the main 1D track [256].")
    parser.add_argument("-nseqs_full", default=2048, type=int, help="The number of MSA sequences to sample in the wide-MSA 1D track [2048].")
    parser.add_argument("-cyclize", default=False, help="Model as N-C cyclized peptide", action='store_true')
    args = parser.parse_args()
    return args

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 36,
        "n_ref_block"     : 4,
        "d_msa"           : 256 ,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 32,
        "p_drop"       : 0.0,
        }

SE3_param_full = {
        "num_layers"    : 1,
        "num_channels"  : 48,
        "num_degrees"   : 2,
        "l0_in_features": 32,
        "l0_out_features": 32,
        "l1_in_features": 2,
        "l1_out_features": 2,
        "num_edge_features": 32,
        "div": 4,
        "n_heads": 4
        }

SE3_param_topk = {
        "num_layers"    : 1,
        "num_channels"  : 128,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 2,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param_full'] = SE3_param_full
MODEL_PARAM['SE3_param_topk'] = SE3_param_topk

def get_striping_parameters(low_vram=False):
    stripe = {
        "msa2msa":1024,
        "msa2pair":1024,
        "pair2pair":1024,
        "str2str":1024,
        "iter":1024,
        "ff_m2m":1024,
        "ff_p2p":1024,
        "ff_s2s":1024,
        "attn":1024,
        "msarow_n":1024,
        "msarow_l":1024,
        "msacol":1024,
        "biasedax":512,
        "trimult":512,
        "recycl":1024,
        "msa_emb":1024,
        "templ_emb":1024,
        "templ_pair":1024,
        "templ_attn":1024,
    }

    # adjust for low vram
    if (low_vram):
        # msa2msa
        stripe['msa2msa'] = 256
        stripe['msarow_n'] = 256
        stripe['msarow_l'] = 256
        stripe['msacol'] = 256
        stripe['ff_m2m'] = 256

        # pair2pair
        stripe['pair2pair'] = 256
        stripe['ff_p2p'] = 256
        stripe['biasedax'] = 128
        stripe['trimult'] = 128

        stripe['recycl'] = 512

    return stripe

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

def merge_a3m_homo(msa_orig, ins_orig, nmer, mode="default"):
    N, L = msa_orig.shape[:2]
    if mode == "repeat":

        # AAAAAA
        # AAAAAA

        msa = torch.tile(msa_orig,(1,nmer))
        ins = torch.tile(ins_orig,(1,nmer))

    elif mode == "diag":

        # AAAAAA
        # A-----
        # -A----
        # --A---
        # ---A--
        # ----A-
        # -----A

        N = N - 1
        new_N = 1 + N * nmer
        new_L = L * nmer
        msa = torch.full((new_N, new_L), 20, dtype=msa_orig.dtype, device=msa_orig.device)
        ins = torch.full((new_N, new_L), 0, dtype=ins_orig.dtype, device=msa_orig.device)

        start_L = 0
        start_N = 1
        for i_c in range(nmer):
            msa[0, start_L:start_L+L] = msa_orig[0] 
            msa[start_N:start_N+N, start_L:start_L+L] = msa_orig[1:]
            ins[0, start_L:start_L+L] = ins_orig[0]
            ins[start_N:start_N+N, start_L:start_L+L] = ins_orig[1:]
            start_L += L
            start_N += N
    else:

        # AAAAAA
        # A-----
        # -AAAAA

        msa = torch.full((2*N-1, L*nmer), 20, dtype=msa_orig.dtype, device=msa_orig.device)
        ins = torch.full((2*N-1, L*nmer), 0, dtype=ins_orig.dtype, device=msa_orig.device)

        msa[:N, :L] = msa_orig
        ins[:N, :L] = ins_orig
        start = L

        for i_c in range(1,nmer):
            msa[0, start:start+L] = msa_orig[0] 
            msa[N:, start:start+L] = msa_orig[1:]
            ins[0, start:start+L] = ins_orig[0]
            ins[N:, start:start+L] = ins_orig[1:]
            start += L        

    return msa, ins

class Predictor():
    def __init__(self, model_weights, device="cuda:0", model_param=None):
        # define model name
        self.model_weights = model_weights
        if not os.path.exists(model_weights):
            self.model_weights = model_weights

        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model_param = MODEL_PARAM if model_param is None else model_param

        self.model = RoseTTAFoldModule(
            **self.model_param,
        ).to(self.device)

        could_load = self.load_model(self.model_weights)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        # from xyz to get xxxx or from xxxx to xyz
        self.l2a = util.long2alt.to(self.device)
        self.aamask = util.allatom_mask.to(self.device)
        self.lddt_bins = torch.linspace(1.0/50, 1.0, 50) - 1.0/100
        self.xyz_converter = XYZConverter()


    def load_model(self, model_weights):
        if not os.path.exists(model_weights):
            print(f"ERROR: model weights {model_weights} not found")
            return False
        checkpoint = torch.load(model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)

        return True

    def predict(
        self, inputs, out_prefix, symm="C1", ffdb=None,
        n_recycles=4, n_models=1, subcrop=-1, topk=-1, low_vram=False, nseqs=256, nseqs_full=2048,
        n_templ=4, msa_mask=0.0, is_training=False, msa_concat_mode="diag", cyclize=False
    ):
        def to_ranges(txt):
            return [[int(x) for x in r.strip().split('-')]
                    for r in txt.strip().split(',')]

        self.xyz_converter = self.xyz_converter.cpu()

        ###
        # pass 1, combined MSA
        Ls_blocked, Ls, msas, inss = [], [], [], []
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            a3m_i = fseq_i[0]

            a3m_i =  a3m_i.split('[')
            a3m_range = None
            if (len(a3m_i)>1):
              assert a3m_i[1][-1] == ']'
              a3m_range = to_ranges(a3m_i[1][:-1])
            a3m_i = a3m_i[0]

            msa_i, ins_i, Ls_i = parse_a3m(a3m_i)
            msa_i = torch.tensor(msa_i).long()
            ins_i = torch.tensor(ins_i).long()
            if (msa_i.shape[0] > nseqs_full):
                idxs_tokeep = np.random.permutation(msa_i.shape[0])[:nseqs_full]
                idxs_tokeep[0] = 0  # keep best
                msa_i = msa_i[idxs_tokeep]
                ins_i = ins_i[idxs_tokeep]

            if (a3m_range is not None):
                a3m_mask = torch.zeros(sum(Ls_i), dtype=torch.bool)

                Ls_new = []
                for i in a3m_range:
                    if len(i) == 1:
                        a3m_mask[i[0]-1] = True
                        Ls_new.append(1)
                    else:
                        a3m_mask[(i[0]-1):i[1]] = True
                        Ls_new.append(i[1]-i[0]+1)

                msa_i = msa_i[:,a3m_mask]
                ins_i = ins_i[:,a3m_mask]
                start_i = 0

                Ls_i = Ls_new

            msas.append(msa_i)
            inss.append(ins_i)
            Ls.extend(Ls_i)
            Ls_blocked.append(msa_i.shape[1])

        msa_orig = {'msa':msas[0],'ins':inss[0]}
        for i in range(1,len(Ls_blocked)):
            msa_orig = util.merge_a3m_hetero(msa_orig, {'msa':msas[i],'ins':inss[i]}, [sum(Ls_blocked[:i]),Ls_blocked[i]])
        msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

        # pseudo symmetry
        if symm.startswith("X"):
            Osub = int(symm[1:])
            if Osub > 1:
                msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub, mode=msa_concat_mode)
                Ls = sum([Ls] * Osub,[])
            symm = "C1"

        symmids,symmRs,symmmeta,symmoffset = symm_subunit_matrix(symm)
        O = symmids.shape[0]

        ###
        # pass 2, templates
        L = sum(Ls)
        #xyz_t = INIT_CRDS.reshape(1,1,27,3).repeat(n_templ,L,1,1) + torch.rand(n_templ,L,1,3)*5.0 - 2.5
        # dummy template
        SYMM_OFFSET_SCALE = 1.0
        xyz_t = (
            INIT_CRDS.reshape(1,1,27,3).repeat(n_templ,L,1,1) 
            + torch.rand(n_templ,L,1,3)*5.0 - 2.5
            + SYMM_OFFSET_SCALE*symmoffset*L**(1/2)  # note: offset based on symmgroup
        )

        mask_t = torch.full((n_templ, L, 27), False) 
        t1d = torch.nn.functional.one_hot(torch.full((n_templ, L), 20).long(), num_classes=21).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((n_templ,L,1)).float()), -1)

        maxtmpl=1
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            if (len(fseq_i) == 3):
                hhr_i,atab_i = fseq_i[1:3]
                startres,stopres = sum(Ls_blocked[:i]), sum(Ls_blocked[:(i+1)])
                xyz_t_i, t1d_i, mask_t_i = read_templates(Ls_blocked[i], ffdb, hhr_i, atab_i, n_templ=n_templ)
                ntmpl_i = xyz_t_i.shape[0]
                maxtmpl = max(maxtmpl, ntmpl_i)
                xyz_t[:ntmpl_i,startres:stopres,:,:] = xyz_t_i
                t1d[:ntmpl_i,startres:stopres,:] = t1d_i
                mask_t[:ntmpl_i,startres:stopres,:] = mask_t_i

            elif (len(fseq_i) == 2):
                templ_fn = fseq_i[1]
                startres,stopres = sum(Ls_blocked[:i]), sum(Ls_blocked[:(i+1)])
                xyz_t_i, t1d_i, mask_t_i = read_template_pdb(Ls_blocked[i], templ_fn, align_conf=0.2)
                ntmpl_i = 1
                xyz_t[:ntmpl_i,startres:stopres,:,:] = xyz_t_i
                t1d[:ntmpl_i,startres:stopres,:] = t1d_i
                mask_t[:ntmpl_i,startres:stopres,:] = mask_t_i

        same_chain = torch.zeros((1,L,L), dtype=torch.bool, device=xyz_t.device)
        stopres = 0
        for i in range(1,len(Ls)):
            startres,stopres = sum(Ls[:(i-1)]), sum(Ls[:i])
            same_chain[:,startres:stopres,startres:stopres] = True
        same_chain[:,stopres:,stopres:] = True

        # template features
        xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
        mask_t = mask_t[:maxtmpl].unsqueeze(0)
        t1d = t1d[:maxtmpl].float().unsqueeze(0)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, mask_in=mask_t.reshape(-1,L,27))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,10,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3*10)

        ###
        # pass 3, symmetry
        xyz_prev = xyz_t[:,0]
        xyz_prev, symmsub = find_symm_subs(xyz_prev[:,:L],symmRs,symmmeta)

        Osub = symmsub.shape[0]
        mask_t = mask_t.repeat(1,1,Osub,1)
        alpha_t = alpha_t.repeat(1,1,Osub,1)
        mask_prev = mask_t[:,0]
        xyz_t = xyz_t.repeat(1,1,Osub,1,1)
        t1d = t1d.repeat(1,1,Osub,1)

        # symmetrize msa
        effL = Osub*L
        if (Osub>1):
            msa_orig, ins_orig = merge_a3m_homo(msa_orig, ins_orig, Osub, mode=msa_concat_mode)

        # index
        idx_pdb = torch.arange(Osub*L)[None,:]

        same_chain = torch.zeros((1,Osub*L,Osub*L)).long()
        i_start = 0
        for o_i in range(Osub):
            for li in Ls:
                i_stop = i_start + li
                idx_pdb[:,i_stop:] += 100
                same_chain[:,i_start:i_stop,i_start:i_stop] = 1
                i_start = i_stop

        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float()*same_chain.float()[:,None] # (ignore inter-chain region)

        if is_training:
            self.model.train()
        else:
            self.model.eval()
        for i_trial in range(n_models):
            #if os.path.exists("%s_%02d_init.pdb"%(out_prefix, i_trial)):
            #    continue
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            self.run_prediction(
                msa_orig, ins_orig, 
                t1d, xyz_t, alpha_t, mask_t_2d, 
                xyz_prev, mask_prev, same_chain, idx_pdb,
                symmids, symmsub, symmRs, symmmeta,  Ls, 
                n_recycles, nseqs, nseqs_full, subcrop, topk, low_vram, cyclize,
                "%s_%02d"%(out_prefix, i_trial),
                msa_mask=msa_mask
            )
            runtime = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"runtime={runtime:.2f} vram={vram:.2f}")
            torch.cuda.empty_cache()

    def run_prediction(
        self, msa_orig, ins_orig, 
        t1d, xyz_t, alpha_t, mask_t, 
        xyz_prev, mask_prev, same_chain, idx_pdb, 
        symmids, symmsub, symmRs, symmmeta, L_s, 
        n_recycles, nseqs, nseqs_full, subcrop, topk, low_vram, cyclize, out_prefix,
        msa_mask=0.0,
    ):
        self.xyz_converter = self.xyz_converter.to(self.device)
        self.lddt_bins = self.lddt_bins.to(self.device)

        STRIPE = get_striping_parameters(low_vram)

        with torch.no_grad():
            msa = msa_orig.long().to(self.device) # (N, L)
            ins = ins_orig.long().to(self.device)

            print(f"N={msa.shape[0]} L={msa.shape[1]}")
            N, L = msa.shape[:2]
            O = symmids.shape[0]
            Osub = symmsub.shape[0]
            Lasu = L//Osub

            B = 1
            #
            t1d = t1d.to(self.device).half()
            t2d = xyz_to_t2d(xyz_t, mask_t).half()
            if not low_vram:
                t2d = t2d.to(self.device) #.half()
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t[:,:,:,1].to(self.device)
            mask_t = mask_t.to(self.device)
            alpha_t = alpha_t.to(self.device)
            xyz_prev = xyz_prev.to(self.device)
            mask_prev = mask_prev.to(self.device)
            same_chain = same_chain.to(self.device)
            symmids = symmids.to(self.device)
            symmsub = symmsub.to(self.device)
            symmRs = symmRs.to(self.device)

            subsymms, _ = symmmeta
            for i in range(len(subsymms)):
                subsymms[i] = subsymms[i].to(self.device)

            msa_prev=None
            pair_prev=None
            state_prev=None
            mask_recycle = mask_prev[:,:,:3].bool().all(dim=-1)
            mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
            mask_recycle = same_chain.float()*mask_recycle.float()

            best_lddt = torch.tensor([-1.0], device=self.device)
            best_xyz = None
            best_logit = None
            best_pae = None

            for i_cycle in range(n_recycles + 1):
                seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                    msa, ins, p_mask=msa_mask, params={'MAXLAT': nseqs, 'MAXSEQ': nseqs_full, 'MAXCYCLE': 1})

                seq = seq.unsqueeze(0)
                msa_seed = msa_seed.unsqueeze(0)
                msa_extra = msa_extra.unsqueeze(0)

                #fd memory savings
                msa_seed = msa_seed.half()  # GPU ONLY
                msa_extra = msa_extra.half()  # GPU ONLY

                xyz_prev_prev = xyz_prev.clone()

                with torch.cuda.amp.autocast(True):
                    logit_s, _, _, logits_pae, p_bind, xyz_prev, alpha, symmsub, pred_lddt, msa_prev, pair_prev, state_prev = self.model(
                                                               msa_seed, msa_extra,
                                                               seq, xyz_prev, 
                                                               idx_pdb,
                                                               t1d=t1d, t2d=t2d, xyz_t=xyz_t,
                                                               alpha_t=alpha_t, mask_t=mask_t,
                                                               same_chain=same_chain,
                                                               msa_prev=msa_prev,
                                                               pair_prev=pair_prev,
                                                               state_prev=state_prev,
                                                               p2p_crop=subcrop,
                                                               topk_crop=topk,
                                                               mask_recycle=mask_recycle,
                                                               symmids=symmids,
                                                               symmsub=symmsub,
                                                               symmRs=symmRs,
                                                               symmmeta=symmmeta, 
                                                               striping=STRIPE,
                                                               nc_cycle=cyclize )
                    alpha = alpha[-1].to(seq.device)
                    xyz_prev = xyz_prev[-1].to(seq.device)
                    _, xyz_prev = self.xyz_converter.compute_all_atom(seq, xyz_prev, alpha)

                mask_recycle=None
                pair_prev = pair_prev.cpu()
                msa_prev = msa_prev.cpu()

                pred_lddt = nn.Softmax(dim=1)(pred_lddt.half()) * self.lddt_bins[None,:,None]
                pred_lddt = pred_lddt.sum(dim=1)
                logits_pae = pae_unbin(logits_pae.half())

                # TODO: RMSD
                #rmsd,_,_,_ = calc_rmsd(xyz_prev_prev[None].float(), xyz_prev.float(), torch.ones((1,L,27),dtype=torch.bool))

                print (f"recycle {i_cycle} plddt {pred_lddt.mean():.3f} pae {logits_pae.mean():.3f} rmsd {rmsd[0]:.3f}")

                torch.cuda.empty_cache()
                if pred_lddt.mean() < best_lddt.mean():
                    pred_lddt, logits_pae, logit_s = None, None, None
                    continue

                best_xyz = xyz_prev
                best_logit = logit_s
                best_lddt = pred_lddt.half().cpu()
                best_pae = logits_pae.half().cpu()
                best_logit = [l.half().cpu() for l in logit_s]
                pred_lddt, logits_pae, logit_s = None, None, None

            # free more memory
            pair_prev, msa_prev, t2d = None, None, None

            prob_s = list()
            for logit in best_logit:
                prob = self.active_fn(logit.to(self.device).float()) # distogram
                prob_s.append(prob.half().cpu())

        # full complex
        best_xyz = best_xyz.float().cpu()
        symmRs = symmRs.cpu()
        best_xyzfull = torch.zeros( (B,O*Lasu,27,3) )
        best_xyzfull[:,:Lasu] = best_xyz[:,:Lasu]
        seq_full = torch.zeros( (B,O*Lasu),dtype=seq.dtype )
        seq_full[:,:Lasu] = seq[:,:Lasu]
        best_lddtfull = torch.zeros( (B,O*Lasu) )
        best_lddtfull[:,:Lasu] = best_lddt[:,:Lasu]
        for i in range(1,O):
            best_xyzfull[:,(i*Lasu):((i+1)*Lasu)] = torch.einsum('ij,braj->brai', symmRs[i], best_xyz[:,:Lasu])
            seq_full[:,(i*Lasu):((i+1)*Lasu)] = seq[:,:Lasu]
            best_lddtfull[:,(i*Lasu):((i+1)*Lasu)] = best_lddt[:,:Lasu]

        outdata = {}

        # RMS
        outdata['mean_plddt'] = best_lddt.mean().item()
        Lstarti = 0
        for i,li in enumerate(L_s):
            Lstartj = 0
            for j,lj in enumerate(L_s):
                if (j>i):
                  outdata['pae_chain_'+str(i)+'_'+str(j)] = 0.5 * (
                      best_pae[:, Lstarti:(Lstarti+li), Lstartj:(Lstartj+lj)].mean() 
                      + best_pae[:, Lstartj:(Lstartj+lj),Lstarti:(Lstarti+li)].mean()
                  ).item()
                Lstartj += lj
            Lstarti += li

        util.writepdb("%s_pred.pdb"%(out_prefix), best_xyzfull[0], seq_full[0], L_s, bfacts=100*best_lddtfull[0])

        prob_s = [prob.permute(0,2,3,1).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        np.savez_compressed("%s.npz"%(out_prefix),
            dist=prob_s[0].astype(np.float16),
            lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16),
            pae=best_pae[0].detach().cpu().numpy().astype(np.float16))



if __name__ == "__main__":
    args = get_args()

    if (args.db is not None):
        FFDB = args.db
        FFindexDB = namedtuple("FFindexDB", "index, data")
        ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                         read_data(FFDB+'_pdb.ffdata'))
    else:
        ffdb = None

    if (torch.cuda.is_available()):
        print ("Running on GPU")
        pred = Predictor(args.model, torch.device("cuda:0"))
    else:
        print ("Running on CPU")
        pred = Predictor(args.model, torch.device("cpu"))

    pred.predict(
        inputs=args.inputs, 
        out_prefix=args.prefix, 
        symm=args.symm, 
        n_recycles=args.n_recycles, 
        n_models=args.n_models, 
        subcrop=args.subcrop, 
        topk=args.topk, 
        low_vram=args.low_vram, 
        nseqs=args.nseqs, 
        nseqs_full=args.nseqs_full, 
        cyclize=args.cyclize,
        ffdb=ffdb)

