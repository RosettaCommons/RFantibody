import numpy as np
import torch
import torch.nn as nn
from chemical import aa2num
from kinematics import get_dih
from scoring import HbHybType
from torch import einsum
from util import rigid_from_3_points

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss

def calc_rmsd(pred, true, mask):
    # pred (N,B,Lasu,natom,3)
    # true (B,Lasu,natom,3)
    # mask (B,Lasu,natom)
    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V-W)*(V-W), dim=(1,2)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    N, B, L, Natm = pred.shape[:4]
    resmask = mask[0,:,1]
    pred = pred[:,:,resmask,1].squeeze(1)
    true = true[:,resmask,1]
    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = torch.einsum('bji,njk->bik', pred, true)
    V, S, W = torch.svd(C.float())
    d = torch.ones([N,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    rpred = torch.matmul(pred, U) # (IB, L*3, 3)
    rms = rmsd(rpred, true).reshape(N)
    return rms, U, cP, cT


def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-5):
    loss_s = list()
    for i in range(len(logit_s)):
        loss = nn.CrossEntropyLoss(reduction='none')(logit_s[i], label_s[...,i]) # (B, L, L)
        loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    loss_s = torch.stack(loss_s)
    return loss_s

# use improved coordinate frame generation
def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), non_ideal=non_ideal, eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts[:,:,None] - Ts[:,:,:,None] # t[0,1] = residue 0 -> residue 1 vector
    return einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3)

def calc_str_loss(pred, true, logit_pae, mask_2d, same_chain, negative=False, d_clamp=10.0, d_clamp_inter=30.0, A=10.0, gamma=1.0, eps=1e-6):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2], non_ideal=True)
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])
    
    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    eij_label = difference[-1].clone().detach()
    
    if d_clamp != None:
        clamp = torch.where(same_chain.bool(), d_clamp, d_clamp_inter)
        clamp = clamp[None]
        difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    loss = (mask[None]*loss).sum(dim=(1,2,3)) / (mask.sum()+eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    
    # calculate pae loss
    nbin = logit_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
    pae_loss = torch.nn.CrossEntropyLoss(reduction='none')(
        logit_pae, true_pae_label)

    pae_loss = (pae_loss * mask).sum() / (mask.sum() + eps)
    return tot_loss, loss.detach(), pae_loss

#resolve rotationally equivalent sidechains
def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm( xs[:,:,None,:] - xs[atm_mask,:][None,None,:,:], dim=-1)
    dists_nat = torch.linalg.norm( xsnat[:,:,None,:] - xsnat[atm_mask,:][None,None,:,:], dim=-1)
    dists_natalt = torch.linalg.norm( xsnat_alt[:,:,None,:] - xsnat_alt[atm_mask,:][None,None,:,:], dim=-1)

    drms_nat = torch.sum(torch.abs(dists_nat-dists),dim=(-1,-2))
    drms_natalt = torch.sum(torch.abs(dists_nat-dists_natalt), dim=(-1,-2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt<drms_nat

    Rsnat_symm[toflip,...] = Rsnat_all_alt[toflip,...]
    xs_symm[toflip,...] = xsnat_alt[toflip,...]

    return Rsnat_symm, xs_symm

# resolve "equivalent" natives
def resolve_equiv_natives(xs, natstack, maskstack):
    if (len(natstack.shape)==4):
        return natstack, maskstack
    if (natstack.shape[1]==1):
        return natstack[:,0,...], maskstack[:,0,...]
    dx = torch.norm( xs[:,None,:,None,1,:]-xs[:,None,None,:,1,:], dim=-1)
    dnat = torch.norm( natstack[:,:,:,None,1,:]-natstack[:,:,None,:,1,:], dim=-1)
    delta = torch.sum( torch.abs(dnat-dx), dim=(-2,-1))
    return natstack[:,torch.argmin(delta),...], maskstack[:,torch.argmin(delta),...]


# resolve symmetric predictions
# def resolve_symmetry_predictions(pred, true, mask, symmR):
#     # pred (N,B,Lasu,natom,3)
#     # true (B,L,natom,3)
#     # mask (B,L,natom)
#     # symmR (S,3,3)
#     N, B, Lasu = pred.shape[:3]
#     L = true.shape[1]
#     O = L//Lasu
#     S = symmR.shape[0]
# 
#     # U[i] rotates layer i pred to native[0]
#     r, U, cP, cT = calc_rmsd(pred.detach(), true[:,:Lasu], mask[:,:Lasu]) # no grads through RMSd
# 
#     # symmetrize the prediction
#     pcoms = (
#         torch.sum(pred[:,0,:,1].detach()*mask[0,:Lasu,1,None], dim=-2) 
#         / torch.sum(mask[0,:Lasu,1], dim=-1)
#     )
#     Spcoms = torch.einsum('sjk,lk->lsj',symmR,pcoms)
#     Spcoms = torch.einsum('lij,lsj->lsi',U,Spcoms-cP)+cT
# 
#     # get com for each subunit in true
#     ncoms = (
#         torch.sum(
#             true[0,:,1].view(-1,Lasu,3).detach()
#             *mask[0,:,1,None].view(-1,Lasu,1)
#             , dim=-2)
#         / torch.sum(mask[0,:,1].view(-1,Lasu,1), dim=-2)
#     )
# 
#     ds = torch.linalg.norm(Spcoms[:,None] - ncoms[None,:,None], dim=-1) # (N,O,S)
# 
#     # find correspondance O->S
#     # note: this does not ensure all map to unique targets...
#     #di = torch.argmin(ds, dim=2)
#     # slower version but ensures 1->1 mapping
#     mapping_O_to_S = torch.full((N,O),-1, device=ds.device)
#     for i in range(2):
#         minj, dj = torch.min(ds, dim=2)
#         di = torch.argmin(minj, dim=1)
#         dj = dj.gather(1,di[:,None]).squeeze(1)
#         mapping_O_to_S.scatter_(1,di[:,None],dj[:,None])
#         ds.scatter_(1,di[:,None,None].repeat(1,1,S),torch.full((N,1,S),9999.0, device=ds.device))
#         ds.scatter_(2,dj[:,None,None].repeat(1,O,1),torch.full((N,O,1),9999.0, device=ds.device))
# 
#     # reindex
#     allsymmRs = symmR[mapping_O_to_S]
#     predsout = torch.einsum('noij,nbraj->nborai', allsymmRs, pred).reshape(N,B,O*Lasu,27,3)
#     return predsout

def resolve_symmetry_predictions(pred, true, mask, Lasu):
    # pred (N,B,Lpred,natom,3)
    # true (B,Ltrue,natom,3)
    # mask (B,Ltrue,natom)
    # symmR (S,3,3)
    
    N,B,Lpred = pred.shape[:3]
    Ltrue = true.shape[1]
    Opred = Lpred//Lasu
    Otrue = Ltrue//Lasu
    if (Opred < Otrue):
        print (Opred,Otrue,Lpred,Ltrue,Lasu)

    # U[i] rotates layer i pred to native[0]
    r, U, cP, cT = calc_rmsd(pred[:,:,:Lasu].detach(), true[:,:Lasu], mask[:,:Lasu]) # no grads through RMSd

    # get com for each subunit in pred, align to true
    pcoms = (
        torch.sum(
            pred[:,:,:,1].view(N,Opred,Lasu,3).detach()
            *mask[0,:Lasu,1].view(1,1,Lasu,1)
            , dim=-2)
        / torch.sum(mask[0,:Lasu,1].view(1,1,Lasu,1), dim=-2)
    )
    pcoms = torch.einsum('lij,lsj->lsi',U,pcoms-cP)+cT

    # get com for each subunit in true
    ncoms = (
        torch.sum(
            true[0,:,1].view(-1,Lasu,3).detach()
            *mask[0,:,1,None].view(-1,Lasu,1)
            , dim=-2)
        / torch.sum(mask[0,:,1].view(-1,Lasu,1), dim=-2)
    )
    ds = torch.linalg.norm(pcoms[:,None] - ncoms[None,:,None], dim=-1) # (N,Otrue,Opred)

    # find correspondance P->T
    mapping_T_to_P = torch.full((N,Otrue),-1, device=ds.device)
    for i in range(Otrue):
        minj, dj = torch.min(ds, dim=2)
        di = torch.argmin(minj, dim=1)
        dj = dj.gather(1,di[:,None]).squeeze(1)
        mapping_T_to_P.scatter_(1,di[:,None],dj[:,None])
        ds.scatter_(1,di[:,None,None].repeat(1,1,Opred),torch.full((N,1,Opred),9999.0, device=ds.device))
        ds.scatter_(2,dj[:,None,None].repeat(1,Otrue,1),torch.full((N,Otrue,1),9999.0, device=ds.device))

    # convert subunit indices to residue indices
    mapping_T_to_P = torch.repeat_interleave(mapping_T_to_P,Lasu,dim=1)
    mapping_T_to_P = mapping_T_to_P*Lasu + torch.arange(Lasu, device=ds.device).repeat(1,Otrue)
    return mapping_T_to_P

#torsion angle predictor loss
def torsionAngleLoss( alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8 ):
    I = alpha.shape[0]
    lnat = torch.sqrt( torch.sum( torch.square(alpha), dim=-1 ) + eps )
    anorm = alpha / (lnat[...,None])

    l_tors_ij = torch.min(
            torch.sum(torch.square( anorm - alphanat[None] ),dim=-1),
            torch.sum(torch.square( anorm - alphanat_alt[None] ),dim=-1)
        )

    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar

def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum('rji,rsj->rsi', Rs, xs[None,...] - Ts[:,None,...])
    xij_t = torch.einsum('rji,rsj->rsi', Rsnat, xsnat[None,...] - Tsnat[:,None,...])

    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t), dim=-1 ) + eps )
    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss

def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

def calc_BB_bond_geom(pred, idx, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, sig_len=0.02, sig_ang=0.05):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    B, L = pred.shape[:2]

    bonded = (idx[:,1:] - idx[:,:-1])==1

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = bonded*torch.clamp( torch.square(blen_CN_pred - ideal_NC) - sig_len**2, min=0.0 )
    n_viol = (CN_loss > 0.0).sum()
    blen_loss = CN_loss.sum() / (n_viol + eps)

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = bonded*torch.clamp( torch.square(bang_CACN_pred - ideal_CACN) - sig_ang**2,  min=0.0 )
    CNCA_loss = bonded*torch.clamp( torch.square(bang_CNCA_pred - ideal_CNCA) - sig_ang**2,  min=0.0 )
    bang_loss = CACN_loss + CNCA_loss
    n_viol = (bang_loss > 0.0).sum()
    bang_loss = bang_loss.sum() / (n_viol+eps)

    return blen_loss, bang_loss

# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, same_chain, ljparams, ljcorr, num_bonds, use_H=False, negative=False,
    lj_lin=0.75, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8, normalize=True, reswise=False, atom_mask=None,
):
    def ljV(dist, sigma, epsilon, lj_lin, lj_maxrad):
        linpart = dist<lj_lin*sigma
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])
        if (lj_maxrad>0):
            sdmax = sigma / lj_maxrad
            sd2 = sd*sd
            sd6 = sd2 * sd2 * sd2
            sd12 = sd6 * sd6
            ljE = ljE - epsilon * (sd12 - 2 * sd6)
        return ljE

    L = xs.shape[0]

    # mask keeps running total of what to compute
    if atom_mask != None:
        mask = atom_mask[...,None,None]*atom_mask[None,None,...]
    else:
        aamask = aamask[seq]
        if not use_H:
            aamask[...,14:] = False
        mask = aamask[...,None,None]*aamask[None,None,...]
    
    # ignore CYS-CYS (disulfide bonds)
    is_CYS = (seq == aa2num['CYS']) #(L)
    is_CYS_pair = is_CYS[:,None]*is_CYS[None,:]
    is_CYS_pair = is_CYS_pair.view(L,1,L,1)
    mask *= ~is_CYS_pair

    if negative:
        # ignore inter-chains
        mask *= same_chain.bool()[:,None,:,None]

    idxes1r = torch.tril_indices(L,L,-1)
    mask[idxes1r[0],:,idxes1r[1],:] = False
    idxes2r = torch.arange(L)
    idxes2a = torch.tril_indices(27,27,0)
    mask[idxes2r[:,None],idxes2a[0:1],idxes2r[:,None],idxes2a[1:2]] = False

    # "countpair" can be enforced by making this a weight
    mask[idxes2r,:,idxes2r,:] *= num_bonds[seq,:,:] > 3 #intra-res
    mask[idxes2r[:-1],:,idxes2r[1:],:] *= (
        num_bonds[seq[:-1],:,2:3] + num_bonds[seq[1:],0:1,:] + 1 > 3 #inter-res
    )
    si,ai,sj,aj = mask.nonzero(as_tuple=True)
    ds = torch.sqrt( torch.sum ( torch.square( xs[si,ai]-xs[sj,aj] ), dim=-1 ) + eps )

    # hbond correction
    use_hb_dis = (
        ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,1] 
        + ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0] )
    use_ohdon_dis = ( # OH are both donors & acceptors
        ljcorr[seq[si],ai,0]*ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,0] 
        +ljcorr[seq[si],ai,0]*ljcorr[seq[sj],aj,0]*ljcorr[seq[sj],aj,1] 
    )

    ljrs = ljparams[seq[si],ai,0] + ljparams[seq[sj],aj,0]
    ljrs[use_hb_dis] = lj_hb_dis
    ljrs[use_ohdon_dis] = lj_OHdon_dis
    
    if use_H:
        use_hb_hdis = (
            ljcorr[seq[si],ai,2]*ljcorr[seq[sj],aj,1] 
            +ljcorr[seq[si],ai,1]*ljcorr[seq[sj],aj,2] 
        )
        ljrs[use_hb_hdis] = lj_hbond_hdis
    
    # disulfide correction
    potential_disulf = ljcorr[seq[si],ai,3]*ljcorr[seq[sj],aj,3] 

    ljss = torch.sqrt( ljparams[seq[si],ai,1] * ljparams[seq[sj],aj,1] + eps )
    ljss [potential_disulf] = 0.0

    ljval = ljV(ds,ljrs,ljss,lj_lin,lj_maxrad)
    
    if reswise:
        ljval_res = torch.zeros_like(mask.float())
        ljval_res[si,ai,sj,aj] = ljval
        #ljval_res[:,:4,:,:4] = 0.0 # ignore clashes btw backbones?
        ljval_res = ljval_res.sum(dim=(1,3))
        ljval_res = ljval_res + ljval_res.permute(1,0)
        return ljval_res.sum(dim=-1)

    if (normalize):
        return (torch.sum( ljval )/torch.sum(aamask[seq]))
    else:
        return torch.sum( ljval )

def calc_hb(
    seq, xs, aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8, normalize=True
):
    def evalpoly( ds, xrange, yrange, coeffs ):
        v = coeffs[...,0]
        for i in range(1,10):
            v = v * ds + coeffs[...,i]
        minmask = ds<xrange[...,0]
        v[minmask] = yrange[minmask][...,0]
        maxmask = ds>xrange[...,1]
        v[maxmask] = yrange[maxmask][...,1]
        return v
    
    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh,ah = (hbts[...,0]>=0).nonzero(as_tuple=True)
    ra,aa = (hbts[...,1]>=0).nonzero(as_tuple=True)
    D_xs = xs[rh,hbba[rh,ah,0]][:,None,:]
    H_xs = xs[rh,ah][:,None,:]
    A_xs = xs[ra,aa][None,:,:]
    B_xs = xs[ra,hbba[ra,aa,0]][None,:,:]
    B0_xs = xs[ra,hbba[ra,aa,1]][None,:,:]
    hyb = hbts[ra,aa,2]
    polys = hbpolys[hbts[rh,ah,0][:,None],hbts[ra,aa,1][None,:]]

    AH = torch.sqrt( torch.sum( torch.square( H_xs-A_xs), axis=-1) + eps )
    AHD = torch.acos( cosangle( B_xs, A_xs, H_xs) )
    
    Es = polys[...,0,0]*evalpoly(
        AH,polys[...,0,1:3],polys[...,0,3:5],polys[...,0,5:])
    Es += polys[...,1,0] * evalpoly(
        AHD,polys[...,1,1:3],polys[...,1,3:5],polys[...,1,5:])

    Bm = 0.5*(B0_xs[:,hyb==HbHybType.RING]+B_xs[:,hyb==HbHybType.RING])
    cosBAH = cosangle( Bm, A_xs[:,hyb==HbHybType.RING], H_xs )
    Es[:,hyb==HbHybType.RING] += polys[:,hyb==HbHybType.RING,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.RING,2,1:3], 
        polys[:,hyb==HbHybType.RING,2,3:5], 
        polys[:,hyb==HbHybType.RING,2,5:])

    cosBAH1 = cosangle( B_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    cosBAH2 = cosangle( B0_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    Esp3_1 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH1, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Esp3_2 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH2, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Es[:,hyb==HbHybType.SP3] += torch.log(
        torch.exp(Esp3_1 * hb_sp3_softmax_fade)
        + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
    ) / hb_sp3_softmax_fade

    cosBAH = cosangle( B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs )
    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.SP2,2,1:3], 
        polys[:,hyb==HbHybType.SP2,2,3:5], 
        polys[:,hyb==HbHybType.SP2,2,5:])

    BAH = torch.acos( cosBAH )
    B0BAH = get_dih(B0_xs[:,hyb==HbHybType.SP2], B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs)

    d,m,l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like( B0BAH, m-0.5 )

    mask1 = BAH>np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH>np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * Echi

    tosquish = torch.logical_and(Es > -0.1,Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.
    if (normalize):
        return (torch.sum( Es ) / torch.sum(aamask[seq]))
    else:
        return torch.sum( Es )

def calc_lddt(pred_ca, true_ca, mask_crds, mask_2d, same_chain, negative=False, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]
    
    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    delta = torch.abs(pred_dist-true_dist) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    
    true_lddt = mask_crds[None]*true_lddt
    true_lddt = true_lddt.sum(dim=(1,2)) / (mask_crds.sum() + eps)
    return true_lddt

#fd allatom lddt
def calc_allatom_lddt_w_loss(P, Q, atm_mask, pred_lddt, idx, same_chain, negative=False, eps=1e-8):
    # Inputs
    #  - P: predicted coordinates (L, 14, 3)
    #  - Q: ground truth coordinates (L, 14, 3)
    #  - atm_mask: valid atoms (L, 14)
    #  - idx: residue index (L)

    # distance matrix
    Pij = torch.square(P[:,None,:,None,:]-P[None,:,None,:,:]) # (L, L, 14, 14)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[:,None,:,None,:]-Q[None,:,None,:,:]) # (L, L, 14, 14)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,None,:,None] * atm_mask[None,:,None,:]).float()
    # ignore atoms within same residue
    pair_mask *= (idx[:,None,None,None] != idx[None,:,None,None]).float() # (L, L, 14, 14)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:,:,None,None]

    delta_PQ = torch.abs(Pij-Qij) # (L, L, 14, 14)

    true_lddt = torch.zeros( P.shape[:2], device=P.device ) # (L, 14)
    for distbin in (0.5,1.0,2.0,4.0):
        true_lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(1,3)
            ) / ( torch.sum( pair_mask, dim=(1,3) ) + 1e-8)
    true_lddt = true_lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    
    res_mask = atm_mask.any(dim=-1) # L
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    true_lddt_label = torch.bucketize(true_lddt[None], lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction='none')(pred_lddt,
                                                      true_lddt_label)
    lddt_loss = (lddt_loss * res_mask[None]).sum() / (res_mask.sum() + eps)
    

    true_lddt = (res_mask*true_lddt).sum() / (res_mask.sum() + 1e-8)

    return lddt_loss, true_lddt

def calc_allatom_lddt(P, Q, atm_mask, idx, same_chain, negative=False, eps=1e-8):
    # Inputs
    #  - P: predicted coordinates (L, 14, 3)
    #  - Q: ground truth coordinates (L, 14, 3)
    #  - atm_mask: valid atoms (L, 14)
    #  - idx: residue index (L)

    # distance matrix
    Pij = torch.square(P[:,None,:,None,:]-P[None,:,None,:,:]) # (L, L, 14, 14)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[:,None,:,None,:]-Q[None,:,None,:,:]) # (L, L, 14, 14)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,None,:,None] * atm_mask[None,:,None,:]).float()
    # ignore atoms within same residue
    pair_mask *= (idx[:,None,None,None] != idx[None,:,None,None]).float() # (L, L, 14, 14)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:,:,None,None]

    delta_PQ = torch.abs(Pij-Qij) # (L, L, 14, 14)

    true_lddt = torch.zeros( P.shape[:2], device=P.device ) # (L, 14)
    for distbin in (0.5,1.0,2.0,4.0):
        true_lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(1,3)
            ) / ( torch.sum( pair_mask, dim=(1,3) ) + 1e-8)
    true_lddt = true_lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    
    res_mask = atm_mask.any(dim=-1) # L
    
    true_lddt = (res_mask*true_lddt).sum() / (res_mask.sum() + 1e-8)

    return true_lddt
