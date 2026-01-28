import torch
from icecream import ic

from rfantibody.rfdiffusion.chemical import INIT_CRDS
from rfantibody.rfdiffusion.kinematics import xyz_to_t2d
from rfantibody.rfdiffusion.parsers import *
from rfantibody.rfdiffusion.util import reference_angles as REF_ANGLES
from rfantibody.rfdiffusion.util import torsion_can_flip as TOR_CAN_FLIP
from rfantibody.rfdiffusion.util import torsion_indices as TOR_INDICES

ab_tasks = [
    'ab_diff',
    'ab_diff_fixed_dock'
]

schemes = {
    'single_T',
    'single_T_better_confidence',
    'no_T',
    'fixed_dock',
    'noSeq_onlyT',
    'single_T_correct_selfcond',
    'noSeq_single_T_correct_selfcond',
    'selfcond_emb'
} 

selfcond_emb_schemes = {
    'selfcond_emb'
}


def process_init_selfcond(
                    t2d: torch.Tensor(),
                    xyz_t: torch.Tensor(),
                    ab_param: dict(),
                    device
                    ):
    '''
        Given the t2d feature and xyz_t, generate the initial self conditioning features. These are the features used
        by the non-sc step of model training. At this point in time, we just zero these out but this can be made
        more sophisticated in the future.

    '''

    if ab_param['T_scheme'] in selfcond_emb_schemes:
        sc2d   = torch.zeros_like(t2d).to(device) # [B,T,L,L,44]
        xyz_sc = torch.zeros_like(xyz_t).to(device) # [B,T,L,27,3]
    else:
        sc2d   = None
        xyz_sc = None

    return sc2d, xyz_sc


def process_selfcond(
                     xyz: torch.Tensor(),
                     t2d: torch.Tensor(),
                     xyz_t: torch.Tensor(),
                     ab_param: dict(),
                     device
                    ):
    '''
        Given the network's prediction of xyz. Encode this into features for the next
        pass through the model.

        Arguments:
            
            xyz (torch.tensor)   : [B,L,3,3] The model's prediction of x0

            t2d (torch.tensor)   : [B,T,L,L,44] The t2d feature provided by the dataloader

            xyz_t (torch.tensor) : [B,T,L,27,3] The xyz_t feature provided by the dataloader

            ab_param (dict)      : Parameters describing how to generate template features

            device (str)         : The device to which we will send our tensors
        
        Returns:
           
           xyz_t (torch.tensor)  : [B,T,L,27,3]

           t2d (torch.tensor)    : [B,T,L,L,44]

    '''

    L = xyz.shape[1]

    if ab_param['T_scheme'] in selfcond_emb_schemes:
        # Just pass t2d and xyz_t through unchanged

        zeros  = torch.zeros(1,1,L,24,3).float().to(xyz.device)
        xyz_sc = torch.cat((xyz.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

        sc2d   = xyz_to_t2d(xyz_sc) # [B,T,L,L,44]

    elif ab_param['T_scheme'] in schemes:
        zeros = torch.zeros(1,1,L,24,3).float().to(device) # Make a bunch of zeros for sidechain padding
        xyz_t = torch.cat((xyz.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

        t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]

        if ab_param['T_scheme'] in {'single_T_correct_selfcond', 'noSeq_single_T_correct_selfcond'}:
            zeros = torch.zeros_like(t2d[...,:1]) # [B,T,L,L,1]
            t2d   = torch.cat((t2d, zeros), dim=-1) # [B,T,L,L,1]

        sc2d   = None
        xyz_sc = None

    else:
        raise NotImplementedError(f'The antibody templating scheme {ab_param["T_scheme"]} is not yet implemented')

    return xyz_t, t2d, xyz_sc, sc2d


def correct_selfcond(
                     t2d: torch.Tensor(),
                     ab_param: dict(),
                     xyz_true: torch.Tensor(),
                     loop_mask: torch.Tensor(),
                     target_mask: torch.Tensor(),
                     interchain_mask: torch.Tensor()
                    ) -> torch.Tensor:
    '''
        Given t2d calculated from the model's previous prediction, correct the target and framework Ab
        regions. This is intended to prevent drift in the structure of the target and Ab framework

        Arguments:
            
            t2d (torch.tensor)  : [B,N,L,L,44] The t2d feature calculated from the model's previous prediction

            ab_item (dict)      : The dictionary of featrues describing the Ab example we are working with
        
        Returns:
           
            t2d (torch.tensor)  : [B,N,L,L,44] The corrected t2d feature, ready to feed into the model

    '''

    corrected_schemes = {
                'single_T_correct_selfcond',
                'noSeq_single_T_correct_selfcond'
              }

    if ab_param['T_scheme'] not in corrected_schemes:
        print('Not correcting templated regions in the self conditioning t2d feature')
        return t2d

    print('Correcting constant t2d entries')

    t2d_true = xyz_to_t2d(xyz_true[None,None]).to(t2d.device) # [B,T,L,L,44]

    zeros = torch.zeros_like(t2d_true[...,:1]).to(t2d.device) # [B,T,L,L,1]
    t2d_true = torch.cat((t2d_true,zeros), dim=-1) # [B,T,L,L,45]; Added selfcond one-hot dimension

    target_mask2d = target_mask[None] * target_mask[:,None]

    design_mask   = ~target_mask * ~loop_mask # This is a mask of the constant binder region

    # We are leaving out the interchain contacts
    binder_mask2d = design_mask[None] * design_mask[:,None] * ~interchain_mask

    t2d[...,44] = 1 # All entries in t2d originally come from self conditioning so we will mark them as true
    t2d[...,43] = 0 # Mark all entries initially as non-templates - this will be updated below

    # Correct the target contact entries
    t2d[:,:,target_mask2d] = t2d_true[:,:,target_mask2d]

    # Correct the binder framework contact entries
    t2d[:,:,binder_mask2d] = t2d_true[:,:,binder_mask2d]

    return t2d


def apply_templating_scheme(item,
                            xyz_t,
                            t1d,
                            t2d,
                            T_scheme,
                            hotspot_dim,
                            timestep):
    '''
    Take the template features which have been featurized normally and apply the selected
    templating scheme to them

    Arguments:

        xyz_t (torch.tensor)       :

        t1d (torch.tensor)         :

        t2d (torch.tensor)         :

        xyz_true (torch.tensor)    :

        target_mask (torch.tensor) :

        loop_mask (torch.tensor)   :

        hotspot_dim (int)          :

        timestep (torch.tensor)    :

    Returns:

        xyz_t (torch.tensor) :

        t1d (torch.tensor)   :

        t2d (torch.tensor)   :

    '''

    ret_xyz_t = torch.clone(xyz_t)

    # For all templating shemes, we will need the true t2d
    ret_t2d = xyz_to_t2d(item.inputs.xyz_true[None,None]).squeeze(0) # [T,L,L,44]

    # Annotate which residues in t1d are missing/masked
    ret_t1d = torch.clone(t1d)
    ret_t1d[:,~item.loop_mask,20] = 0 
    ret_t1d[:,item.loop_mask,20]  = 1 

    ic(f'Featurizing with {T_scheme}')

    if item.inputs.fixed_dock or T_scheme == 'fixed_dock':
        ######################################################
        ### Fixed Dock Schemes
        ######################################################

        ## t1d ##
        #########
        # Same as in nonfixed dock case
        ret_t1d[:,~item.loop_mask,21] = 1
        ret_t1d[:,item.loop_mask,21]  = timestep # 22nd entry is confidence

        # Nate thinks it is not necessary to do both fixed dock and hotspot
        # features. 
        #if item.target:
        #    # Add in hotspot labels
        #    ret_t1d[:,:,22] = item.hotspots # 23nd entry is hotspots

        ## t2d ##
        #########
        ret_t2d[:,item.loop_mask] = 0
        ret_t2d[:,:,item.loop_mask] = 0

    else:
        ######################################################
        ### Mobile (non-fixed) Dock Schemes
        ######################################################
        if T_scheme == 'selfcond_emb':
            # In this scheme, we will use a different track for the self-conditioning embedding
            # The template track is used for just the template features

            # Here we are JUST providing the template for the H and L chains, not the target
            # Target information is passed to the model through xyz_prev, this is similar to
            # how PPI RFdiffusion works

            ## t1d ##
            #########
            # Add in appropriate confidence metrics
            if item.target:
                ret_t1d[:,item.target_mask,21]  = 1
                ret_t1d[:,~item.target_mask,21] = timestep # 22nd entry is confidence
            else:
                # In the monomer case, we have no target so all confidence is timestep
                ret_t1d[:,:,21] = timestep

            if item.target:
                # Add in hotspot labels
                # Subtract 1 from hotspot dim since hotspot dim is 1 indexed
                ret_t1d[:,:,hotspot_dim-1]  = item.hotspots

            ## t2d ##
            #########

            L = item.loop_mask.shape[0]

            # Mask out the target contacts from the t2d
            if item.target:
                ret_t2d[:,item.target_mask]   = 0
                ret_t2d[:,:,item.target_mask] = 0

            # Mask out the loops from the t2d
            ret_t2d[:,item.loop_mask]   = 0
            ret_t2d[:,:,item.loop_mask] = 0

            # This is a mask that only selects the intrachain ab contacts
            ab_intrachain_mask = torch.zeros(L,L).bool() # [L,L]

            offset = 0
            if item.target and item.target_mask[0]:
                offset += item.T['xyz'].shape[0]

            # We know the order of the chains in the pdb is T,H,L
            if item.H is not None and item.L is not None:
                # There exists an ab interchain region to mask
                H_mask = torch.zeros(L,L).bool()
                L_mask = torch.zeros(L,L).bool()

                H_end_idx = offset+item.H['xyz'].shape[0]
                H_mask[offset:H_end_idx, offset:H_end_idx] = 1
                L_mask[H_end_idx:, H_end_idx:] = 1

                ab_intrachain_mask += H_mask
                ab_intrachain_mask += L_mask

                # Mask H-L contacts
                ret_t2d *= ab_intrachain_mask[None,:,:,None]

            # We will also just make ret_xyz_t be all zeros
            ret_xyz_t = torch.zeros_like(ret_xyz_t)
            

        elif T_scheme == 'single_T':
            # We are going to mask the interchain blocks in t2d, unclear how to handle xyz_t here
            # Then handle t1d by adding the correct amount of template dimensions

            ## t1d ##
            #########
            # Add in appropriate confidence metrics
            ret_t1d[:,~item.loop_mask,21] = 1
            ret_t1d[:,item.loop_mask,21]  = timestep # 22nd entry is confidence

            if item.target:
                # Add in hotspot labels
                # Subtract 1 from hotspot dim since hotspot dim is 1 indexed
                ret_t1d[:,:,hotspot_dim-1]           = item.hotspots

            ## t2d ##
            #########
            if item.target:
                # Mask the interchain contacts
                upper_right_diag = ~item.target_mask[None] * item.target_mask[:,None]
                lower_left_diag  = item.target_mask[None] * ~item.target_mask[:,None]

                # Simply writing to slices was not working for me, doing this more
                # convoluted method to write to the diagonals - NRB

                ret_t2d[:,upper_right_diag] = 0
                ret_t2d[:,lower_left_diag]  = 0

                ret_t2d[:,item.loop_mask] = 0
                ret_t2d[:,:,item.loop_mask]  = 0
            else:
                # In the monomer case, just mask the loops
                ret_t2d[:,item.loop_mask] = 0
                ret_t2d[:,:,item.loop_mask] = 0

        elif T_scheme in {'single_T_better_confidence', 'noSeq_onlyT', 'single_T_correct_selfcond', 'noSeq_single_T_correct_selfcond'}:
            # We are going to mask the interchain blocks in t2d, unclear how to handle xyz_t here
            # Then handle t1d by adding the correct amount of template dimensions

            if item.target:
                ## t1d ##
                #########
                # Here we are going to mark all ab residues as diffused
                ret_t1d[:,item.target_mask,21]  = 1
                ret_t1d[:,~item.target_mask,21] = timestep # 22nd entry is confidence

                # Add in hotspot labels
                # Subtract 1 from hotspot dim since hotspot dim is 1 indexed
                ret_t1d[:,:,hotspot_dim-1] = item.hotspots

                ## t2d ##
                #########
                # Mask the interchain contacts
                upper_right_diag = ~item.target_mask[None] * item.target_mask[:,None]
                lower_left_diag  = item.target_mask[None] * ~item.target_mask[:,None]

                # Simply writing to slices was not working for me, doing this more
                # convoluted method to write to the diagonals - NRB

                ret_t2d[:,upper_right_diag] = 0
                ret_t2d[:,lower_left_diag]  = 0

                ret_t2d[:,item.loop_mask] = 0
                ret_t2d[:,:,item.loop_mask]  = 0

            else:
                ## t1d ##
                #########
                ret_t1d[:,~item.loop_mask,21] = 1
                ret_t1d[:,item.loop_mask,21]  = timestep # 22nd entry is confidence

                ## t2d ##
                #########
                # In the monomer case, just mask the loops
                ret_t2d[:,item.loop_mask] = 0
                ret_t2d[:,:,item.loop_mask] = 0


            if T_scheme in {'single_T_correct_selfcond', 'noSeq_single_T_correct_selfcond'}:
                # Expand t2d feature to allow for a feature which indicates which positions are true
                # and which are predicted

                L = item.loop_mask.shape[0]

                # Order in pdb is always T,H,L. Can use this to mask Ab-interchain contacts
                item.ab_interchain_mask = torch.zeros(L,L).bool() # [L,L]

                offset = 0
                if item.target and item.target_mask[0]:
                    offset += item.T['xyz'].shape[0]
                if item.H is not None and item.L is not None:
                   # There exists an ab interchain region to mask
                   H_mask = torch.zeros(L,L).bool()
                   L_mask = torch.zeros(L,L).bool()

                   H_end_idx = offset+item.H['xyz'].shape[0]
                   H_mask[offset:H_end_idx, offset:H_end_idx] = 1
                   L_mask[H_end_idx:, H_end_idx:] = 1

                   if item.target:
                       upper_interchain = ~item.target_mask *  H_mask * ~L_mask
                       lower_interchain = ~item.target_mask * ~H_mask *  L_mask
                   else:
                       upper_interchain =  H_mask * ~L_mask
                       lower_interchain = ~H_mask *  L_mask

                   item.ab_interchain_mask[upper_interchain] = True
                   item.ab_interchain_mask[lower_interchain] = True

                   # Mask H-L contacts
                   ret_t2d[:,item.ab_interchain_mask] = 0 

                zeros   = torch.zeros_like(ret_t2d[...,:1]) # [T,L,L,1]
                ret_t2d = torch.cat((ret_t2d, zeros), dim=-1) # [T,L,L,45]


        elif T_scheme == 'no_T':
            # In this templating scheme we are not going to provide any templating information
            # Only the sequence of the framework will be provided

            if item.target:
                ## t1d ##
                #########
                # Here we are going to mark all ab residues as diffused
                ret_t1d[:,item.target_mask,21]  = 1
                ret_t1d[:,~item.target_mask,21] = timestep # 22nd entry is confidence

                # Add in hotspot labels
                # Subtract 1 from hotspot dim since hotspot dim is 1 indexed
                ret_t1d[:,:,hotspot_dim-1] = item.hotspots

            else:
                ## t1d ##
                #########
                ret_t1d[:,~item.loop_mask,21] = 1
                ret_t1d[:,item.loop_mask,21]  = timestep # 22nd entry is confidence

            ## t2d ##
            #########
            # Mask all template information
            ret_t2d = torch.zeros_like(ret_t2d)

            ## xyz_t ##
            ###########
            # Mask all template coordinates
            ret_xyz_t = torch.zeros_like(xyz_t)

        elif T_scheme == 'fixed_dock':
            raise NotImplementedError(f'The combination of fixed dock T scheme and the ab item' + \
                                       'not doing fixed dock is not supported')

        else:
            raise NotImplementedError(f'The antibody templating scheme {T_scheme} is not yet implemented')

    return ret_xyz_t, ret_t1d, ret_t2d


def featurize(item,
              seq,
              xyz,
              d_t1d,
              hotspot_dim,
              T_scheme,
              timestep,
              mask_all_sc,
              bugfix_t1d_mask):
    '''
    Takes a sequence and a noised structure and returns features ready to be fed into the model

    Arguments:

        seq (torch.tensor)         : [L,22] One hot representation of sequence

        x_t (torch.tensor)         : [L,27,3] Diffused full atom protein structure

        hotspot_dim (int)          : Which dimension of t1d to put the hotspot labels

        xyz_true (torch.tensor)    : [L,27,3] True full atom protein structure

        timestep (torch.tensor)    : [1] The timestep converted into a confidence: 1- (little_t / T)

        target_mask (torch.tensor) : [L,27,3] Full atom protein structure

        loop_mask (torch.tensor)   : [L,27,3] Full atom protein structure

        hotspots  (torch.tensor)   : [L,27,3] Full atom protein structure

    Returns:

        feature_dict (dict) 

            seq (torch.tensor)        : [I,L,22]

            msa_masked (torch.tensor) : [I,N,L,48]

            msa_full (torch.tensor)   : [I,N,L,25]

            xyz_t (torch.tensor)      : [T,L,27,3] 

            t1d (torch.tensor)        : [T,L,d_t1d]

            t2d (torch.tensor)        : [T,L,L,44]

            alpha_t (torch.tensor)    : [T,L,30]

            xyz_prev (torch.tensor)   : [L,27,3]

    '''

    L = seq.shape[0]
    
    ## seq ##
    #########
    seq_feat = torch.clone(seq).unsqueeze(0) # (I,L,22)
    
    ## msa_masked ##
    ################
    msa_masked = torch.zeros((1,1,L,48))
    msa_masked[:,:,:,:22] = seq[None, None]
    msa_masked[:,:,:,22:44] = seq[None, None] # (I,N,L,48)
    
    ## msa_full ##
    ##############
    msa_full = torch.zeros((1,1,L,25))
    msa_full[:,:,:,:22] = seq[None, None] # (I,N,L,25)

    ## xyz_t ##
    ###########
    xyz_t=torch.clone(xyz.unsqueeze(0)) # (T,L,27,3)

    # Mask the diffused residues' sidechains in xyz_t
    xyz_t[:,item.loop_mask,3:] = float('nan')
    # Option to mask all sidechains in xyz_t
    if mask_all_sc:
        xyz_t[:,:,3:] = float('nan')

    ## Changed this to only replace NaNs and not center - NRB
    #xyz_t = get_init_xyz(xyz_t, item.inputs.mask_t, item) 

    ## t1d ##
    #########
    t1d = torch.zeros((1,L,d_t1d)) # (1,L,d_t1d)
    # We will mask the sequence in t1d in the apply_templating_scheme call
    t1d[:,:,:20] = seq[None,:,:20]
    if bugfix_t1d_mask:
        t1d[:,:,20]  = seq[None,:,21] # The 22nd entry in seq is mask and the 21st entry in t1d is mask

    ## t2d ##
    #########
    t2d = xyz_to_t2d(xyz_t.unsqueeze(0)) # (B,T,L,L,44), this function requires a batch dimension
    t2d = t2d.squeeze(0) # (T,L,L,44)

    ## alpha_t ##
    #############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 30) # (T,L,30)

    ## xyz_prev ##
    ##############
    xyz_prev = torch.clone(xyz_t[0]) # (L,27,3) 

    xyz_t, t1d, t2d = apply_templating_scheme(
                                               item,
                                               xyz_t,
                                               t1d,
                                               t2d,
                                               T_scheme,
                                               hotspot_dim,
                                               timestep
                                             )

    input_dict = {
                    'seq'        : seq_feat,
                    'msa_masked' : msa_masked,
                    'msa_full'   : msa_full,
                    'xyz_t'      : xyz_t,
                    't1d'        : t1d,
                    't2d'        : t2d,
                    'alpha_t'    : alpha_t,
                    'xyz_prev'   : xyz_prev
                 }

    return input_dict
