import numpy as np
import os
import sys
from omegaconf import DictConfig
from rfantibody.rfdiffusion.kinematics import xyz_to_t2d
import torch
import torch.nn.functional as nn
from rfantibody.rfdiffusion.util import get_torsions
from rfantibody.rfdiffusion.diffusion import get_beta_schedule, get_aa_schedule, get_chi_betaT
from rfantibody.rfdiffusion.diff_util import get_aa_schedule, th_interpolate_angles, th_min_angle, th_interpolate_angle_single
from icecream import ic
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp
from rfantibody.rfdiffusion.util import torsion_indices as TOR_INDICES
from rfantibody.rfdiffusion.util import torsion_can_flip as TOR_CAN_FLIP
from rfantibody.rfdiffusion.util import reference_angles as REF_ANGLES
from rfantibody.rfdiffusion.util import rigid_from_3_points
from rfantibody.rfdiffusion.util_module import ComputeAllAtomCoords
from rfantibody.rfdiffusion.potentials.manager import PotentialManager
from rfantibody.rfdiffusion import util
import random
import logging
import string 
import hydra
import glob

###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################

# These functions behave exactly the same as before but now do not rely on class fields from the Denoiser

def slerp_update(r_t, r_0, t, mask=0):
    """slerp_update uses SLERP to update the frames at time t to the
    predicted frame for t=0

    Args:
        R_t, R_0: rotation matrices of shape [3, 3]
        t: time step
        mask: set to 1 / True to skip update.

    Returns:
        slerped rotation for time t-1 of shape [3, 3]
    """
    # interpolate FRAMES between one and next 
    if not mask:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_0], axis=0))
    else:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_t], axis=0))

    key_times = [0,1]

    interpolator = Slerp(key_times, key_rots)
    alpha = np.array([1/t])
    
    # grab the interpolated FRAME 
    interp_frame  = interpolator(alpha)
    
    # constructed rotation matrix which when applied YIELDS interpolated frame 
    interp_rot = (interp_frame.as_matrix().squeeze() @ np.linalg.inv(r_t.squeeze()) )[None,...]

    return interp_rot

def slerp_update_vectorized(R_t, R_0, t, mask=0):
    """vectorized version of slerp_update. Not really much faster..."""

    Interp_frame = np.full(R_t.shape, np.nan)
    key_times = [0,1]
    alpha = np.array([1/t])

    for i in range(R_t.shape[0]):
        if ( not mask[i] ):
            key_rots = scipy_R.from_matrix(np.stack([R_t[i], R_0[i]], axis=0))
        else:
            key_rots = scipy_R.from_matrix(np.stack([R_t[i], R_t[i]], axis=0))
        interpolator = Slerp(key_times, key_rots)

        Interp_frame[i] = interpolator(alpha).as_matrix()

    Interp_rot = np.einsum('...ij,...kj->...ik', Interp_frame, R_t)

    return Interp_rot

def get_next_frames(xt, px0, t, diffuser, so3_type, diffusion_mask, noise_scale=1., rotation_scaling=None):
    """get_next_frames gets updated frames using either SLERP or the IGSO(3) + score_based reverse diffusion.
    

    based on self.so3_type use slerp or score based update.

    SLERP xt frames towards px0, by factor 1/t
    Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0
   
    Args:
        xt: noised coordinates of shape [L, 14, 3]
        px0: prediction of coordinates at t=0, of shape [L, 14, 3]
        t: integer time step
        diffuser: Diffuser object for reverse igSO3 sampling
        so3_type: The type of SO3 noising being used ('igso3', or 'slerp')
        diffusion_mask: of shape [L] of type bool, True means not to be
            updated (e.g. mask is true for motif residues)
        noise_scale: scale factor for the noise added (IGSO3 only)
    
    Returns:
        backbone coordinates for step x_t-1 of shape [L, 3, 3]
    """
    N_0  = px0[None,:,0,:]
    Ca_0 = px0[None,:,1,:]
    C_0  = px0[None,:,2,:]

    R_0, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)

    N_t  = xt[None, :, 0, :]
    Ca_t = xt[None, :, 1, :]
    C_t  = xt[None, :, 2, :]

    R_t, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)

    # this must be to normalize them or something
    R_0 = scipy_R.from_matrix(R_0.squeeze().numpy()).as_matrix()
    R_t = scipy_R.from_matrix(R_t.squeeze().numpy()).as_matrix()


    L = R_t.shape[0]
    all_rot_transitions = np.broadcast_to(np.identity(3), (L, 3, 3)).copy()
    # Sample next frame for each residue
    if so3_type == "igso3":
        # don't do calculations on masked positions since they end up as identity matrix
        all_rot_transitions[~diffusion_mask] = diffuser.so3_diffuser.reverse_sample_vectorized(R_t[~diffusion_mask], R_0[~diffusion_mask], t, 
                                                                                        noise_level=noise_scale, mask=None, return_perturb=True, rotation_scaling=rotation_scaling)
    elif so3_type == "slerp":
        vect_all_rot_transitions[~diffusion_mask] = slerp_update_vectorized(R_t[~diffusion_mask], R_0[~diffusion_mask], t, 
                                                                                                    mask=diffusion_mask[~diffusion_mask])    
    else:
        assert False, "so3 diffusion type %s not implemented"%so3_type

    all_rot_transitions = all_rot_transitions[:,None,:,:]


    assert_vectorized_is_equal = False
    if assert_vectorized_is_equal:
        # Sample next frame for each residue
        check_all_rot_transitions = []
        for i in range(len(xt)):
            r_0 = R_0[i] #.as_matrix()
            r_t = R_t[i] #.as_matrix()
            mask_i = diffusion_mask[i]

            if so3_type == "igso3":
                r_t_next = diffuser.so3_diffuser.reverse_sample(r_t, r_0, t,
                        mask=mask_i, noise_level=noise_scale)[None,...]
                interp_rot =  r_t_next @ (r_t.T)
            elif so3_type == "slerp":
                interp_rot = slerp_update(r_t, r_0, t, diffusion_mask[i])
            else:
                assert False, "so3 diffusion type %s not implemented"%so3_type

            check_all_rot_transitions.append(interp_rot)

        check_all_rot_transitions = np.stack(check_all_rot_transitions, axis=0)

        assert np.allclose(check_all_rot_transitions, all_rot_transitions)

    # Apply the interpolated rotation matrices to the coordinates
    next_crds   = np.einsum('lrij,laj->lrai', all_rot_transitions, xt[:,:3,:] - Ca_t.squeeze()[:,None,...].numpy()) + Ca_t.squeeze()[:,None,None,...].numpy()

    # (L,3,3) set of backbone coordinates with slight rotation 
    return next_crds.squeeze(1)

def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1
    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*beta_schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*beta_schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-beta_schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca

    mu = a + b

    return mu, sigma

def get_next_ca(xt, px0, t, diffusion_mask, crd_scale, beta_schedule, alphabar_schedule, noise_scale=1., rotation_scaling=None):
    """
    Given full atom x0 prediction (xyz coordinates), diffuse to x(t-1)
    
    Parameters:
        
        xt (L, 14/27, 3) set of coordinates
        
        px0 (L, 14/27, 3) set of coordinates

        t: time step. Note this is zero-index current time step, so are generating t-1    

        logits_aa (L x 20 ) amino acid probabilities at each position

        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t

        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused 

        noise_scale: scale factor for the noise being added

    """
    get_allatom = ComputeAllAtomCoords().to(device=xt.device)
    L = len(xt)

    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale 
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule)
    
    if rotation_scaling is None:
        sampled_crds = torch.normal(mu, torch.sqrt(sigma*noise_scale))
    else:
        sampled_crds = mu
    delta = sampled_crds - xt[:,1,:] #check sign of this is correct

    if not diffusion_mask is None:
        # calculate the mean displacement between the current motif and where 
        # RoseTTAFold thinks it should go 
        # print('Got motif delta')
        # motif_delta = (px0[diffusion_mask,:3,...] - xt[diffusion_mask,:3,...]).mean(0).mean(0)

        delta[diffusion_mask,...] = 0
        # delta[diffusion_mask,...] = motif_delta

    out_crds = xt + delta[:, None, :]

    return out_crds/crd_scale, delta/crd_scale

def get_noise_schedule(T, noiseT, noise1, schedule_type):
    """
    Function to create a schedule that varies the scale of noise given to the model over time

    Parameters:
        
        T: The total number of timesteps in the denoising trajectory

        noiseT: The inital (t=T) noise scale

        noise1: The final (t=1) noise scale

        schedule_type: The type of function to use to interpolate between noiseT and noise1

    Returns:
        
        noise_schedule: A function which maps timestep to noise scale

    """

    noise_schedules = {
            'constant': lambda t: noiseT,
            'linear': lambda t: ( (t-1) / (T-1) ) * (noiseT - noise1) + noise1
            }

    assert(schedule_type in noise_schedules),\
            f'noise_schedule must be one of {noise_schedules.keys()}. Received noise_schedule={schedule_type}. Exiting.'

    return noise_schedules[schedule_type]

class DecodeSchedule():
    """
    Class for managing AA decoding schedule stuff
    """

    def __init__(self, L, visible, aa_decode_steps=40, mode='distance_based'):

        # only distance based for now
        assert mode in ['distance_based']# , 'uniform_linear', 'ar_fixbb']
        self.mode = mode

        self.visible = visible

        # start as all high - only matters when a residue is being decoded
        # at which point we will know the true T
        self.T = torch.full((L,), 999)

        # number of residues being decoded on each step
        if aa_decode_steps > 0:
            tmp = np.array(list(range((~self.visible).sum())))
            np.random.shuffle(tmp)
            ndecode_per_step = np.array_split(tmp, aa_decode_steps)
            np.random.shuffle(ndecode_per_step)
            self.ndecode_per_step = [len(a) for a in ndecode_per_step]


    def get_next_idx(self, cur_indices, dmap):
        """
        Given indices being currently sampled and a distance map, return one more index which is allowed to
        be sampled at the same time as cur indices

        Parameters:

            cur_indices (list, required): indices of residues also being decoded this step

            dmap (torch.tensor, required): (L,L) distance map of CA's
        """
        L = dmap.shape[0]
        options = torch.arange(L)[~self.visible] # indices of non-decoded residues

        # find the index with the largest average distance from all decoded residues
        #mean_distances = dmap[cur_indices,options]
        d_rows    = dmap[cur_indices]
        d_columns = d_rows[:,options]
        mean_distances = d_columns.mean(dim=0)


        #mean_distances = mean_distances.mean(dim=0)


        best_idx_local  = torch.argmax(mean_distances) # index within options tensor
        best_idx_global = options[best_idx_local]      # the option itself

        return best_idx_global


    def get_decode_positions(self, t_idx, px0):
        """
        Returns the next (0-indexed) positions to decode for this timestep
        """
        L = px0.shape[0]
        assert t_idx < len( self.ndecode_per_step ) # taken care of outside this class in sampling loop

        N = self.ndecode_per_step[t_idx]
        decode_list = []

        if self.mode == 'distance_based':
            # perform dynamic distance based sampling
            ca   = px0[:,1,:]
            dmap = torch.sqrt( (ca[None,:] - ca[:,None]).square().sum(dim=-1) + 1e-6 )

            for i in range(N):
                if i == 0:
                    # sample a random residue which hasn't been decoded yet
                    first_idx = np.random.choice(torch.arange(L)[~self.visible])
                    decode_list.append(int(first_idx))
                    self.visible[first_idx] = True
                    self.T[first_idx] = t_idx + 1
                    continue

                # given already sampled indices, get another
                decode_idx = self.get_next_idx(decode_list,dmap)

                decode_list.append(int(decode_idx))
                self.visible[decode_idx] = True # set this now because get_next_idx depends on it
                self.T[decode_idx] = t_idx+1    # now that we know this residue is decoded, set its big T value

        return decode_list


    @property
    def idx2steps(self):
        return self.T

class Denoise():
    """
    Class for getting x(t-1) from predicted x0 and x(t)
    Strategy:
        Ca coordinates: Rediffuse to x(t-1) from predicted x0
        Frames: SLERP 1/t of the way to x0 prediction
        Torsions: 1/t of the way to the x0 prediction

    """
    def __init__(self,
                 T,
                 L,
                 diffuser,
                 visible,
                 seq_diffuser=None,
                 b_0=0.001,
                 b_T=0.1,
                 min_b=1.0,
                 max_b=12.5,
                 min_sigma=0.05,
                 max_sigma=1.5,     # DJ- simply put this in as dummy 
                 noise_level=0.5,
                 schedule_type='cosine',
                 so3_schedule_type='linear',
                 schedule_kwargs={},
                 so3_type='slerp',
                 chi_type='interp',
                 noise_scale_ca=1.,
                 final_noise_scale_ca=1,
                 ca_noise_schedule_type='constant',
                 noise_scale_frame=0.5,
                 final_noise_scale_frame=0.5,
                 frame_noise_schedule_type='constant',
                 noise_scale_torsion=1.,
                 crd_scale=1/15,
                 aa_decode_steps=100,
                 potential_manager=None,
                 softmax_T=1e-5,
                 partial_T=None,
                 rotation_scaling=None):
        """
        
        Parameters:
            noise_level: scaling on the noise added (set to 0 to use no noise,
                to 1 to have full noise)
            
        """
        self.T = T
        self.L = L 
        self.diffuser = diffuser
        self.seq_diffuser = seq_diffuser
        self.b_0 = b_0
        self.b_T = b_T
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.chi_type = chi_type
        self.crd_scale = crd_scale
        self.noise_scale_ca = noise_scale_ca
        self.final_noise_scale_ca = final_noise_scale_ca
        self.ca_noise_schedule_type = ca_noise_schedule_type
        self.noise_scale_frame = noise_scale_ca#noise_scale_frame
        self.final_noise_scale_frame = final_noise_scale_ca#final_noise_scale_frame
        self.frame_noise_schedule_type = ca_noise_schedule_type#frame_noise_schedule_type
        self.noise_scale_torsion = noise_scale_torsion
        self.aa_decode_steps=aa_decode_steps
        self.potential_manager = potential_manager
        self.rotation_scaling=rotation_scaling
        self._log = logging.getLogger(__name__)

        self.schedule, self.alpha_schedule, self.alphabar_schedule = get_beta_schedule(self.T, self.b_0, self.b_T, self.schedule_type, inference=True)

        self.noise_schedule_ca     = get_noise_schedule(self.T, self.noise_scale_ca, self.final_noise_scale_ca, self.ca_noise_schedule_type)
        self.noise_schedule_frame  = get_noise_schedule(self.T, self.noise_scale_frame, self.final_noise_scale_frame, self.frame_noise_schedule_type)

        # dynamic schedule dictionaries for chi angles 
        max_chi_T = 160
        chi_b_0 = 0.001
        chi_abar_T=1e-3
        assert max_chi_T >= aa_decode_steps # assume never decoding for more than 100 steps, change if otherwise
        self.chi_beta_T = get_chi_betaT(max_chi_T, chi_b_0, chi_abar_T, method='cosine') # precalculate chi beta schedules for dynamic T


        # amino acid decoding schedule 
        #out = get_aa_schedule(T,L,nsteps=aa_decode_steps)
        #self.aa_decode_times, self.decode_order, self.idx2steps, self.aa_mask_stack = out
        if seq_diffuser is None: self.decode_scheduler = DecodeSchedule(L, visible, aa_decode_steps, mode='distance_based')

    @property
    def idx2steps(self):
        return self.decode_scheduler.idx2steps.numpy()

    @staticmethod 
    def get_dynamic_mu_sigma(chi_t, chi_0, T, t, chi_beta_T, beta_0, schedule_type='cosine'):
        """
        Given currente chis, prediction of chi0, dynamic T, current t, and chi_beta_T schedules, 
        sample new angles 
        
        Need to make this faster probably.
        """
        t_idx = t-1
        assert t > 1 # t must be 2+ to do this
        assert len(chi_t) == len(chi_0)
        
        max_t = max(chi_beta_T.keys())
        betas_t = torch.full_like(chi_t, float('nan'))
        abars_t = torch.full_like(chi_t, float('nan'))
        abars_t_minus1 = torch.full_like(chi_t, float('nan')) 
        
        for i,_ in enumerate(chi_t):
            T_for_this_chi = T[i]
            
            # make sure T is in domain of chi_beta_T
            # if we choose max T, it's clearly a masked residue and doesn't matter anyway 
            cur_T = min(T_for_this_chi, max_t)

            if t <= cur_T: # it's a valid position to find a beta for 
                # get custom schedules for this amino acid based on its T 
                beta_T = chi_beta_T[int(cur_T)]

                beta_schedule, alpha_schedule, abar_schedule = get_beta_schedule(cur_T, 
                                                                                 beta_0, 
                                                                                 beta_T, 
                                                                                 schedule_type)
                
                betas_t[i] = beta_schedule[t_idx]
                abars_t[i] = abar_schedule[t_idx]
                abars_t_minus1[i] = abar_schedule[t_idx-1]

            else: # it's a position who will be masked so beta doesn't matter 
                betas_t[i] = 0
                abars_t[i] = 0
                abars_t_minus1[i] = 0


        
        # Now that we have abars and betas, create mu and sigma for all angles 
        variance = ((1 - abars_t_minus1)/(1-abars_t))*betas_t
        
        a = (torch.sqrt(abars_t_minus1)*betas_t)/(1-abars_t)*chi_0
        b = (torch.sqrt(1-betas_t)*(1-abars_t_minus1)/(1-abars_t))*chi_t
        
        mean = a+b
        
        return mean, variance

    def reveal_residues(self, seq_t, seq_px0, px0, t):
        '''
        Reveal some amino acids in the sequence according to schedule and predictions

        seq_t (torch.tensor): [L] Integer sequence

        seq_px0 (torch.tensor): [L] Model prediction of sequence

        px0 (torch.tensor): [L,14,3] Predicted set of full atom crds

        t (int): Current timestep

        '''
        next_seq = torch.clone(seq_t)

        if t <= self.aa_decode_steps:
            t_idx = t-1
            decode_positions           = self.decode_scheduler.get_decode_positions(t_idx, px0)
            replacement                = seq_px0[decode_positions]
            replacement                = replacement.to(next_seq.device)
            next_seq[decode_positions] = replacement

        return next_seq

    def get_next_torsions(self, xt, px0, seq_t, seq_logit_px0, t, diffusion_mask=None, noise_scale=1.):
        """
        Samples the next chi angles and amino acid identities for the pose 

        xt (torch.tensor, required): L,14,3 or L,27,3 set of fullatom crds 

        px0 (torch.tensor, required): Predicted set of full atom crds 

        seq_t (torch.tensor, required): Current sequence in the pose 

        pseq_0 (torch.tensor, required): predicted sequence logits for the x0 prediction

        t (int, required): timestep 

        noise_scale: scale factor for the noise added (only applies to "wrapped_normal")

        """
        L = len(xt)
        # argmaxed sequence logits for x0 prediction
        seq_px0 = torch.argmax(seq_logit_px0, dim=-1) 

        ### initialize full atom reps to calculate torsions 
        RAD = True # use radians for angle stuff 

        # build 27-atom representations
        if not xt.shape[1] == 27:
            xt_full = torch.full((L,27,3),np.nan).float()
            xt_full[:,:14,:] = xt[:,:14]


        if not px0.shape[1] == 27:
            px0_full = torch.full((L,27,3),np.nan).float()
            px0_full[:,:14,:] = px0[:,:14]



        # there is no situation where we should have any NaN BB crds here  
        mask = torch.full((L, 27), False)
        mask[:,:14] = True 

        ### Calcualte torsions and interpolate between them 

        # Xt torsions
        torsions_sincos_xt, torsions_alt_sincos_xt, tors_mask_xt, tors_planar_xt = get_torsions(xt_full[None], 
                                                                                                seq_t[None], 
                                                                                                TOR_INDICES, 
                                                                                                TOR_CAN_FLIP, 
                                                                                                REF_ANGLES)
        torsions_angle_xt     = torch.atan2(torsions_sincos_xt[...,1],
                                            torsions_sincos_xt[...,0]).squeeze()
        torsions_angle_xt_alt = torch.atan2(torsions_alt_sincos_xt[...,1],
                                            torsions_alt_sincos_xt[...,0]).squeeze()



        # pX0 torsions - uses same seq_t to calculate 
        torsions_sincos_px0, torsions_alt_sincos_px0, tors_mask_px0, tors_planar_px0 = get_torsions(px0_full[None], 
                                                                                                    seq_t[None], 
                                                                                                    TOR_INDICES, 
                                                                                                    TOR_CAN_FLIP, 
                                                                                                    REF_ANGLES)
        torsions_angle_px0     = torch.atan2(torsions_sincos_px0[...,1],
                                             torsions_sincos_px0[...,0]).squeeze()

        # Take the first one as ground truth --> don't need alt for px0 
        # torsions_angle_alt_px0 = torch.atan2(torsions_alt_sincos_px0[...,1],
        #                                      torsions_alt_sincos_px0[...,0]).squeeze()


        # calculate the minimum angle between both options in xt and the angles in px0
        a = th_min_angle(torsions_angle_xt*180/np.pi,     torsions_angle_px0*180/np.pi, radians=False)*np.pi/180
        b = th_min_angle(torsions_angle_xt_alt*180/np.pi, torsions_angle_px0*180/np.pi, radians=False)*np.pi/180
        condition = torch.abs(a) < torch.abs(b)

        torsions_xt_mindiff = torch.where(condition, torsions_angle_xt, torsions_angle_xt_alt)
        mindiff = torch.where(condition, a, b)

        # these are in radians now 
        start,end = torsions_xt_mindiff.flatten(), torsions_angle_px0.flatten()


        if self.chi_type == 'interp':
            # everybody who is being diffused is going 1/t of the way to px0
            # so interpolate over t steps and use the first interpolation point 
            next_interp = th_interpolate_angle_single(start*180/np.pi, end*180/np.pi, 1, t, mindiff=None, radians=False)*np.pi/180
            next_interp = next_interp.reshape(L,10).float()

            # don't allow movement where diffusion mask is True 
            if diffusion_mask is not None:
                next_interp[diffusion_mask,:] = end.reshape(L,10)[diffusion_mask]

            temp=0
            if t > 1:
                next_torsions = next_interp
            elif t == 1:
                next_torsions = torsions_angle_px0
            else:
                # t is 0
                raise RuntimeError('Cannot operate on t=0 input, lowest is t=1')


        elif self.chi_type == 'wrapped_normal':
            # sample from wrapped normal distributions using dynamic aa schedule 
            # Instead of deterministic interpolation, sample from p(Xt-1|Xt,X0)

            #T      = torch.from_numpy(self.idx2steps).repeat(10,1).T # L,10 repeat of idx2steps 
            T      = self.decode_scheduler.idx2steps.repeat(10,1).T # L,10 repeat of idx2steps
            T_flat = T.flatten()

            # sample mean angles and stddevs based on 
            # start angle, end angle, t and dynamic T's 

            mu, sigma = self.get_dynamic_mu_sigma(chi_t=start, 
                                                  chi_0=end, 
                                                  T=T_flat, 
                                                  t=t, 
                                                  chi_beta_T=self.chi_beta_T,
                                                  beta_0=0.001)

            # now sample from wrapped normal with scheduled mean and variance to get new angles 
            next_torsions = torch.normal(mu, sigma*noise_scale).reshape(L,10) % (2*np.pi)


            if diffusion_mask is not None:
                # replace the sampled torsions with the torsions from the motif 
                # next_torsions[diffusion_mask,...] = torsions_angle_xt[diffusion_mask,...]
                next_torsions[diffusion_mask,...] = torsions_xt_mindiff[diffusion_mask,...]

        else:
            raise NotImplementedError(f'chi_type {self.chi_type} not implemented yet.')

        next_torsions_trig = torch.stack([torch.cos(next_torsions),
                                          torch.sin(next_torsions)], dim=-1)

        next_seq = self.reveal_residues(seq_t, seq_px0, px0, t)

        return next_torsions_trig, next_seq


    def align_to_xt_motif(self, px0, xT, diffusion_mask, eps=1e-6):
        """
        Need to align px0 to motif in xT. This is to permit the swapping of residue positions in the px0 motif for the true coordinates.
        First, get rotation matrix from px0 to xT for the motif residues.
        Second, rotate px0 (whole structure) by that rotation matrix
        Third, centre at origin
        """

        #if True:
        #    return px0
        def rmsd(V,W, eps=0):
            # First sum down atoms, then sum down xyz
            N = V.shape[-2]
            return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)

        assert xT.shape[1] == px0.shape[1], f'xT has shape {xT.shape} and px0 has shape {px0.shape}'

        L,n_atom,_ = xT.shape # A is number of atoms
        atom_mask = ~torch.isnan(px0)
        #convert to numpy arrays
        px0 = px0.cpu().detach().numpy()
        xT = xT.cpu().detach().numpy()
        diffusion_mask = diffusion_mask.cpu().detach().numpy()

        #1 centre motifs at origin and get rotation matrix
        px0_motif = px0[diffusion_mask,:3].reshape(-1,3)
        xT_motif  =  xT[diffusion_mask,:3].reshape(-1,3)
        px0_motif_mean = np.copy(px0_motif.mean(0)) #need later
        xT_motif_mean  = np.copy(xT_motif.mean(0))

        # center at origin
        px0_motif  = px0_motif-px0_motif_mean
        xT_motif   = xT_motif-xT_motif_mean

        # A = px0_motif
        # B = xT_motif 
        A = xT_motif
        B = px0_motif

        C = np.matmul(A.T, B)

        # compute optimal rotation matrix using SVD
        U,S,Vt = np.linalg.svd(C)


        # ensure right handed coordinate system
        d = np.eye(3)
        d[-1,-1] = np.sign(np.linalg.det(Vt.T@U.T))

        # construct rotation matrix
        R = Vt.T@d@U.T

        # get rotated coords
        rB = B@R

        # calculate rmsd
        rms = rmsd(A,rB)
        self._log.info(f'Sampled motif RMSD: {rms:.2f}')

        #2 rotate whole px0 by rotation matrix
        atom_mask = atom_mask.cpu()
        px0[~atom_mask] = 0 #convert nans to 0
        px0 = px0.reshape(-1,3) - px0_motif_mean
        px0_ = px0 @ R
        # xT_motif_out = xT_motif.reshape(-1,3)
        # xT_motif_out = (xT_motif_out @ R ) + px0_motif_mean
        # ic(xT_motif_out.shape)
        # xT_motif_out = xT_motif_out.reshape((diffusion_mask.sum(),3,3))


        #3 put in same global position as xT
        px0_ = px0_ + xT_motif_mean
        px0_ = px0_.reshape([L,n_atom,3])
        px0_[~atom_mask] = float('nan')
        return torch.Tensor(px0_)
        # return torch.tensor(xT_motif_out)


    def get_potential_gradients(self, seq, xyz, diffusion_mask ):
        '''
        This could be moved into potential manager if desired - NRB

        Function to take a structure (x) and get per-atom gradients used to guide diffusion update

        Inputs:

            seq (torch.tensor, required): [L,?] The current sequence. TODO determine sequence representation being used
            xyz (torch.tensor, required): [L,27,3] Coordinates at which the gradient will be computed

        Outputs:

            Ca_grads (torch.tensor): [L,3] The gradient at each Ca atom
        '''

        if self.potential_manager == None or self.potential_manager.is_empty(): return torch.zeros(xyz.shape[0], 3)

        use_Cb = False

        # seq.requires_grad = True
        xyz.requires_grad = True

        if not seq.grad is None: seq.grad.zero_()
        if not xyz.grad is None: xyz.grad.zero_()

        current_potential = self.potential_manager.compute_all_potentials(seq, xyz)
        current_potential.backward()

        # Since we are not moving frames, Cb grads are same as Ca grads - NRB
        # Need access to calculated Cb coordinates to be able to get Cb grads though
        Ca_grads = xyz.grad[:,1,:]

        if not diffusion_mask == None:
            Ca_grads[diffusion_mask,:] = 0

        # check for NaN's 
        if torch.isnan(Ca_grads).any():
            print('WARNING: NaN in potential gradients, replacing with zero grad.')
            Ca_grads[:] = 0

        return Ca_grads

    def get_next_pose(self,
                      xt,
                      px0,
                      t,
                      diffusion_mask,
                      seq_diffusion_mask,
                      seq_t,
                      pseq0,
                      diffuse_sidechains,
                      fix_motif=True,
                      align_motif=True,
                      include_motif_sidechains=True):
        """
        Wrapper function to take px0, xt and t, and to produce xt-1
        First, aligns px0 to xt
        Then gets coordinates, frames and torsion angles

        Parameters:
            
            xt (torch.tensor, required): Current coordinates at timestep t 

            px0 (torch.tensor, required): Prediction of x0 

            t (int, required): timestep t

            diffusion_mask (torch.tensor, required): Mask for structure diffusion

            seq_diffusion_mask (torch.tensor, required): Mask for sequence diffusion

            seq_t (torch.tensor, required): [L,22] Sequence at the current timestep 

            pseq0 (torch.tensor, required): AR decoding: [L,22] Seq Diff: [L,20] Model's prediction of sequence

            diffuse_sidechains (bool): Do diffusive sidechain prediction

            fix_motif (bool): Fix the motif structure

            align_motif (bool): Align the model's prediction of the motif to the input motif

            include_motif_sidechains (bool): Provide sidechains of the fixed motif to the model
        """

        get_allatom = ComputeAllAtomCoords().to(device=xt.device)
        L,n_atom = xt.shape[:2]
        assert (xt.shape[1]  == 14) or (xt.shape[1]  == 27)
        assert (px0.shape[1] == 14) or (px0.shape[1] == 27)# need full atom rep for torsion calculations   

        pseq0 = pseq0.to(seq_t.device)

        #align to motif
        # DJ - I altered self.align_to_xt_motif to do a different alignment
        #      Commenting both out for now. Next line was using original functionality,
        #      Line after that uses new functionality which aligns xt motif to px0 motif
        if align_motif and diffusion_mask.any():
            px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        # xT_motif_aligned = self.align_to_xt_motif(px0, xt, diffusion_mask)

        px0=px0.to(xt.device)
        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if not fix_motif:
            diffusion_mask[:] = False

        # get the next set of CA coordinates 
        noise_scale_ca = self.noise_schedule_ca(t)
        _, ca_deltas = get_next_ca(xt, px0, t, diffusion_mask,
                crd_scale=self.crd_scale, beta_schedule=self.schedule, alphabar_schedule=self.alphabar_schedule, noise_scale=noise_scale_ca, rotation_scaling=self.rotation_scaling)
        
        # get the next set of backbone frames (coordinates)
        noise_scale_frame = self.noise_schedule_frame(t)
        frames_next = get_next_frames(xt, px0, t, diffuser=self.diffuser,
                so3_type=self.so3_type, diffusion_mask=diffusion_mask, noise_scale=noise_scale_frame, rotation_scaling=self.rotation_scaling)

        # Apply gradient step from guiding potentials
        # This can be moved to below where the full atom representation is calculated to allow for potentials involving sidechains
        
        grad_ca = self.get_potential_gradients(seq_t.clone(), xt.clone(), diffusion_mask=diffusion_mask)
    
        ca_deltas += self.potential_manager.get_guide_scale(t) * grad_ca
        
        # add the delta to the new frames 
        frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate

        if diffuse_sidechains:
            if self.seq_diffuser:
                raise NotImplementedError('Sidechain diffusion and sequence diffusion cannot be performed at the same time')

            seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
            pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
            torsions_next, seq_next = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask, noise_scale = self.noise_scale_torsion)
            # build full atom representation with the new torsions but the current seq
            _, fullatom_next =  get_allatom(seq_t[None], frames_next[None], torsions_next[None])
            seq_next = torch.nn.functional.one_hot(
                    seq_next, num_classes=22).float()

        else:
            fullatom_next = torch.full_like(xt,float('nan')).unsqueeze(0)
            fullatom_next[:,:,:3] = frames_next[None]
            # This is never used so just make it a fudged tensor - NRB
            torsions_next = torch.zeros(1,1)

            if self.seq_diffuser:
                seq_next = self.seq_diffuser.get_next_sequence(seq_t[:,:20], pseq0, t, seq_diffusion_mask) # [L,20]

                # Add zeros to make the sequence have 22 classes and match the AR case
                zeros = torch.zeros(L,2)
                seq_next = torch.cat((seq_next, zeros), dim=-1) # [L,22]
            else:
                seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
                pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
                seq_next = self.reveal_residues(seq_t, pseq0, px0, t)
                seq_next = torch.nn.functional.one_hot(
                        seq_next, num_classes=22).float()
        
        if include_motif_sidechains:
            fullatom_next[:,diffusion_mask,:14] = xt[None,diffusion_mask]

        return fullatom_next.squeeze()[:,:14,:], seq_next, torsions_next, px0

def preprocess(seq, xyz_t, t, T, ppi_design, binderlen, target_res, device):
    """
    Function to prepare inputs to diffusion model
        
        seq (torch.tensor, required): (L) integer sequence 

        msa_masked (torch.tensor, required): (1,1,L,48)

        msa_full  (torch,.tensor, required): (1,1, L,25)
        
        xyz_t (torch,tensor): (L,14,3) template crds (diffused) 
        
        t1d (torch.tensor, required): (1,L,22) this is the t1d before tacking on the chi angles. Last plane is 1/t (conf hacked as timestep)
    """
    L = seq.shape[-1]
    ### msa_masked ###
    ##################
    msa_masked = torch.zeros((1,1,L,48))
    msa_masked[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]
    msa_masked[:,:,:,22:44] = nn.one_hot(seq, num_classes=22)[None, None]

    ### msa_full ###
    ################
    msa_full = torch.zeros((1,1,L,25))
    msa_full[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.zeros((1,1,L,21))
    t1d[:,:,:,:21] = nn.one_hot(torch.where(seq == 21, 20, seq), num_classes=21)[None,None]
    
    """
    DJ -  next line (commented out) was an attempt to see if model more readily 
          moved the motif if they were all set to same confidence 
          in order to alleveate chainbreaks at motif. Maybe sorta worked? definitely not full fix
    """
    # conf = conf = torch.where(seq == 21, 1-t/T, 1-t/T)[None,None,...,None]
    conf = torch.where(seq == 21, 1-t/T, 1.)[None,None,...,None]
    t1d = torch.cat((t1d, conf), dim=-1)

    # NRB: Adding in dimension for target hotspot residues
    target_residue_feat = torch.zeros_like(t1d[...,0])[...,None]
    if ppi_design and not target_res is None:
        absolute_idx = [resi+binderlen for resi in target_res]
        target_residue_feat[...,absolute_idx,:] = 1

    t1d = torch.cat((t1d, target_residue_feat), dim=-1)

    ### xyz_t ###
    #############
    xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
    xyz_t=xyz_t[None, None]
    xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)
    
    ### t2d ###
    ###########
    t2d = xyz_to_t2d(xyz_t)
  
    ### idx ###
    ###########
    idx = torch.arange(L)[None]
    if ppi_design:
        idx[:,binderlen:] += 200

    ### alpha_t ###
    ###############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
    
    #put tensors on device
    msa_masked = msa_masked.to(device)
    msa_full = msa_full.to(device)
    seq = seq.to(device)
    xyz_t = xyz_t.to(device)
    idx = idx.to(device)
    t1d = t1d.to(device)
    t2d = t2d.to(device)
    alpha_t = alpha_t.to(device)
    return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True, parse_ab_loops=False):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num
    loops = {
             'H1': [],
             'H2': [],
             'H3': [],
             'L1': [],
             'L2': [],
             'L3': []
            }

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:21] == 'REMARK PDBinfo-LABEL:':
            if not parse_ab_loops: continue

            loop_type = l[27:].strip()
            loop_idx  = int(l[22:26])
            if loop_type not in loops:
                print(f'WARNING: encountered unknown loop type: {loop_type} while parsing pdb. Skipping this loop')
                continue

            loops[loop_type].append(loop_idx)
            continue
            
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]][:14]): # Nate's proposed change
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

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
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    mask_h1 = torch.zeros(xyz.shape[0]).bool()
    mask_h2 = torch.zeros(xyz.shape[0]).bool()
    mask_h3 = torch.zeros(xyz.shape[0]).bool()
    mask_l1 = torch.zeros(xyz.shape[0]).bool()
    mask_l2 = torch.zeros(xyz.shape[0]).bool()
    mask_l3 = torch.zeros(xyz.shape[0]).bool()
    loop_key = {
                'H1': mask_h1,
                'H2': mask_h2,
                'H3': mask_h3,
                'L1': mask_l1,
                'L2': mask_l2,
                'L3': mask_l3
               }

    # Parse the dict of loop indices (1-indexed) to torch mask tensors
    for loop in loops:
        for idx in loops[loop]:
            loop_key[loop][idx-1] = True

    out = {
        'xyz':xyz, # cartesian coordinates, [Lx14]
        'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
        'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
        'seq':np.array(seq), # amino acid sequence, [L]
        'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
        'mask_h1': mask_h1,  # A mask indicating which residues belong to the H1 loop 
        'mask_h2': mask_h2,   
        'mask_h3': mask_h3,   
        'mask_l1': mask_l1,   
        'mask_l2': mask_l2,   
        'mask_l3': mask_l3,   
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out


def parse_a3m(filename):

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



def process_target(pdb_path, parse_hetatom=False, center=True):

    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct['xyz'][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0
    xyz = torch.from_numpy(target_struct['xyz'] - ca_center)
    seq_orig = torch.from_numpy(target_struct['seq'])
    atom_mask = torch.from_numpy(target_struct['mask'])
    seq_len = len(xyz)

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    xyz_27[:, :14, :] = xyz[:, :14, :]

    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :14] = atom_mask
    out = {
           'xyz_27': xyz_27,
            'mask_27': mask_27,
            'seq': seq_orig,
            'pdb_idx': target_struct['pdb_idx']
            } 
    if parse_hetatom:
        out['xyz_het'] = target_struct['xyz_het']
        out['info_het'] = target_struct['info_het']
    return out
    

def get_idx0_hotspots(mappings, ppi_conf, binderlen):
    '''
    Take pdb-indexed hotspot resudes and the length of the binder, and makes the 0-indexed tensor of hotspots
    '''
    
    hotspot_idx=None
    if binderlen > 0:
        if ppi_conf.hotspot_res is not None:
            assert all([i[0].isalpha() for i in ppi_conf.hotspot_res]), "Hotspot residues need to be provided in pdb-indexed form. E.g. A100,A103"
            hotspots = [(i[0],int(i[1:])) for i in ppi_conf.hotspot_res]
            hotspot_idx=[]
            for i,res in enumerate(mappings['receptor_con_ref_pdb_idx']):
                if res in hotspots:
                    hotspot_idx.append(mappings['receptor_con_hal_idx0'][i])
    return hotspot_idx

def recycle_schedule(T, rec_sched=None, num_designs=1):
    """  
    Function to convert input recycle schedule into a list of recycles.
    Input:
        - T: Max T
        - rec_sched: timestep:num_recycles|timestep:num_recycles
            e.g. T=200, rec_sched = 50:2/25:4/2:10. At timestep 50, start having 2 recycles, then 4 at t=25, 2=10
    """
    if rec_sched is not None:
        schedule = np.ones(T, dtype='int')
        if "/" in rec_sched:
            parts = rec_sched.split("/")
        else:
            parts = [rec_sched]
        indices=[int(i.split(":")[0]) for i in parts]
        assert all(indices[i] > indices[i+1] for i in range(len(indices) - 1)), "Recycle schedule indices must be in decreasing order"
        for part in parts:
            idx, num = part.split(":")
            schedule[:int(idx)] = int(num)
    else:
        schedule = np.ones(T, dtype='int') * int(num_designs)
    return schedule
 
class BlockAdjacency():
    """
    Class for handling PPI design inference with ss/block_adj inputs.
    Basic idea is to provide a list of scaffolds, and to output ss and adjacency
    matrices based off of these, while sampling additional lengths.
    Inputs:
        - scaffold_list: list of scaffolds (e.g. ['2kl8','1cif']). Can also be a .txt file.
        - scaffold dir: directory where scaffold ss and adj are precalculated
        - sampled_insertion: how many additional residues do you want to add to each loop segment? Randomly sampled 0-this number
        - sampled_N: randomly sample up to this number of additional residues at N-term
        - sampled_C: randomly sample up to this number of additional residues at C-term
        - ss_mask: how many residues do you want to mask at either end of a ss (H or E) block. Fixed value
        - num_designs: how many designs are you wanting to generate? Currently only used for bookkeeping
        - systematic: do you want to systematically work through the list of scaffolds, or randomly sample (default)
        - num_designs_per_input: Not really implemented yet. Maybe not necessary
    Outputs:
        - L: new length of chain to be diffused
        - ss: all loops and insertions, and ends of ss blocks (up to ss_mask) set to mask token (3). Onehot encoded. (L,4)
        - adj: block adjacency with equivalent masking as ss (L,L)     
    """

    def __init__(self, conf, num_designs):
        """
        Parameters:
          inputs:
             conf.scaffold_list as conf
             conf.inference.num_designs for sanity checking
        """

        # either list or path to .txt file with list of scaffolds
        if conf.scaffold_list is not None:
            if type(conf.scaffold_list) == list:
                self.scaffold_list = scaffold_list
            elif conf.scaffold_list[-4:] == '.txt':
                #txt file with list of ids
                list_from_file = []
                with open(conf.scaffold_list,'r') as f:
                    for line in f:
                        list_from_file.append(line.strip())
                self.scaffold_list = list_from_file
            else:
                raise NotImplementedError
        else:
            self.scaffold_list=[os.path.split(i)[1][:-6] for i in glob.glob(f'{conf.scaffold_dir}/*_ss.pt')]
        print(self.scaffold_list)
        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = conf.scaffold_dir

        # maximum sampled insertion in each loop segment
        self.sampled_insertion = conf.sampled_insertion

        # maximum sampled insertion at N- and C-terminus
        if '-' in str(conf.sampled_N):
            self.sampled_N = [int(str(conf.sampled_N).split("-")[0]), int(str(conf.sampled_N).split("-")[1])]
        else:
            self.sampled_N = [0, int(conf.sampled_N)]
        if '-' in str(conf.sampled_C):
            self.sampled_C = [int(str(conf.sampled_C).split("-")[0]), int(str(conf.sampled_C).split("-")[1])]
        else:
            self.sampled_C = [0, int(conf.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = conf.ss_mask

        # whether or not to work systematically through the list
        self.systematic = conf.systematic

        self.num_designs = num_designs

        if len(self.scaffold_list) > self.num_designs:
            print("WARNING: Scaffold set is bigger than num_designs, so not every scaffold type will be sampled")


        # for tracking number of designs
        self.num_completed = 0
        if self.systematic:
            self.item_n = 0
        
        # whether to mask loops or not
        if not conf.mask_loops:
            assert conf.sampled_N == 0, "can't add length if not masking loops"
            assert conf.sampled_C == 0, "can't add lemgth if not masking loops"
            assert conf.sampled_insertion == 0, "can't add length if not masking loops"
            self.mask_loops = False
        else:
            self.mask_loops = True
    def get_ss_adj(self, item):
        """
        Given at item, get the ss tensor and block adjacency matrix for that item
        """
        ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'))
        adj = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt'))

        return ss, adj

    def mask_to_segments(self, mask):
        """
        Takes a mask of True (loop) and False (non-loop), and outputs list of tuples (loop or not, length of element)
        """
        segments = []
        begin=-1
        end=-1
        for i in range(mask.shape[0]):
            # Starting edge case
            if i == 0:
                begin = 0
                continue

            if not mask[i] == mask[i-1]:
                end=i
                if mask[i-1].item() is True:
                    segments.append(('loop', end-begin))
                else:
                    segments.append(('ss', end-begin))
                begin = i

        # Ending edge case: last segment is length one
        if not end == mask.shape[0]:
            if mask[i].item() is True:
                segments.append(('loop', mask.shape[0]-begin))
            else:
                segments.append(('ss', mask.shape[0]-begin))
        return segments

    def expand_mask(self, mask, segments):
        """
        Function to generate a new mask with dilated loops and N and C terminal additions
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss == 'ss':
                output.extend(length*[True])
            else:
                # randomly sample insertion length
                ins = random.randint(0, self.sampled_insertion)
                output.extend((length + ins)*[False])
        output.extend(C_add*[False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(self, ss, adj, mask, expanded_mask):
        """
        Given an expanded mask, populate a new ss and adj based on this
        """
        ss_out = torch.ones(expanded_mask.shape[0])*3 #set to mask token
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), 0.)
        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        #mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:,~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:,mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out


    def mask_ss_adj(self, ss, adj, expanded_mask):
        """
        Given an expanded ss and adj, mask some number of residues at either end of non-loop ss
        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            for i in range(1, self.ss_mask+1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]

        if self.mask_loops:
            ss[~expanded_mask] = 3
            adj[~expanded_mask,:] = 0
            adj[:,~expanded_mask] = 0

        # mask adjacency
        adj[~expanded_mask] = 2
        adj[:,~expanded_mask] = 2

        return ss, adj

    def get_scaffold(self):
        """
        Wrapper method for pulling an item from the list, and preparing ss and block adj features
        """
        if self.systematic:
            # reset if num designs > num_scaffolds
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        print("Scaffold constrained based on file: ", item)
        # load files
        ss, adj = self.get_ss_adj(item)
        adj_orig=torch.clone(adj)
        # separate into segments (loop or not)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)

        # insert into loops to generate new mask
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        # and then update num_completed
        self.num_completed += 1

        return ss.shape[0], torch.nn.functional.one_hot(ss.long(), num_classes=4), adj

class Target():
    """
    Class to handle targets (fixed chains).
    Inputs:
        - path to pdb file
        - hotspot residues, in the form B10,B12,B60 etc
        - whether or not to crop, and with which method
    Outputs:
        - Dictionary of xyz coordinates, indices, pdb_indices, pdb mask
    """

    def __init__(self, conf: DictConfig, hotspots=None):
        
        self.pdb = parse_pdb(conf.target_path)
        
        if hotspots is not None:
            self.hotspots = hotspots
        else:
            self.hotspots = []
        # sanity check
        if conf.radial_crop is not None and conf.contig_crop is not None:
            raise ValueError("Cannot have both radial and contig cropping")

        if conf.radial_crop:
#             self.pdb = self.radial_crop(radial_crop)
            raise NotImplementedError("Haven't implemented radial cropping yet")

        elif conf.contig_crop:
            self.pdb = self.contig_crop(conf.contig_crop)
    
    def parse_contig(self, contig_crop):
        """
        Takes contig input and parses
        """
        contig_list = []
        for contig in contig_crop.split(" "):
            subcon=[]
            for crop in contig.split(","):
                if crop[0].isalpha():
                    subcon.extend([(crop[0], p) for p in np.arange(int(crop.split("-")[0][1:]), int(crop.split("-")[1])+1)])
            contig_list.append(subcon)

        return contig_list

    def contig_crop(self, contig_crop, residue_offset=200):
        """
        Method to take a contig string referring to the receptor and output a pdb dictionary with just this crop
        NB there are two ways to provide inputs:
            - 1) e.g. B1-30,0 B50-60,0. This will add a residue offset between each chunk
            - 2) e.g. B1-30,B50-60,B80-100. This will keep the original indexing of the pdb file. 
        Can handle the target being on multiple chains
        """

        # add residue offset between chains if multiple chains in receptor file
        for idx, val in enumerate(self.pdb['pdb_idx']):
            if idx != 0 and val != self.pdb['pdb_idx'][idx-1]:
                self.pdb['idx'][idx:] += (residue_offset + idx)


        # convert contig to mask
        contig_list = self.parse_contig(contig_crop)

        # add residue offset to different parts of contig_list
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb['idx'][start:] += residue_offset

        contig_list = np.array(contig_list).flatten()

        mask = np.array([True if i in contig_list else False for i in self.pdb['pdb_idx']])

        # sanity check
        assert np.sum(self.pdb['hotspots']) == np.sum(self.pdb['hotspots'][mask]), "Supplied hotspot residues are missing from the target contig!"

        for key, val in self.pdb:
            self.pdb[key] = val[mask]

    def centre_pdb(self):
        self.pdb['xyz'] = self.pdb['xyz'] - self.pdb['xyz'][:,:1,:].mean(axis=0)

    def radial_crop(self, radial_crop):
        #TODO
        pass

    def get_target(self):
        return self.pdb

def assemble_config_from_chk(conf: DictConfig):
    """
    Function for loading model config from checkpoint directly.
    
    Takes:
        - config file
    
    Actions:
        - Loads model checkpoint and looks for "Config"
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    """
    
    # TODO change this so we don't have to load the model twice
    pass

