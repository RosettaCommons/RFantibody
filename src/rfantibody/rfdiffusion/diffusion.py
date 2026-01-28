# script for diffusion protocols 
import logging
import os
import pickle
import time
from typing import List

import numpy as np
import torch
from icecream import ic
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp

import rfantibody.rfdiffusion.igso3 as igso3
import rfantibody.rfdiffusion.rotation_conversions as rotation_conversions
from rfantibody.rfdiffusion.chemical import INIT_CRDS
from rfantibody.rfdiffusion.diff_util import (
    get_aa_schedule,
    th_interpolate_angles,
    th_min_angle,
)
from rfantibody.rfdiffusion.util import get_torsions
from rfantibody.rfdiffusion.util import reference_angles as REF_ANGLES
from rfantibody.rfdiffusion.util import rigid_from_3_points
from rfantibody.rfdiffusion.util import torsion_can_flip as TOR_CAN_FLIP
from rfantibody.rfdiffusion.util import torsion_indices as TOR_INDICES
from rfantibody.rfdiffusion.util_module import ComputeAllAtomCoords

torch.set_printoptions(sci_mode=False)

def cosine_interp(T, eta_max, eta_min):
    """
    Cosine interpolation of some value between its max <eta_max> and its min <eta_min>

    from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    
    Parameters:
        T (int, required): total number of steps 
        eta_max (float, required): Max value of some parameter eta 
        eta_min (float, required): Min value of some parameter eta 
    """
    
    t = torch.arange(T)
    out = eta_max + 0.5*(eta_min-eta_max)*(1+torch.cos((t/T)*np.pi))
    return out 

def get_chi_betaT(max_timestep=100, beta_0=0.01, abar_T=1e-3, method='cosine'):
    """
    Function to precalculate beta_T for chi angles (decoded at different time steps, so T in beta_T varies).
    Calculated empirically
    """

    name = f'./cached_schedules/T{max_timestep}_beta_0{beta_0}_abar_T{abar_T}_method_{method}.pkl'

    if not os.path.exists(name):
        print('Calculating chi_beta_T dictionary...')

        if method not in ['cosine', 'linear']:
            raise NotImplementedError("Only cosine and linear interpolations are implemented for chi angle beta schedule")
        beta_Ts = {1:1.}
        for timestep in range(2,101):
            best=999.99
            for i in torch.linspace(beta_0,0.999,5000): #sampling bT
                if method == 'cosine':
                    interp = cosine_interp(timestep, i, beta_0)
                elif method == 'linear':
                    interp = torch.linspace(beta_0, i, timestep)
                temp = torch.cumprod(1-interp, dim=0)
                if torch.abs(temp[-1] - abar_T) < best:
                    best = temp[-1] - abar_T
                    idx = i
            beta_Ts[timestep] = idx.item()

        # save cached schedule
        if not os.path.isdir('./cached_schedules/'):
            os.makedirs('./cached_schedules/', exist_ok=True)
        with open(name, 'wb') as fp:
            pickle.dump(beta_Ts, fp)

        print('Done calculating chi_beta_T dictionaries. They are now cached.')

    else:
        print('Using cached chi_beta_T dictionary.')
        with open(name, 'rb') as fp:
            beta_Ts = pickle.load(fp)


    print('Done calculating chi_beta_T, chi_alphas_T, and chi_abars_T dictionaries.')
    return beta_Ts

def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}, inference=False):
    """
    Given a noise schedule type, create the beta schedule 
    """
    assert schedule_type in ['linear', 'geometric', 'cosine']

    # Adjust b0 and bT for different T
    # This is a good approximation, with the beta correction below, unless T is very small
    assert T >= 15, "With discrete time and T < 15, the schedule is badly approximated"
    b0 *= (200 / T)
    bT *= (200 / T)

    # linear noise schedule 
    if schedule_type == 'linear':
        schedule = torch.linspace(b0, bT, T) 

    # geometric noise schedule 
    elif schedule_type == 'geometric': 
        raise NotImplementedError('geometric schedule not ready yet')
    
    # cosine noise schedule 
    else:
        schedule = cosine_interp(T, bT, b0) 
    
    
    #get alphabar_t for convenience
    alpha_schedule = 1-schedule
    alphabar_t_schedule  = torch.cumprod(alpha_schedule, dim=0)
    
    if inference:
        print(f"With this beta schedule ({schedule_type} schedule, beta_0 = {b0}, beta_T = {bT}), alpha_bar_T = {alphabar_t_schedule[-1]}")

    return schedule, alpha_schedule, alphabar_t_schedule 


class EuclideanDiffuser():
    # class for diffusing points 

    def __init__(self,
                 T, 
                 b_0, 
                 b_T, 
                 schedule_type='linear',
                 schedule_kwargs={},
                 ):
        
        self.T = T 
        
        # make noise/beta schedule 
        self.beta_schedule, _, self.alphabar_schedule  = get_beta_schedule(T, b_0, b_T, schedule_type, **schedule_kwargs)
        self.alpha_schedule = 1-self.beta_schedule 

    
    # NOTE: this one seems fishy - doesn't match apply_kernel
    #def apply_kernel_closed(self, x0, t, var_scale=1, mask=None):
    #    """
    #    Applies a noising kernel to the points in x 

    #    Parameters:
    #        x0 (torch.tensor, required): (N,3,3) set of backbone coordinates from ORIGINAL backbone 

    #        t (int, required): Which timestep

    #        noise_scale (float, required): scale for noise 
    #    """
    #    t_idx = t-1 # bring from 1-indexed to 0-indexed

    #    assert len(x0.shape) == 3
    #    L,_,_ = x0.shape 

    #    # c-alpha crds 
    #    ca_xyz = x0[:,1,:]


    #    b_t = self.beta_schedule[t_idx]    
    #    a_t = self.alpha_schedule[t_idx]


    #    # get the noise at timestep t
    #    a_bar = torch.prod(self.alpha_schedule[:t_idx], dim=0)

    #    mean  = torch.sqrt(a_bar)*ca_xyz 
    #    var   = torch.ones(L,3)*(1-a_bar)*var_scale


    #    sampled_crds = torch.normal(mean, var)
    #    delta = sampled_crds - ca_xyz

    #    if mask != None:
    #        delta[mask,...] = 0

    #    out_crds = x0 + delta[:,None,:]

    #    return out_crds 


    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)


    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Applies a noising kernel to the points in x 

        Parameters:
            x (torch.tensor, required): (N,3,3) set of backbone coordinates 

            t (int, required): Which timestep

            noise_scale (float, required): scale for noise 
        """
        t_idx = t-1 # bring from 1-indexed to 0-indexed

        assert len(x.shape) == 3
        L,_,_ = x.shape 

        # c-alpha crds 
        ca_xyz = x[:,1,:]


        b_t = self.beta_schedule[t_idx]    


        # get the noise at timestep t
        mean  = torch.sqrt(1-b_t)*ca_xyz
        var   = torch.ones(L,3)*(b_t)*var_scale

        sampled_crds = torch.normal(mean, torch.sqrt(var)) 
        delta = sampled_crds - ca_xyz  

        if not diffusion_mask is None:
            delta[diffusion_mask,...] = 0

        out_crds = x + delta[:,None,:]

        return out_crds, delta


    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Repeatedly apply self.apply_kernel T times and return all crds 
        """
        bb_stack = []
        T_stack  = []

        cur_xyz  = torch.clone(xyz)  

        for t in range(1,self.T+1):     
            cur_xyz, cur_T = self.apply_kernel(cur_xyz, 
                                        t, 
                                        var_scale=var_scale, 
                                        diffusion_mask=diffusion_mask)
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)
        

        return torch.stack(bb_stack).transpose(0,1), torch.stack(T_stack).transpose(0,1)

        
#TODO:  This class uses scipy+numpy for the slerping/matrix generation 
#       Probably could be much faster if everything was in torch

def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, 'wb') as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f'Failed to read {read_path}')
            raise(e)

class IGSO3():
    """
    Class for taking in a set of backbone crds and performing IGSO3 diffusion
    on all of them
    """

    def __init__(self, *, T, min_sigma, max_sigma, min_b, max_b,
            cache_dir, num_omega=1000, schedule="linear", L=2000):
        """

        Args:
            T: total number of time steps
            min_sigma: smallest allowed variance, should be at least 0.01 to maintain numerical stability.  Recommended value is 0.05.
            max_sigma: for exponential schedule, the largest variance. Ignored for recommeded linear schedule
            min_b: lower value of beta in Ho schedule analogue
            max_b: upper value of beta in Ho schedule analouge
            num_omega: discretization level in the angles across [0, pi]
            schedule: currently only linear and exponential are supported.  The exponential schedule may be noising too slowly.
            ]
            L: truncation level
        """
        self._log = logging.getLogger(__name__)


        self.T = T


        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        if self.schedule == 'linear':
            self.min_b = min_b
            self.max_b = max_b
            self.max_sigma = self.sigma(1.)
        self.num_omega = num_omega
        self.num_sigma = 500
        # Calculate igso3 values.
        self.L = L # truncation level
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

    def _calc_igso3_vals(self, L=2000):
        """_calc_igso3_vals computes numerical approximations to the
        relevant analytically intractable functionals of the igso3
        distribution.

        The calculated values are cached, or loaded from cache if they already
        exist.

        Args:
            L: truncation level for power series expansion of the pdf.
        """
        replace_period = lambda x: str(x).replace('.', '_')
        if self.schedule == 'linear':
            cache_fname = os.path.join(
                self.cache_dir, f'T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}'+
                f'_min_b_{replace_period(self.min_b)}_max_b_{replace_period(self.max_b)}_schedule_{self.schedule}.pkl'
            )
        elif self.schedule == 'exponential':
            cache_fname = os.path.join(
                self.cache_dir, f'T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}'
                f'_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}'
            )
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')



        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(cache_fname):
            self._log.info('Using cached IGSO3.')
            igso3_vals = read_pkl(cache_fname)
        else:
            self._log.info('Calculating IGSO3.')
            igso3_vals = igso3.calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega,
                L=L)
            write_pkl(cache_fname, igso3_vals)

        return igso3_vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals['discrete_sigma']

    def sigma_idx(self, sigma: np.ndarray):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray):
        """Helper function to go from discrete time index t to corresponding sigma_idx.
        
        Args:
            t: time index (integer between 1 and 200) 
        """
        continuous_t = t/self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sigma(self, t: torch.tensor):
        """Extract \sigma(t) corresponding to chosen sigma schedule.
        
        Args:
            t: torch tensor with time between 0 and 1
        """
        if not type(t) == torch.Tensor: t = torch.tensor(t)
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'exponential':
            sigma = t * np.log10(self.max_sigma) + (1 - t) * np.log10(self.min_sigma)
            return 10 ** sigma
        elif self.schedule == 'linear': # Variance exploding analogue of Ho schedule
            # add self.min_sigma for stability
            return self.min_sigma + t*self.min_b  + (1/2)*(t**2)*(self.max_b - self.min_b)
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def g(self, t):
        """g returns the drift coefficient at time t

        since 
            sigma(t)^2 := \int_0^t g(s)^2 ds,
        for arbitrary sigma(t) we invert this relationship to compute 
            g(t) = sqrt(d/dt sigma(t)^2).
        
        Args:
            t: scalar time between 0 and 1
        
        Returns:
            drift cooeficient as a scalar.
        """
        t = torch.tensor(t, requires_grad=True)
        sigma_sqr = self.sigma(t)**2
        grads = torch.autograd.grad(sigma_sqr.sum(), t)[0]
        return torch.sqrt(grads)


    def sample(self, ts, n_samples=1):
        """sample uses the inverse cdf to sample an angle of rotation from
        IGSO(3)
        TODO: add shapes to this.
        Args:
            ts: array of integer time steps to sample from.
            n_samples: nubmer of samples to draw.
        Returns:
        sampled angles of rotation. [len(ts), N]
        """
        assert sum(ts==0) == 0, "assumes one-indexed, not zero indexed"
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t) 
            sample_i = np.interp(
                np.random.rand(n_samples),
                self.igso3_vals['cdf'][sigma_idx],
                self.igso3_vals['discrete_omega'])  # [N, 1]
            all_samples.append(sample_i)
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):
        """sample_vec generates a rotation vector(s) from IGSO(3) at time steps
        ts.
        Return:
            Sampled vector of shape [len(ts), N, 3]
        """
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        """score_norm computes the score norm based on the time step and angle
        Args:
            t: integer time step
            omega: angles (scalar or shape [N])
        Return:
            score_norm with same shape as omega
        """
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
                omega,
                self.igso3_vals['discrete_omega'],
                self.igso3_vals['score_norm'][sigma_idx]
                )
        return score_norm_t

    def score_vec(self, ts, vec):
        """score_vec computes the score of the IGSO(3) density as a rotation
        vector. This score vecotr is in the direction of the sampled vector,
        and has magnitude given by _score_norms.
        Args:
            ts: times of shape [T]
            vec: where to compute the score of shape [T, N, 3]
        Returns:
            score vectors of shape [T, N, 3]
        """
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i]
            t_idx = t-1 
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals['discrete_omega'],
                self.igso3_vals['score_norm'][sigma_idx]
            )[:, None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / omega[..., None]

    def exp_score_norm(self, ts):
        """exp_score_norm returns the expected value of norm of the score for
        IGSO(3) with time parameter ts of shape [T].
        """
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals['exp_score_norms'][sigma_idcs]

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        """
        Perform spherical linear interpolation from the True coordinate frame for each
        residue to a randomly sampled coordinate frame
        Parameters:
            xyz (np.array or torch.tensor, required): (L,3,3) set of backbone coordinates
            mask (np.array or torch.tensor, required): (L,) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            np.array : N/CA/C coordinates for each residue in the SLERP
                        (T,L,3,3), where T is num timesteps
        """

        if torch.is_tensor(xyz):
            xyz = xyz.numpy()

        t = np.arange(self.T)+1 # 1-indexed!! 
        num_res = len(xyz)

        N  = torch.from_numpy(  xyz[None,:,0,:]  )
        Ca = torch.from_numpy(  xyz[None,:,1,:]  )  # [1, num_res, 3, 3]
        C  = torch.from_numpy(  xyz[None,:,2,:]  )

        # scipy rotation object for true coordinates
        R_true, Ca = rigid_from_3_points(N,Ca,C)
        R_true = R_true[0]
        Ca = Ca[0]

        # Sample rotations and scores from IGSO3
        sampled_rots = self.sample_vec(t, n_samples=num_res)  # [T, N, 3]
        rot_score = self.score_vec(t, sampled_rots)  # [T, N, 3]
        rot_exp_score_norm = self.exp_score_norm(t)  # [T]

        if diffusion_mask is not None:
            non_diffusion_mask = 1 - diffusion_mask[None, :, None]
            sampled_rots = sampled_rots * non_diffusion_mask
            rot_score = rot_score * non_diffusion_mask

        # Apply sampled rot.
        R_sampled = scipy_R.from_rotvec(
            sampled_rots.reshape(-1, 3)).as_matrix().reshape(
                self.T, num_res, 3, 3)
        R_perturbed = np.einsum(
            'tnij,njk->tnik', R_sampled, R_true)
        perturbed_crds = np.einsum(
            'tnij,naj->tnai',
            R_sampled,
            xyz[:,:3,:] - Ca[:,None,...].numpy()) + Ca[None, :, None].numpy()

        if t_list != None:
            idx = [i-1 for i in t_list]
            perturbed_crds = perturbed_crds[idx]
            R_perturbed    = R_perturbed[idx]


        return (perturbed_crds.transpose(1, 0, 2, 3),   # [L, T, 3, 3]
                R_perturbed.transpose(1, 0, 2, 3))

    def reverse_sample(self, r_t, r_0, t, noise_level, mask=None, rotation_scaling=None):
        """reverse_sample uses an approximation to the IGSO3 score to sample
        a rotation at the previous time step.
        
        Roughly - this update follows the reverse time SDE for Reimannian
        manifolds proposed by de Bortoli et al. Theorem 1 [1]. But with an
        approximation to the score based on the prediction of R0.
        Unlike in reference [1], this diffusion on SO(3) relies on geometric
        variance schedule.  Specifically we follow [2] (appendix C) and assume
            sigma_t = sigma_min * (sigma_max / sigma_min)^{t/T},
        for time step t.  When we view this as a discretization  of the SDE
        from time 0 to 1 with step size (1/T).  Following Eq. 5 and Eq. 6, 
        this maps on to the forward  time SDEs
            dx = g(t) dBt [FORWARD]
        and 
            dx = g(t)^2 score(xt, t)dt + g(t) B't, [REVERSE]
        where g(t) = sigma_t * sqrt(2 * log(sigma_max/ sigma_min)), and Bt and
        B't are Brownian motions. The formula for g(t) obtains from equation 9
        of [2], from which this sampling function may be generalized to
        alternative noising schedules.
        Args:
            r_t: noisy rotation of shape [3, 3]
            r_0: prediction of un-noised rotation
            t: integer time step
            noise_level: scaling on the noise added when obtaining sample
                (preliminary performance seems empirically better with noise 
                level=0.5)
            mask: whether the residue is to be updated.  A value of 1 means the
                rotation is not updated from r_t.  A value of 0 means the
                rotation is updated.
        Return:
            sampled rotation matrix for time t-1 of shape [3, 3]
        Reference:
        [1] De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y.
        W., & Doucet, A. (2022). Riemannian score-based generative modeling.
        arXiv preprint arXiv:2202.02763.
        [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
        & Poole, B. (2020). Score-based generative modeling through stochastic
        differential equations. arXiv preprint arXiv:2011.13456.
        """
        # NB this has been written for 1-indexed t, so no need for t_idx

        # compute rotation vector corresponding to prediction of how r_t goes to r_0
        r_0, r_t = torch.tensor(r_0), torch.tensor(r_t)
        r_0t = torch.einsum('ij,kj->ik', r_t, r_0)
        r_0t_rotvec = torch.tensor(scipy_R.from_matrix(
            r_0t.cpu().numpy()).as_rotvec()).to(r_0.device)

        # Approximate the score based on the prediction of R0.
        # This approximation would be exactly equal to the conditional score
        # grad_{rt} \log p(r_t |r_0) if the prediction of r_0 were exactly
        # equal to r_0.  While this will not be the case in practice, the below
        # approximation puts the magnitude score_approx on the appropriate
        # scale as a function of variance at time t.  Additionally, scaling
        # implicitly provides a roughly linear scaling in the size of the 
        # update of the  rotation with the distance of r_0 to 
        omega = torch.linalg.norm(r_0t_rotvec).numpy()
        score_approx = r_0t_rotvec*self.score_norm(t, omega)/omega  

        # Compute scaling for score and sampled noise (following Eq 6 of [2])
        continuous_t = t/self.T
        rot_g = self.g(continuous_t).to(score_approx.device)

        # Sample and scale noise to add to the rotation perturbation in the
        # SO(3) tangent space.  Since IG-SO(3) is the Brownian motion on SO(3)
        # (up to a deceleration of time by a factor of two), for small enough 
        # time-steps, this is equivalent to perturbing r_t with IG-SO(3) noise.
        # See e.g. Algorithm 1 of De Bortoli et al.
        z = np.random.normal(size=(3))
        z = torch.Tensor(
            z.reshape(3)).to(score_approx.device)
        z *= noise_level # scale down added noise by noise_level

        # sample perturbation from discretized SDE (following eq. 6 of [2])
        if rotation_scaling is None:
            perturb_rotvec = (rot_g ** 2) * self.step_size * score_approx + rot_g * np.sqrt(self.step_size) * z
        else:
            perturb_rotvec = (rot_g ** 2) * self.step_size * score_approx * 0.5 * rotation_scaling

        # Mask perturbation if residue is masked
        if mask is not None: perturb_rotvec *= (1-mask.long())
        # Convert perturbation to a rotation matrix and apply to r_t
        perturb = rotation_conversions.axis_angle_to_matrix(perturb_rotvec)
        interp_rot = torch.einsum('ij,jk->ik', perturb, r_t) # interp_rot represents the sampled r_t-1
        return interp_rot


    def reverse_sample_vectorized(self, R_t, R_0, t, noise_level, mask=None, return_perturb=False, rotation_scaling=None):
        """ Vectorized version of reverse_sample() """

        R_0, R_t = torch.tensor(R_0), torch.tensor(R_t)
        R_0t = torch.einsum('...ij,...kj->...ik', R_t, R_0)
        R_0t_rotvec = torch.tensor(scipy_R.from_matrix(
                    R_0t.cpu().numpy()).as_rotvec()).to(R_0.device)


        Omega = torch.linalg.norm(R_0t_rotvec, axis=-1).numpy()
        Score_approx = R_0t_rotvec*(self.score_norm(t, Omega)/Omega)[:,None]

        continuous_t = t/self.T
        rot_g = self.g(continuous_t).to(Score_approx.device)

        Z = np.random.normal(size=(R_0.shape[0], 3))
        Z = torch.from_numpy(Z).to(Score_approx.device)
        Z *= noise_level
        
        if rotation_scaling is None:
            Perturb_rotvec = (rot_g ** 2) * self.step_size * Score_approx + rot_g * np.sqrt(self.step_size) * Z
        else:
            Perturb_rotvec = (rot_g ** 2) * self.step_size * Score_approx * 0.5 * rotation_scaling

        if mask is not None: Perturb_rotvec *= (1-mask.long())[:,None]

        Perturb = rotation_conversions.axis_angle_to_matrix(Perturb_rotvec)
        if return_perturb :
            return Perturb

        Interp_rot = torch.einsum('...ij,...jk->...ik', Perturb, R_t)

        return Interp_rot

class SLERP():
    """
    Class for taking in a set of backbone crds and performing slerp
    on all of them 
    """

    def __init__(self, T):

        self.T = T 

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        # slerp
        slerped_crds, slerped_frames = self.slerp(xyz, diffusion_mask)

        # filter 
        if t_list != None:
            t_idx = [t-1 for t in t_list]
            slerped_crds   = slerped_crds[:,t_idx]
            slerped_frames = slerped_frames[:,t_idx]

        return slerped_crds, slerped_frames 
    
    def slerp(self, xyz, diffusion_mask=None):
        """
        Perform spherical linear interpolation from the True coordinate frame for each 
        residue to a randomly sampled coordinate frame 

        Parameters:
            xyz (np.array or torch.tensor, required): (L,3,3) set of backbone coordinates 

            mask (np.array or torch.tensor, required): (L,1) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            np.array : N/CA/C coordinates for each residue in the SLERP 
                        (T,L,3,3), where T is num timesteps
            
        """
        # diffusion_mask = None 

        if torch.is_tensor(xyz):
            xyz = xyz.numpy()

        t = np.arange(self.T)
        alpha = t/self.T
        
        R_rand = scipy_R.random(len(xyz))
        
        N  = torch.from_numpy(  xyz[None,:,0,:]  )
        Ca = torch.from_numpy(  xyz[None,:,1,:]  )
        C  = torch.from_numpy(  xyz[None,:,2,:]  )
        
        # scipy rotation object for true coordinates
        R_true, Ca = rigid_from_3_points(N,Ca,C)
        R_true = scipy_R.from_matrix(R_true.squeeze())
        
        # bad - could certainly vectorize somehow 
        all_interps = []
        for i in range(len(xyz)):

            r_true = R_true[i].as_matrix()
            r_rand = R_rand[i].as_matrix()

            # handle potential nans in BB frames / crds 
            if not np.isnan(r_true).any():
            
                if not diffusion_mask[i]:
                    key_rots = scipy_R.from_matrix(np.stack([r_true, r_rand], axis=0))
                else:
                    key_rots = scipy_R.from_matrix(np.stack([r_true, r_true], axis=0))

            else:
                key_rots = scipy_R.from_matrix(np.stack([np.eye(3), np.eye(3)], axis=0))
        
            key_times = [0,1]
        
            interpolator = Slerp(key_times, key_rots)
            interp_time = alpha
            
            # grab the interpolated FRAMES 
            interp_frame  = interpolator(interp_time)
            
            # construct the rotation matrix which when applied YIELDS interpolated frames 
            interp_rot = (interp_frame.as_matrix().squeeze() @ np.linalg.inv(r_true.squeeze()) )[None,...]

            all_interps.append(interp_rot)
        
        all_interps = np.concatenate(all_interps, axis=0)
        
        # Now apply all the interpolated rotation matrices to the original rotation matrices and get the frames at each timestep
        slerped_frames = np.einsum('lrij,ljk->lrik', all_interps, R_true.as_matrix())
        
        # apply the slerped frames to the coordinates
        slerped_crds   = np.einsum('lrij,laj->lrai', all_interps, xyz[:,:3,:] - Ca.squeeze()[:,None,...].numpy()) + Ca.squeeze()[:,None,None,...].numpy()

        # (T,L,3,3) set of backbone coordinates and frames 
        return slerped_crds, slerped_frames


class INTERP():
    """
    Class for diffusing chi angles by randomly 
    sampling in [-pi,pi] and interpolating on the unit circle from 
    random to native
    """

    def __init__(self, T):
        self.T = T

    def diffuse_torsions(self, xyz14, seq, mask14, diffusion_mask=None, n_steps=100):
        return self.interp(xyz14, seq, mask14, diffusion_mask, n_steps)
    

    def interp(self, xyz14, seq, mask14, diffusion_mask, n_steps):
        """
        Creates a linear interpolation between the true chi angle of a sidechain 
        and a randomly chosen chi angle between -pi<theta<pi

        Parameters:
            xyz14 (torch.tensor, required): (L,14,3) set of heavyatom coordinates 

            seq (torch.tensor, required): integer sequence 

            mask (torch.tensor, required): Lx14 atom mask

            decode_times (torch.tensor, optional): For each residue position, how many 
                                diffusion steps will it get. 

        Returns:
            torch.tensor - (L,T,4,2) tensor of the sin/cos of each interpolated set of chis 
        """

        RAD = True # in radians 
        L = len(xyz14)

        # calculate random decoding order - linear w.r.t. time 
        if n_steps != 0:
            decode_times, decode_order, idx2steps, aa_masks = get_aa_schedule(self.T, L, n_steps)
        else:
            decode_times = np.array([])
            decode_order = []
            idx2steps = np.zeros(L)
            aa_masks=torch.zeros(self.T, L).bool()
        idx2steps = torch.from_numpy(idx2steps)
        xyz = torch.full((L,27,3),np.nan).float()
        xyz[:,:14,:] = xyz14

        mask = torch.full((L, 27), False)
        mask[:,:14] = mask14 

        # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
        init = INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
        xyz = torch.where(mask[...,None], xyz, init).contiguous()
        xyz = torch.nan_to_num(xyz)


        # cos first, sin second 
        torsions_sincos, torsions_alt_sincos, tors_mask, tors_planar = get_torsions(xyz[None], seq[None], TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)

        # convert sin/cos to degrees
        torsions_angle     = torch.atan2(torsions_sincos[...,1], 
                                         torsions_sincos[...,0]).squeeze()*(180/np.pi)

        torsions_angle_alt = torch.atan2(torsions_alt_sincos[...,1],
                                         torsions_alt_sincos[...,0]).squeeze()*(180/np.pi)

        # Now sample random torsions from -pi < theta < pi
        r1 = -np.pi 
        r2 =  np.pi
        random_torsions = ((r1 - r2) * torch.rand(L,10,1) + r2).squeeze()*(180/np.pi)
        
        # Grab the torsions which have the minimum difference to randomly sampled one
        # Takes in degrees, then immediately change back to radians because we trust 
        # th_min_angle in degrees
        a = th_min_angle(torsions_angle,     random_torsions, radians=False)
        b = th_min_angle(torsions_angle_alt, random_torsions, radians=False)
        
        condition = torch.abs(a) < torch.abs(b)
        torsions_mindiff = torch.where(condition, torsions_angle, torsions_angle_alt)
        torsions_mindiff *= (np.pi/180)

        # Don't allow movement where diffusion_mask is True 
        if not diffusion_mask is None:
            random_torsions[diffusion_mask,...] = torsions_mindiff[diffusion_mask,...]


        # Now interpolate between the mindiff torsions and the random ones 
        c,d = torsions_mindiff.flatten(), random_torsions.flatten()
        e = idx2steps.repeat(10,1).T.flatten()
        angle_interpolations = th_interpolate_angles(c,d, self.T, n_diffuse=e).reshape(L,10,self.T)
        angle_interpolations = angle_interpolations.transpose(1,2)

        return angle_interpolations, aa_masks


class Diffuser():
    # wrapper for yielding diffused coordinates/frames/rotamers  


    def __init__(self,
                 T,
                 b_0,
                 b_T,
                 min_sigma,
                 max_sigma,
                 min_b,
                 max_b,
                 schedule_type,
                 so3_schedule_type,
                 so3_type,
                 chi_type,
                 crd_scale,
                 aa_decode_steps,
                 schedule_kwargs={},
                 chi_kwargs={},
                 var_scale=1.0,
                 cache_dir='.',
                 partial_T=None,
                 truncation_level=2000
                 ):
        """
        
        Parameters:
            truncation_level: for the igso3 numerical approximation
            
        """
        #print('**********16')
        self.T = T
        self.b_0 = b_0
        self.b_T = b_T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.aa_decode_steps=aa_decode_steps
        self.cache_dir = cache_dir

        # get backbone frame diffuser 
        if so3_type == 'slerp':
            self.so3_diffuser =  SLERP(self.T)
        elif so3_type == 'igso3':
            self.so3_diffuser =  IGSO3(
                T=self.T,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                schedule=so3_schedule_type,
                min_b=min_b,
                max_b=max_b,
                cache_dir=self.cache_dir,
                L=truncation_level, 
            )        
        else:
            raise NotImplementedError()

        # get backbone translation diffuser
        self.eucl_diffuser = EuclideanDiffuser(self.T, b_0, b_T, schedule_type=schedule_type, **schedule_kwargs)

        # get chi angle diffuser 
        self.torsion_diffuser = INTERP(self.T)

        print('Successful diffuser __init__')
    
    def diffuse_pose(self, xyz, seq, atom_mask, diffuse_sidechains=False, include_motif_sidechains=True, diffusion_mask=None, t_list=None):
        """
        Given full atom xyz, sequence and atom mask, diffuse the protein 
        translations, rotations, and chi angles

        Parameters:
            
            xyz (L,14/27,3) set of coordinates 

            seq (L,) integer sequence 

            atom_mask: mask describing presence/absence of an atom in pdb 

            diffusion_mask (torch.tensor, optional): Tensor of bools, True means NOT diffused at this residue, False means diffused

            t_list (list, optional): If present, only return the diffused coordinates at timesteps t within the list 


        """

        if diffusion_mask is None:
            diffusion_mask = torch.zeros(len(xyz.squeeze())).to(dtype=bool)

        get_allatom = ComputeAllAtomCoords().to(device=xyz.device)
        L = len(xyz)

        # bring to origin and scale 
        # check if any BB atoms are nan before centering 
        nan_mask = ~torch.isnan(xyz.squeeze()[:,:3]).any(dim=-1).any(dim=-1)
        assert torch.sum(~nan_mask) == 0

        #Centre unmasked structure at origin, as in training (to prevent information leak)
        if torch.sum(diffusion_mask) != 0:
            self.motif_com=xyz[diffusion_mask,1,:].mean(dim=0) # This is needed for one of the potentials
            xyz = xyz - self.motif_com
        elif torch.sum(diffusion_mask) == 0:
            xyz = xyz - xyz[:,1,:].mean(dim=0)

        #xyz = xyz - xyz[nan_mask][:,1,:].mean(dim=0) # DJ aug 23, 2022 - commenting out bc now better logic to assert no nans 
        xyz_true = torch.clone(xyz)

        xyz = xyz * self.crd_scale

        
        # 1 get translations 
        tick = time.time()
        diffused_T, deltas = self.eucl_diffuser.diffuse_translations(xyz[:,:3,:].clone(), diffusion_mask=diffusion_mask)
        #print('Time to diffuse coordinates: ',time.time()-tick)
        diffused_T /= self.crd_scale
        deltas     /= self.crd_scale


        # 2 get  frames
        tick = time.time()
        diffused_frame_crds, diffused_frames = self.so3_diffuser.diffuse_frames(xyz[:,:3,:].clone(), diffusion_mask=diffusion_mask.numpy(), t_list=None)
        diffused_frame_crds /= self.crd_scale 
        #print('Time to diffuse frames: ',time.time()-tick)


        ##### Now combine all the diffused quantities to make full atom diffused poses 
        tick = time.time()
        cum_delta = deltas.cumsum(dim=1)
        # The coordinates of the translated AND rotated frames
        diffused_BB = (torch.from_numpy(diffused_frame_crds) + cum_delta[:,:,None,:]).transpose(0,1) # [n,L,3,3]
        #diffused_BB  = torch.from_numpy(diffused_frame_crds).transpose(0,1)

        # Full atom diffusions at all timepoints 
        if diffuse_sidechains:
            # This section of code only works with integer sequence at the moment - NRB
            assert(seq.shape[-1] == L), 'Tried to feed non-integer sequence to diffuse torsions'

            # diffuse chi angles/planar angles and sequence information 
            tick = time.time()
            diffused_torsions,aa_masks = self.torsion_diffuser.diffuse_torsions(xyz[:,:14].clone(), 
                                                                                seq, 
                                                                                atom_mask[:,:14].clone(),
                                                                                diffusion_mask=diffusion_mask, 
                                                                                n_steps=self.aa_decode_steps)
            #print('Time to diffuse torsions: ',time.time()-tick)

            diffused_torsions_trig = torch.stack([torch.cos(diffused_torsions), 
                                              torch.sin(diffused_torsions)], dim=-1)
            fa_stack = []
            if t_list is None:
                for t,alphas_t in enumerate(diffused_torsions_trig.transpose(0,1)):
                    xyz_bb_t = diffused_BB[t,:,:3]

                    _,fullatom_t = get_allatom(seq[None], xyz_bb_t[None], alphas_t[None])
                    fa_stack.append(fullatom_t)

            else:
                for t in t_list:
                    t_idx=t-1
                    xyz_bb_t  = diffused_BB[t_idx,:,:3]
                    alphas_t = diffused_torsions_trig.transpose(0,1)[t_idx]

                    _,fullatom_t = get_allatom(seq[None], xyz_bb_t[None], alphas_t[None])
                    fa_stack.append(fullatom_t.squeeze())

            fa_stack = torch.stack(fa_stack, dim=0)

        else: 
            # Do aa_mask logic outside of diffuse torsions logic - NRB
            # This is the same code as is present in the diffuse torsions function
            if self.aa_decode_steps != 0:
                _, _, _, aa_masks = get_aa_schedule(self.T, L, self.aa_decode_steps)
            else:
                _, _, _, aa_masks = get_aa_schedule(self.T, L, 10) # 10 is dummy number. No decoding is happening. TODO clean this up

            # diffused_BB is [t_steps,L,3,3]
            t_steps, L  = diffused_BB.shape[:2]

            diffused_fa = torch.zeros(t_steps,L,27,3)
            diffused_fa[:,:,:3,:] = diffused_BB

            # Add in sidechains from motif
            if include_motif_sidechains:
                diffused_fa[:,diffusion_mask,:14,:] = xyz_true[None,diffusion_mask,:14]

            if t_list is None: fa_stack = diffused_fa
            else:
                t_idx_list = [t-1 for t in t_list]
                fa_stack = diffused_fa[t_idx_list]
                aa_masks = aa_masks[t_idx_list]

        return fa_stack, aa_masks, xyz_true
        #return diffused_T, deltas, diffused_frame_crds, diffused_frames, diffused_torsions, fa_stack, aa_masks



