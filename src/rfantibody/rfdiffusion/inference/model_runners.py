import logging
import os

import numpy as np
import torch
import torch.nn.functional as nn
from hydra.core.hydra_config import HydraConfig
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import rfantibody.rfdiffusion.util
from rfantibody.rfdiffusion.chemical import INIT_CRDS, seq2chars
from rfantibody.rfdiffusion.contigs import ContigMap
from rfantibody.rfdiffusion.diffusion import Diffuser
from rfantibody.rfdiffusion.inference import ab_pose, symmetry
from rfantibody.rfdiffusion.inference import utils as iu
from rfantibody.rfdiffusion.inference.ab_util import (
    correct_selfcond,
    featurize,
    process_init_selfcond,
    process_selfcond,
)
from rfantibody.rfdiffusion.kinematics import get_init_xyz, xyz_to_t2d
from rfantibody.rfdiffusion.potentials.manager import PotentialManager
from rfantibody.rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
from rfantibody.rfdiffusion.util import Dotdict
from rfantibody.rfdiffusion.util_module import ComputeAllAtomCoords

TOR_INDICES  = rfantibody.rfdiffusion.util.torsion_indices
TOR_CAN_FLIP = rfantibody.rfdiffusion.util.torsion_can_flip
REF_ANGLES   = rfantibody.rfdiffusion.util.reference_angles


class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def ab_design(self):
        return False

    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_override_path != self._conf.inference.ckpt_override_path

        # Assign config to Sampler
        self._conf = conf

        # Initialize inference only helper objects to Sampler
        # JW now added automatic model selection.
        if conf.inference.ckpt_override_path is not None:
            self.ckpt_path = conf.inference.ckpt_override_path
            print("WARNING: You're overriding the checkpoint path from the defaults. Check that the model you're providing can run with the inputs you're providing.")
        else:
            if conf.contigmap.inpaint_seq is not None or conf.contigmap.provide_seq is not None:
                # use model trained for inpaint_seq
                if conf.contigmap.provide_seq is not None:
                    # this is only used for partial diffusion
                    assert conf.diffuser.partial_T is not None, "The provide_seq input is specifically for partial diffusion"
                if conf.scaffoldguided.scaffoldguided:
                    self.ckpt_path='/net/databases/diffusion/models/seq_alone_models_FoldConditioned_Jan23/BFF_4.pt'
                else:
                    self.ckpt_path = '/net/databases/diffusion/models/seq_alone_models_Dec2022/BFF_6.pt'
            elif conf.ppi.hotspot_res is not None and conf.scaffoldguided.scaffoldguided is False:
                # use complex trained model
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_finetuned_BFF_9.pt'
            elif conf.scaffoldguided.scaffoldguided is True:
                # use complex and secondary structure-guided model
                self.ckpt_path = '/net/databases/diffusion/models/hotspot_models/base_complex_ss_finetuned_BFF_9.pt' 
            else:
                # use default model
                self.ckpt_path = '/net/databases/diffusion/nate_tmp/new_SelfCond_crdscale0.25/models/BFF_4.pt'
        # for saving in trb file:
        assert self._conf.inference.trb_save_ckpt_path is None, "trb_save_ckpt_path is not the place to specify an input model. Specify in inference.ckpt_override_path"
        self._conf['inference']['trb_save_ckpt_path']=self.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        print(' ')
        print('-'*100)
        print(' ')
        print("WARNING: The following options are not currently implemented at inference. Decide if this matters.")
        print("Delete these in inference/model_runners.py once they are implemented/once you decide they are not required for inference -- JW")
        print(" -predict_previous")
        print(" -prob_self_cond")
        print(" -seqdiff_b0")
        print(" -seqdiff_bT")
        print(" -seqdiff_schedule_type")
        print(" -seqdiff")
        print(" -freeze_track_motif")
        print(" -use_motif_timestep")
        print(" ")
        print("-"*100)
        print(" ")
        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.ab_conf = self._conf.antibody
        self.diffuser = Diffuser(**self._conf.diffuser)
        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            ic('Doing AR Sequence Decoding')
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None

        self.allatom = ComputeAllAtomCoords().to(self.device)
        
        if not self.ab_design():
            if self.inf_conf.input_pdb is None:
                # set default pdb
                script_dir=os.path.dirname(os.path.realpath(__file__))
                self.inf_conf.input_pdb=os.path.join(script_dir, '../benchmark/input/1qys.pdb')
            self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)

        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)
        
    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T

    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self) -> None:
        """
        Function for loading model config from checkpoint directly.

        Takes:
            - config file

        Actions:
            - Replaces all -model and -diffuser items
            - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
        
        This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
        This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.
        """
        # get overrides to re-apply after building the config from the checkpoint
        overrides = []
        if HydraConfig.initialized():
            overrides = list(HydraConfig.get().overrides.task)
            ic(overrides)
        # Added to set default T to 50
        overrides.append(f'diffuser.T={self._conf.diffuser.T}') if not any(i.startswith('diffuser.T=') for i in overrides) else None 

        # Add in the preprocess.use_selfcond_emb flag if it is set as the T_scheme
        if not any(i.startswith('preprocess.use_selfcond_emb=') for i in overrides):
            if self.ckpt['config_dict']['antibody']['T_scheme'] == 'selfcond_emb':
                overrides.append('preprocess.use_selfcond_emb=True')

        if 'config_dict' in self.ckpt.keys():
            print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seq_diffuser','preprocess','antibody']:
                #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
                for key in self._conf[cat]:
                    if key == 'chi_type' and self.ckpt['config_dict'][cat][key] == 'circular':
                        ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                        continue
                    try:
                        print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {self._conf[cat][key]} is correct')
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                    print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""
        
        # Now read input dimensions from checkpoint.
        self.d_t1d        = self._conf.preprocess.d_t1d
        self.d_t2d        = self._conf.preprocess.d_t2d
        self.use_selfcond_emb = self._conf.preprocess.use_selfcond_emb
        model             = RoseTTAFoldModule(**self._conf.model, d_t1d=self.d_t1d, d_t2d=self.d_t2d, use_selfcond_emb=self.use_selfcond_emb, T=self._conf.diffuser.T).to(self.device)
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        if self._conf.inference.final_state:
            model.load_state_dict(self.ckpt['final_state_dict'],strict=True)
        else:
            model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        # TODO: Denoiser seems redundant. Combine with diffuser.
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'seq_diffuser': self.seq_diffuser,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """

        raise NotImplementedError('This function should be implemented in a subclass')

    def _preprocess(self, seq, xyz_t, t, repack=False):
        
        """
        Function to prepare inputs to diffusion model
        
            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)
                - contacting residues: for ppi. Target residues in contact with biner (1)
                - chi_angle timestep (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
    """
        raise NotImplementedError('This function should be implemented in a subclass')

        
    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, return_extra=False):
        '''Generate the next pose that the model should be supplied at timestep t-1.

        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
            
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L,22) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        raise NotImplementedError('This function should be implemented in a subclass')

class AbSampler(Sampler):
    '''
    '''

    def ab_design(self):
        return True

    def sample_init(self):
        '''
        We should do some autodetection of Ab chain in this function. The chain which we are designing is the Ab chain
        We should also enforce the H,L,T chain labelling
        '''

        #### 1) Parse pdb to an ab_pose that can be easily manipulated
        ####################################################################
        self.pose = ab_pose.AbPose()

        # Determine which format the input structure has been provided
        if self.inf_conf.input_pdb is not None:
            assert(self.ab_conf.target_pdb is None and self.ab_conf.framework_pdb is None), \
                    "Both inference.input_pdb and antibody.target + antibody.framework_pdb cannot be active at the same time."

            self.pose.from_HLT(self.inf_conf.input_pdb)

        assert(~((self.ab_conf.target_pdb is None) ^ (self.ab_conf.framework_pdb is None))), \
                "Having antibody.target and not antibody.framework_pdb or vice versa is not currently supported."

        if self.ab_conf.target_pdb is not None and self.ab_conf.framework_pdb is not None:
            assert(self.diffuser_conf.partial_T is None), \
                    "Partial diffusion is only supported when using inference.input_pdb"

            assert(self.inf_conf.input_pdb is None), \
                    "Both inference.input_pdb and antibody.target + antibody.framework_pdb cannot be active at the same time."

            self.pose.framework_from_HLT(self.ab_conf.framework_pdb)
            self.pose.target_from_HLT(self.ab_conf.target_pdb)


        #### 2) Adjust the length of the CDR loops in the AbPose
        ####################################################################################################

        # If we are doing partial diffusion, we will skip the loop length adjustment step
        # Since we are just resampling from the starting scaffold
        if self.diffuser_conf.partial_T:
           print("Partial diffusion detected, skipping loop length adjustment step") 

        else:
            # We are doing full diffusion, so we need to adjust the loop lengths
            ic(self.pose.length())
            ic(self.pose.binder_len())
            ic(self.pose.L.seq)
            self.pose.adjust_loop_lengths(self.ab_conf.design_loops)
            ic(self.pose.length())
            ic(self.pose.binder_len())
            ic(self.pose.L.seq)


        #### 3) Assemble the ab_item for use downstream. Also determine which residues we should design
        ####################################################################################################
        self.ab_item = Dotdict()
        self.ab_item.loop_mask = self.pose.parse_design_mask(self.ab_conf.design_loops)

        assert(torch.any(self.ab_item.loop_mask)), "Received input with no diffused region. Exiting"

        self.L = self.pose.length()
        self.diffusion_mask = torch.ones(self.L).bool() # True is not diffused
        if self.ab_conf.T_scheme == 'fixed_dock':
            # Only diffuse loops in fixed dock case
            self.diffusion_mask[self.ab_item.loop_mask] = False 

        elif self.ab_conf.T_scheme in {'single_T', 'single_T_better_confidence', 'no_T', 'single_T_correct_selfcond', 'noSeq_single_T_correct_selfcond', 'selfcond_emb'}:
            # Diffuse binder region
            self.diffusion_mask[:self.pose.binder_len()] = False 

        else:
            raise NotImplementedError()

        # Needed for downstream processing
        self.ab_item.target = True

        self.ab_item.target_mask = torch.zeros(self.L).bool()
        self.ab_item.target_mask[self.pose.binder_len():] = True

        self.ab_item.interchain_mask = self.pose.get_interchain_mask()

        self.binderlen = self.pose.binder_len()

        #### 4) Parse hotspots
        ##########################################
        self.ab_item.hotspots = self.pose.parse_hotspots(self.ppi_conf.hotspot_res)

        self.ab_item.inputs   = self.pose.to_diffusion_inputs()

        # Assign chain_idx, this is used by run_inference.py to assign chain letters upon writing to disk
        self.chain_idx = self.pose.get_chain_idx()
        self.loop_map  = self.pose.get_loop_map()

        #### 5) Setup Potential Manager, there are not yet any Ab potentials
        ########################################################################
        self.potential_manager = PotentialManager(self.potential_conf,
                                                  self.ppi_conf,
                                                  self.diffuser_conf,
                                                  self.inf_conf,
                                                  self.ab_item.hotspots,
                                                  self.pose.binder_len())

        # These are necessary for compatibility with the parent sample_step function
        self.mask_seq = torch.clone(~self.ab_item.loop_mask)
        #self.mask_seq = torch.clone(self.diffusion_mask)
        self.mask_str = torch.clone(self.diffusion_mask)

        # Determine the timesteps to use for diffusion
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

        t_list = np.arange(1, self.t_step_input+1)

        #### 6) Diffuse the contig mapped regions
        #############################################
        fa_stack, _, _ = self.diffuser.diffuse_pose(
            self.ab_item.inputs.xyz_true,
            self.ab_item.inputs.seq_true,
            self.ab_item.inputs.atom_mask,
            diffusion_mask=self.diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)

        xT = torch.clone(fa_stack[-1].squeeze()[:,:14])

        #### 7) Mask the input sequence of the CDR loops
        ####################################################
        seq_T = nn.one_hot(self.ab_item.inputs.seq_true, num_classes=22).float()
        seq_T[~self.mask_seq,:20] = 0
        seq_T[~self.mask_seq,21]  = 1 # Mask token

        self.denoiser = self.construct_denoiser(self.L, visible=self.diffusion_mask)

        ic(self.ab_item.loop_mask)
        ic(self.ab_item.hotspots)
        return xT, seq_T

    def _preprocess(self, seq, xyz_t, t):
        '''
        xyz_t [T,L,27,3]
        msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t

        '''

        L = xyz_t.shape[0]

        ## 1) Generate the time-dependent features
        ################################################

        tmp_xyz = torch.full((L,27,3), np.nan)
        tmp_xyz[:,:14] = xyz_t # [L,27,3]

        # Featurize expects xyz to have 27 atom dimensions
        features = featurize(
                             self.ab_item,
                             seq,
                             tmp_xyz,
                             self.preprocess_conf.d_t1d,
                             self.preprocess_conf.hotspot_dim,
                             self.ab_conf.T_scheme,
                             1 - (t / self.T),
                             ~self.preprocess_conf.motif_sidechain_input,
                             ~self.ab_conf.no_bugfix_t1d_mask
                            )

        ## 2) Now generate the time-invariant features
        ################################################
        
        ## idx_pdb ##
        #############

        idx_pdb = torch.arange(L) # (L)
        if self.ab_item.target:
            idx_pdb[self.ab_item.target_mask] += 200 # Do idx jump at chainbreak
        
        ## Add hotspots to t1d ##
        #########################
        features['t1d'][...,22] = self.ab_item.hotspots[None,None]
        
        retval = (
                  features['msa_masked'],
                  features['msa_full'],
                  features['seq'],
                  features['xyz_prev'].unsqueeze(0),
                  idx_pdb.unsqueeze(0),
                  features['t1d'].unsqueeze(0),
                  features['t2d'].unsqueeze(0),
                  features['xyz_t'].unsqueeze(0),
                  features['alpha_t'].unsqueeze(0)
                 )

        # Send all inputs to device
        retval = [i.to(self.device) for i in retval]

        return retval

    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):

        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)

        B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Seq Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T) \
            and self.preprocess_conf.selfcondition_msaprev and self.preprocess_conf.msaprev_bugfix:

            ic('Providing Ab Seq Self Cond')
            msa_prev = self.msa_prev

        else:
            msa_prev = None 

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if (t < self.diffuser.T) and (t != self.diffuser_conf.partial_T):

            ic('Providing Ab Str Self Cond')
            xyz_t, t2d, xyz_sc, sc2d = process_selfcond(self.prev_pred, t2d, xyz_t, self.ab_conf, xyz_t.device)

            # Correct selfcond now will detect from the checkpoint file whether it
            # should be run or not
            t2d = correct_selfcond(
                                    t2d,
                                    self.ab_conf,
                                    self.ab_item.inputs.xyz_true,
                                    self.ab_item.loop_mask,
                                    self.ab_item.target_mask,
                                    self.ab_item.interchain_mask
                                  )

        else:
            # The non-selfcond step for antibodies is to just leave the input as-is
            sc2d, xyz_sc = process_init_selfcond(t2d, xyz_t, self.ab_conf, xyz_t.device)
        
        print('Monitoring target centering')
        ic(xyz_t[0,0,self.diffusion_mask,1].mean(dim=0))
        ic(xt_in[0,self.diffusion_mask,1].mean(dim=0))

        with torch.no_grad():
            px0=xt_in
            for rec in range(self.recycle_schedule[t-1]):
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
                                    msa_full,
                                    seq_in,
                                    px0,
                                    idx_pdb,
                                    t1d=t1d,
                                    t2d=t2d,
                                    sc2d=sc2d,
                                    xyz_sc=xyz_sc,
                                    xyz_t=xyz_t,
                                    alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = None,
                                    state_prev = None,
                                    t=torch.tensor(t),
                                    return_infer=True,
                                    motif_mask=self.diffusion_mask.squeeze().to(self.device))   

                # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                if rec < self.recycle_schedule[t-1] -1:
                    zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                    xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
                    t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
                    px0=xt_in

        self.prev_pred = torch.clone(px0)
        self.msa_prev  = torch.clone(msa_prev)

        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0     = px0.squeeze()[:,:14]

        # Default method of decoding sequence
        seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
        sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

        pseq_0 = torch.nn.functional.one_hot(
            sampled_seq, num_classes=22).to(self.device).float()

        pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze()].to(self.device) # [L,22]
        
        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_diffusion_mask=self.mask_seq.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
            self._log.info(
                    f'Timestep {t}, input to next step: { seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}')
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)

        return px0, x_t_1, seq_t_1, tors_t_1, plddt



