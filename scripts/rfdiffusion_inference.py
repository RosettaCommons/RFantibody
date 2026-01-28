"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import glob
import logging
import os
import pickle
import random
import re
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from rfantibody.rfdiffusion.chemical import num2aa
from rfantibody.rfdiffusion.inference import model_runners
from rfantibody.rfdiffusion.util import generate_Cbeta, writepdb, writepdb_multi
from rfantibody.util.io import ab_write_pdblines
from rfantibody.util.quiver import Quiver

conversion = 'ARNDCQEGHILKMFPSTWYV-'

def make_deterministic(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()
    
    # Initialize sampler and target/contig.
    sampler = model_runners.AbSampler(conf)
    
    # Loop over number of designs to sample.
    design_startnum = sampler.inf_conf.design_startnum

    #### Set up Quiver files
    ############################################

    if sampler.inf_conf.quiver is not None:
        # Load the Quiver file
        quiver = Quiver(sampler.inf_conf.quiver, mode='w')

        # Make trajectory Quiver files
        if sampler.inf_conf.write_trajectory:
            # Get Quiver base filename without suffix
            quiver_base = os.path.splitext(sampler.inf_conf.quiver)[0]

            quiver_xt   = Quiver(quiver_base+'_Xt-1_traj.qv', mode='w')
            quiver_px0  = Quiver(quiver_base+'_pX0_traj.qv', mode='w')

    #### Determine the starting design number
    ############################################

    if sampler.inf_conf.quiver is not None:

        tags = quiver.get_tags()

        if sampler.inf_conf.design_startnum == -1:

            # If we are using Quiver files then we need to find the next available design number
            # We will do this by looking at the existing designs and finding the highest number

            tags = quiver.get_tags()

            if len(tags) == 0:
                # No designs yet, start at 0
                design_startnum = 0
            else:
                # Find the highest design number
                indices = [-1]
                for tag in tags:
                    m = re.match('.*_(\d+)$', tag)
                    if not m:
                        continue
                    m = m.groups()[0]
                    indices.append(int(m))
                design_startnum = max(indices) + 1

    elif sampler.inf_conf.design_startnum == -1:
        existing = glob.glob(sampler.inf_conf.output_prefix + '*.pdb')
        indices = [-1]
        for e in existing:
            print(e)
            m = re.match('.*_(\d+)\.pdb$', e)
            print(m)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1

    #### Run the main design loop
    ############################################

    for i_des in range(design_startnum, design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f'{sampler.inf_conf.output_prefix}_{i_des}'
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
            log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
            continue

        if sampler.inf_conf.quiver is not None:
            if out_prefix in tags:
                log.info(f'Skipping this design because tag {out_prefix} already exists.')
                continue

        # to track hotspots through inference
        failed=0
        while True:

            x_init, seq_init = sampler.sample_init()
            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            chi1_stack = []
            plddt_stack = []

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            # Loop over number of reverse diffusion time steps.
            for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
                px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                    t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init, final_step=sampler.inf_conf.final_step)
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                chi1_stack.append(tors_t[:,:])
                plddt_stack.append(plddt[0]) # remove singleton leading dimension
                
                print("Sequence of Hotspot Residues:","".join(conversion[i] for i in torch.argmax(seq_t, dim=1)[sampler.ab_item.hotspots]))
                if conf.antibody.terminate_bad_targeting is not None:
                    #TODO move to a separate function to avoid repetition
                    # Loop through the hotspots, find the closest loop residue by Cb distance
                    # And then average the distance over each of the hotspots. Report min and mean distance
                    Cb = generate_Cbeta(N=px0[:,0], Ca=px0[:,1], C=px0[:,2])
        
                    # We are going to assume constructed Cb is identical to original Cb - NRB
                    dist = torch.cdist(Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask]) # [hotspot_L, loop_L]

                    mindist = torch.min(dist, dim=1).values # The min distance for each hotspot
    
                    overallmin = torch.min(mindist) # The distance of the closest hotspot to a loop

                    print(f'Overall min distance hotspot to designed loop: {overallmin}')
                    if conf.antibody.terminate_bad_targeting == t and overallmin > conf.antibody.hotspot_termination_threshold:
                        print("Not targeting correctly")
                        failed+=1
                        if failed>=conf.antibody.hotspot_termination_failures_permitted:
                            sys.exit("This set of inputs is not efficiently targeting the hotspots")
                        continue
            # break out of while loop
            break
        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        # Save outputs 
        if sampler.inf_conf.quiver is None:
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

        final_seq = seq_stack[-1]

        if conf.seq_diffuser.seqdiff is not None:
            # When doing sequence diffusion the model does not make predictions beyond category 19
            final_seq = final_seq[:,:20] # [L,20]

        # All samplers now use a one-hot seq so they all need this step
        #final_seq = torch.argmax(final_seq, dim=-1)
        # Now just output glycines, except for motif region
        final_seq=torch.where(torch.argmax(seq_init, dim=-1)==21, 7, torch.argmax(seq_init, dim=-1)) #7 is glycine
        
        bfacts = torch.ones_like(final_seq.squeeze())
        # make bfact=0 for diffused coordinates
        bfacts[torch.where(torch.argmax(seq_init, dim=-1)==21,True,False)]=0
        # pX0 last step
        out = f'{out_prefix}.pdb'

        #### Write TRB file
        ####################################
        trb = dict(
            config = OmegaConf.to_container(sampler._conf, resolve=True),
            plddt = plddt_stack.cpu().numpy(),
            device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            time = time.time() - start_time
        )

        if sampler.ab_design(): 
            # Write out the sampled individual loop lengths
            for loop in sampler.loop_map:
                # Sum the values of each boolean loop map
                trb[f"{loop.upper()}_len"] = len(sampler.loop_map[loop])

        if sampler.ab_design() and torch.any(sampler.ab_item.target_mask) and torch.any(sampler.ab_item.hotspots):
            # Loop through the hotspots, find the closest loop residue by Cb distance
            # And then average the distance over each of the hotspots. Report min and mean distance
            Cb = generate_Cbeta(N=denoised_xyz_stack[0,:,0], Ca=denoised_xyz_stack[0,:,1], C=denoised_xyz_stack[0,:,2])

            # We are going to assume constructed Cb is identical to original Cb - NRB
            dist = torch.cdist(Cb[sampler.ab_item.hotspots], Cb[sampler.ab_item.loop_mask]) # [hotspot_L, loop_L]

            mindist = torch.min(dist, dim=1).values # The min distance for each hotspot
            
            overallmin = torch.min(mindist) # The distance of the closest hotspot to a loop
            averagemin = torch.mean(mindist) # The average distance of the hotspots to a loop

            print(f'Overall min distance hotspot to designed loop: {overallmin}')
            print(f'Average min distance hotspot to designed loop: {averagemin}')

            trb['mindist']    = overallmin.cpu().numpy()
            trb['averagemin'] = averagemin.cpu().numpy()

        if hasattr(sampler, 'contig_map'):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value

        if sampler.inf_conf.quiver is None:
            # If we are not using Quiver files then we will write out the TRB file
            with open(f'{out_prefix}.trb','wb') as f_out:
                pickle.dump(trb, f_out)

        #### Write out structure
        ####################################
        if sampler.ab_design():
            # Also mark hotspots in ab design
            bfacts[sampler.ab_item.hotspots] = 0

            if sampler.inf_conf.quiver is None:
                # Write as PDB files
                pdblines = ab_write_pdblines(
                    atoms = denoised_xyz_stack[0,:,:4].cpu().numpy(),
                    seq = final_seq.cpu().numpy(),
                    chain_idx = sampler.chain_idx,
                    bfacts = bfacts.cpu().numpy(),
                    loop_map = sampler.loop_map,
                    num2aa = num2aa,
                )

                with open(out, 'w') as f_out:
                    f_out.write('\n'.join(pdblines))

            else:
                # Add to Quiver file
                pdblines = ab_write_pdblines(
                    atoms = denoised_xyz_stack[0,:,:4].cpu().numpy(),
                    seq = final_seq.cpu().numpy(),
                    chain_idx = sampler.chain_idx,
                    bfacts = bfacts.cpu().numpy(),
                    loop_map = sampler.loop_map,
                    num2aa = num2aa,
                ) 

                # Remove backslashes from out_prefix
                outtag = out_prefix.replace('/', '_')

                if torch.any(sampler.ab_item.target_mask) and torch.any(sampler.ab_item.hotspots):
                    scoreline = f'mindist={float(overallmin):.2f}|averagemin={float(averagemin):.2f}'
                    quiver.add_pdb(pdblines, outtag, scoreline)
                else:
                    quiver.add_pdb(pdblines, outtag)
        else:
            # Now don't output sidechains
            writepdb(out, denoised_xyz_stack[0,:,:4], final_seq, sampler.binderlen, chain_idx=sampler.chain_idx, bfacts=bfacts)

        #### Write Trajectory
        ####################################
        if sampler.inf_conf.write_trajectory:

            if sampler.inf_conf.quiver is not None:
                # Add to Quiver file
                xtpdblines = writepdb_multi(None, denoised_xyz_stack, bfacts,
                    final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx, return_pdblines=True)
                px0pdblines = writepdb_multi(None, px0_xyz_stack, bfacts,
                    final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx, return_pdblines=True)

                quiver_xt.add_pdb(xtpdblines, f'{outtag}_Xt-1')
                quiver_px0.add_pdb(px0pdblines, f'{outtag}_pX0')

            else:
                # trajectory pdbs
                traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

                out_xt  = f'{traj_prefix}_Xt-1_traj.pdb'
                out_px0 =f'{traj_prefix}_pX0_traj.pdb'

                writepdb_multi(out_xt, denoised_xyz_stack, bfacts, 
                    final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)

                writepdb_multi(out_px0, px0_xyz_stack, bfacts, 
                    final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)

        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()
