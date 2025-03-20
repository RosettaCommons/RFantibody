from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

import rfantibody.rf2.modules.pose_util as pu
import rfantibody.rf2.modules.rmsd as rmsd
from rfantibody.rf2.network.predict import Predictor, pae_unbin


class AbPredictor(Predictor):
    """
    Subclass of RF2 Predictor class, to predict Ab structures
    """
    def __init__(self, conf: HydraConfig, preprocess_fn: Preprocess, device='cuda:0'):
        """
        Initialise from config
        """
        self.conf=conf
        self.preprocess_fn=preprocess_fn
        self.device=device
        #self._update_params_from_checkpoint()        
        super().__init__(self.conf.model.model_weights, self.device, model_param=self.conf.model_param)
        self.return_rmsds=any([var is not None for var in [conf.input.pdb, conf.input.pdb_dir, conf.input.quiver]])
        self.xyz_converter.to(self.device)

    def __call__(self, pose: Pose, tag: str) -> None:
        """
        Runs prediction on a yielded pose
        """
        (    
                network_input,
                xyz_prev,
                mask_recycle,
                xyz_true,
                mask_true,
                msa, 
                mask_msa,
                _,   
                _,   
                idx, 
            ) = self.preprocess_fn(pose, device=self.device)
        
        network_input = self._add_batch_dim(network_input)
        outputs = [
            "msa_prev",
            "pair_prev",
            "state_prev",
            "xyz_prev",
            "alpha",
            "mask_recycle",
        ]    
        output_i = (None, None, None, xyz_prev[None], None, mask_recycle)
        output_i = {outputs[i]: val for i, val in enumerate(output_i)}
        msa_prev = pair_prev = None
        mask_prev = network_input["mask_t"][0, :1].clone()
        best_lddt = torch.tensor([-1.0], device=self.device)

        to_write=OrderedDict()
        with torch.no_grad():
            for i_cycle in range(self.conf.inference.num_recycles + 1):
                output_i={i:val for i, val in output_i.items() if i in outputs}
                input_i={**network_input, **output_i}
                
                # TODO clean this up
                input_i.pop('alpha',None)
                input_i['xyz_t'] = input_i['xyz_t'].clone()[...,1,:]
                
                output_i=self._output_dictionary(self.model(**input_i), input_i)
                output_pose_i = pu.pose_from_RF_output(output_i, pose)
                metrics_i=self._process_output(output_i, output_pose_i, pose)
                
                if metrics_i['pred_lddt'].mean() > best_lddt.mean():
                    best_lddt = metrics_i['pred_lddt']
                    best_pose = copy.deepcopy(output_pose_i)
                    best_metrics = copy.deepcopy(metrics_i)
                    to_write['best'] = {'pose': best_pose, 'metrics': best_metrics}
                if self.conf.output.output_intermediates:
                    to_write[i_cycle] = {'pose': output_pose_i, 'metrics': metrics_i}
                torch.cuda.empty_cache()
        write_output(to_write, tag, self.conf)

    def _update_params_from_checkpoint(self) -> None:
        """
        Loads model params from checkpoint
        """
        checkpoint=torch.load(self.conf.model.model_weights, map_location=self.device)
        training_config=checkpoint['config']
        self.conf = OmegaConf.merge(self.conf, training_config)

    def _add_batch_dim(self, network_input: dict) -> dict:
        """
        Required to add the batch (1) dimension to network inputs
        """
        return {key: value[None] for key, value in network_input.items()}

    def _output_dictionary(self, output_i: tuple, input_i: dict) -> dict:
        """
        Converts RF2 output to dictionary
        """
        output_tensors = [
            "logit_s",
            "logit_aa_s",
            "null1",
            "logits_pae",
            "p_bind",
            "xyz_prev",
            "alpha",
            "null2",
            "pred_lddt",
            "msa_prev",
            "pair_prev",
            "state_prev",
        ]
        output_i={output_tensors[i]: val for i, val in enumerate(output_i)}
        output_i['mask_recycle']=None
        _, output_i["xyz_prev"] = self.xyz_converter.compute_all_atom(
                        input_i["seq"], output_i["xyz_prev"][-1], output_i["alpha"][-1]
                    )
        return output_i

    def _process_output(self, output_i: dict, output_pose_i: Pose, input_pose: Pose) -> OrderedDict:
        """
        Gets confidence and rmsds
        """
        metrics=OrderedDict()
        self.get_confidence_scores(output_i, input_pose, metrics)
        if self.return_rmsds:
            get_rmsds(output_pose_i, input_pose, metrics)
        return metrics

    def get_confidence_scores(self, output_i: dict, pose: Pose, metrics: dict) -> None:
        """
        Gets confidence scores from RF2 output.
        Appends to metrics
        """
        device = self.lddt_bins.device

        pred_lddt = output_i["pred_lddt"].to(device)
        logits_pae = output_i["logits_pae"].to(device)

        pred_lddt = (
            nn.Softmax(dim=1)(pred_lddt)
            * self.lddt_bins[None, :, None]
        )
        pred_lddt = pred_lddt.sum(dim=1)
        pae = pae_unbin(logits_pae)
        interaction_pae=pae[0, ~pose.same_chain].mean()
        metrics["interaction_pae"] = interaction_pae
        metrics["pae"] = pae
        metrics["pred_lddt"] = pred_lddt

def get_rmsds(pose1: Pose, pose2: Pose, metrics: dict) -> None:
    """
    Calculates RMSDs between two poses.
    First, does target-based alignment of pose 1 to pose 2
    Then calculates:
        Ca rmsd of whole antibody (without further alignment)
        Ca rmsd of CDRs (without further alignment)
        Ca rmsd of antibody monomer (aligned on framework)
        Ca rmsd of CDRs (aligned on framework)
        Ca rmsd of each CDR (aligned on framework)
    """
    # target-aligned rmsds
    rmsd.tmalign_to_subset(pose1, pose2, subset='target')
    metrics['target_aligned_antibody_rmsd'] = rmsd.calc_prealigned_rmsd(pose1, pose2, pose1.antibody_mask)
    metrics['target_aligned_cdr_rmsd'] = rmsd.calc_prealigned_rmsd(pose1, pose2, pose1.cdrs.mask_1d)

    # framework-aligned monomer rmsds
    rmsd.tmalign_to_subset(pose1, pose2, subset='framework')
    metrics['framework_aligned_antibody_rmsd'] = rmsd.calc_prealigned_rmsd(pose1, pose2, pose1.antibody_mask)
    metrics['framework_aligned_cdr_rmsd'] = rmsd.calc_prealigned_rmsd(pose1, pose2, pose1.cdrs.mask_1d)

    # individual loop rmsds
    # pose1 is already framework-aligned
    for loop in pose1.cdrs.cdr_names():
        metrics[f'framework_aligned_{loop}_rmsd'] = rmsd.calc_prealigned_rmsd(pose1, pose2, getattr(pose1.cdrs, f'{loop}'))
    
def write_output(to_write: OrderedDict, tag: str, conf: HydraConfig) -> None:
    """
    Writes output to file. Either pdb or quiver file
    """
    if sum([var is not None for var in [conf.output.pdb_dir, conf.output.quiver]]) != 1:
        raise ValueError('Exactly one of output.pdb_dir or output.quiver must be specified')
    qv=conf.output.quiver is not None
    for key, val in to_write.items():
        if key == 'best':
            suffix = 'best'
        else:
            suffix = f'cycle_{key}'
        pose = val['pose']
        metrics = val['metrics']
        pdblines=pu.pose_to_remarked_pdblines(pose, metrics=metrics)
        if qv:
            output_qv=Quiver(f'{conf.output.quiver}.qv', mode='w')
            pdblines=[f'QV_{line}' if line.startswith('SCORE') else line for line in pdblines]
            output_quiver.add_pdb(pdblines, tag=f'{tag}_{suffix}')
        else:
            pu.pdblines_to_pdb(pdblines, f'{conf.output.pdb_dir}/{tag}_{suffix}.pdb')
