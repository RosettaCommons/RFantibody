import argparse
import os
import sys
import time

import torch

import rfantibody.proteinmpnn.util_protein_mpnn as mpnn_util
from rfantibody.proteinmpnn.sample_features import SampleFeatures
from rfantibody.proteinmpnn.struct_manager import StructManager

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument("-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model')
parser.add_argument("-quiver", type=str, default="", help='The name of a quiver file to run this metric on.')

parser.add_argument("-outquiver", type=str, default="out.qv",
                    help="The name of the quiver file to which output structs will be written")
parser.add_argument("-outpdbdir", type=str, default="outputs",
                    help='The directory to which the output PDB files will be written')
parser.add_argument("-runlist", type=str, default='',
                    help="The path of a list of pdb tags to run (default: ''; Run all PDBs")
parser.add_argument("-checkpoint_name", type=str, default='check.point',
                    help="The name of a file where tags which have finished will be written (default: check.point)")
parser.add_argument("-debug", action="store_true", default=False,
                    help='When active, errors will cause the script to crash and the error message ' + \
                         'to be printed out (default: False)')

# Design Arguments
parser.add_argument("-loop_string", type=str, default='H1,H2,H3,L1,L2,L3',
                    help='The list of loops which you wish to design')
parser.add_argument("-seqs_per_struct", type=int, default="1",
                    help="The number of sequences to generate for each structure (default: 1)")

# ProteinMPNN Specific Arguments
default_ckpt = os.path.join( os.path.dirname(__file__), '/home/weights/ProteinMPNN_v48_noise_0.2.pt')
parser.add_argument("-checkpoint_path", type=str, default=default_ckpt)
parser.add_argument("-temperature", type=float, default=0.000001, help='An a3m file containing the MSA of your target')
parser.add_argument("-augment_eps", type=float, default=0,
                    help='The variance of random noise to add to the atomic coordinates (default 0)')
parser.add_argument("-protein_features", type=str, default='full',
                    help='What type of protein features to input to ProteinMPNN (default: full)')
parser.add_argument("-omit_AAs", type=str, default='CX',
                    help='A string of all residue types (one letter case-insensitive) that you would not like to ' + \
                         'use for design. Letters not corresponding to residue types will be ignored')
parser.add_argument("-num_connections", type=int, default=48,
                    help='Number of neighbors each residue is connected to, default 48, higher number leads to ' + \
                         'better interface design but will cost more to run the model.')

args = parser.parse_args(sys.argv[1:])

class ProteinMPNN_runner():
    '''
    This class is designed to run the ProteinMPNN model on a single input. This class handles the loading of the model,
    the loading of the input data, the running of the model, and the processing of the output
    '''

    def __init__(self, args, struct_manager):
        self.struct_manager = struct_manager

        if torch.cuda.is_available():
            print('Found GPU will run ProteinMPNN on GPU')
            self.device = "cuda:0"
        else:
            print('No GPU found, running ProteinMPNN on CPU')
            self.device = "cpu"

        self.mpnn_model = mpnn_util.init_seq_optimize_model(
            self.device,
            hidden_dim=128,
            num_layers = 3,
            backbone_noise = args.augment_eps,
            num_connections = args.num_connections,
            checkpoint_path = args.checkpoint_path
        )

        self.temperature = args.temperature
        self.seqs_per_struct = args.seqs_per_struct
        self.omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list("ARNDCQEGHILKMFPSTWYVX") ]

    def sequence_optimize(self, sample_feats: SampleFeatures) -> list[tuple[str, float]]:
        t0 = time.time()

        # Once we have figured out pose I/O without Rosetta this will be easy to swap in
        pdbfile = 'temp.pdb'
        sample_feats.pose.dump_pdb(pdbfile)

        feature_dict = mpnn_util.generate_seqopt_features(pdbfile, sample_feats.chains)

        os.remove(pdbfile)

        arg_dict = mpnn_util.set_default_args(self.seqs_per_struct , omit_AAs=self.omit_AAs)
        arg_dict['temperature'] = self.temperature

        masked_chains = sample_feats.chains[:-1]
        visible_chains = [sample_feats.chains[-1]]

        fixed_positions_dict = {pdbfile[:-len('.pdb')]: sample_feats.fixed_res}

        sequences = mpnn_util.generate_sequences(
            self.mpnn_model,
            self.device,
            feature_dict,
            arg_dict,
            masked_chains,
            visible_chains,
            fixed_positions_dict=fixed_positions_dict
        )
        
        print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" ) 

        print(f'sequence_optimize: {sequences}')

        return sequences

    def proteinmpnn(self, sample_feats: SampleFeatures) -> None:
        '''
        Run MPNN sequence optimization on the pose
        '''
        seqs_scores = self.sequence_optimize(sample_feats)

        # Iterate though each seq score pair and thread the sequence onto the pose
        # Then write each pose to a pdb file
        prefix = f"{sample_feats.tag}_dldesign"
        for idx, (seq, _) in enumerate(seqs_scores): 
            sample_feats.thread_mpnn_seq(seq)

            outtag = f"{prefix}_{idx}"

            self.struct_manager.dump_pose(sample_feats.pose, outtag)

    def run_model(self, tag, args):
        t0 = time.time()

        print(f"Attempting pose: {tag}")
        
        # Load the pose 
        pose = self.struct_manager.load_pose(tag)

        # Initialize the features
        sample_feats = SampleFeatures(pose, tag)

        # Parse the loop string and determine which residues should be designed
        sample_feats.loop_string2fixed_res(args.loop_string)

        self.proteinmpnn(sample_feats)

        seconds = int(time.time() - t0)

        print(f"Struct: {pdb} reported success in {seconds} seconds")


####################
####### Main #######
####################

struct_manager = StructManager(args)
proteinmpnn_runner = ProteinMPNN_runner(args, struct_manager)

for pdb in struct_manager.iterate():

    if args.debug: proteinmpnn_runner.run_model(pdb, args)

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = time.time()

        try: proteinmpnn_runner.run_model(pdb, args)

        except KeyboardInterrupt: sys.exit("Script killed by Control+C, exiting")

        except:
            seconds = int(time.time() - t0)
            print(f"Struct with tag {pdb} failed in {seconds} seconds with error: {sys.exc_info()[0]}")

    # We are done with one pdb, record that we finished
    struct_manager.record_checkpoint(pdb)
    



