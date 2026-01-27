import glob
import os
import uuid

from rfantibody.util.pose import Pose
from rfantibody.util.quiver import Quiver


class StructManager():
    '''
    This class handles all of the input and output for the ProteinMPNN model. It deals with quiver files vs. pdbs,
    checkpointing, and writing of outputs
    '''

    def __init__(self, args) -> None:
        self.args = args

        # Track input and output formats separately
        self.input_pdb = False
        self.input_quiver = False
        self.output_pdb = False
        self.output_quiver = False

        # Setup input from PDB directory
        if args.pdbdir != '':
            self.input_pdb = True
            self.pdbdir = args.pdbdir
            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

            # Parse the runlist and determine which structures to process
            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])

                    # Filter the struct iterator to only include those in the runlist
                    self.struct_iterator = [struct for struct in self.struct_iterator
                                            if os.path.basename(struct).split('.')[0] in self.runlist]

                    print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        # Setup input from quiver file
        if args.quiver != '':
            self.input_quiver = True
            self.inquiver = Quiver(args.quiver, mode='r')
            self.struct_iterator = self.inquiver.get_tags()

        # Setup output - quiver takes precedence over pdb
        if args.outquiver != '':
            self.output_quiver = True
            self.outquiver = Quiver(args.outquiver, mode='w')
        else:
            self.output_pdb = True
            self.outpdbdir = args.outpdbdir

        assert self.input_pdb ^ self.input_quiver, 'Exactly one input source (pdbdir or quiver) must be specified'

        # Setup checkpointing - determine finished structures based on output type
        self.finished_structs = set()

        if self.output_quiver:
            # When using quiver output, check existing tags in the output quiver
            # Output tags follow pattern: {input_tag}_dldesign_{idx}
            # So we extract the base input tag from existing output tags
            self.chkfn = None  # No checkpoint file needed for quiver output
            existing_tags = self.outquiver.get_tags()
            for tag in existing_tags:
                # Extract input tag from output tag (remove _dldesign_N suffix)
                if '_dldesign_' in tag:
                    input_tag = tag.rsplit('_dldesign_', 1)[0]
                    self.finished_structs.add(input_tag)
        else:
            # For PDB output, use checkpoint file next to the output directory
            if args.checkpoint_name != 'check.point':
                # User specified a custom checkpoint name, use it as-is
                self.chkfn = args.checkpoint_name
            else:
                # Default: put checkpoint file next to output directory
                self.chkfn = os.path.join(self.outpdbdir, 'check.point')
                # Ensure output directory exists for checkpoint file
                if not os.path.exists(self.outpdbdir):
                    os.makedirs(self.outpdbdir)

            if os.path.isfile(self.chkfn):
                with open(self.chkfn, 'r') as f:
                    for line in f:
                        self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag: str) -> None:
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs.
        For quiver output, this is a no-op since the quiver file itself tracks what's been written.
        '''
        if self.chkfn is None:
            # Using quiver output - no checkpoint file needed
            return
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self) -> str:
        '''
        Iterate over the silent file or pdb directory and run the model on each structure
        '''

        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            tag = os.path.basename(struct).split('.')[0]
            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct

    def dump_pose(
        self,
        pose: Pose,
        tag: str,
    ) -> None:
        '''
        Dump this pose to either a pdb file, or quiver file depending on the output arguments
        '''
        if self.output_pdb:
            # If the outpdbdir does not exist, create it
            # If there are parents in the path that do not exist, create them as well
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)

            pdbfile = os.path.join(self.outpdbdir, tag + '.pdb')
            pose.dump_pdb(pdbfile)

        if self.output_quiver:
            pdblines = pose.to_pdblines()
            self.outquiver.add_pdb(pdblines, tag)

    def load_pose(self, tag: str) -> Pose:
        '''
        Load a pose from either a pdb file or quiver file depending on the input arguments
        '''

        if self.input_pdb:
            pose = Pose.from_pdb(tag)
        elif self.input_quiver:
            pose = Pose.from_pdblines(self.inquiver.get_pdblines(tag))
        else:
            raise Exception('Neither input_pdb nor input_quiver is set to True. Cannot load pose')

        return pose