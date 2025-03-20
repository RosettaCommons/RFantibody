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

        self.pdb = False
        if not args.pdbdir == '':
            self.pdb = True

            self.pdbdir = args.pdbdir
            self.outpdbdir = args.outpdbdir

            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

            # Parse the runlist and determine which structures to process
            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])

                    # Filter the struct iterator to only include those in the runlist
                    self.struct_iterator = [struct for struct in self.struct_iterator
                                            if os.path.basename(struct).split('.')[0] in self.runlist]

                    print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        self.quiver = False
        if not args.quiver == '':
            self.quiver = True

            self.inquiver = Quiver(args.quiver, mode='r')
            self.outquiver = Quiver(args.outquiver, mode='w')

            self.struct_iterator = self.inquiver.get_tags()

        assert self.pdb ^ self.quiver, 'Either pdb or quiver must be set to True'

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag: str) -> None:
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
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
        Dump this pose to either a pdb file, or quiver file depending on the input arguments
        '''
        if self.pdb:
            # If the outpdbdir does not exist, create it
            # If there are parents in the path that do not exist, create them as well
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)

            pdbfile = os.path.join(self.outpdbdir, tag + '.pdb')
            pose.dump_pdb(pdbfile)
        
        if self.quiver:
            pdblines = pose.to_pdblines()
            self.outquiver.add_pdb(pdblines, tag)

    def load_pose(self, tag: str) -> Pose:
        '''
        Load a pose from either a silent file, pdb file, or quiver file depending on the input arguments
        '''

        if not self.pdb and not self.silent and not self.quiver:
            raise Exception('Neither pdb nor silent nor quiver is set to True. Cannot load pose')

        if self.pdb:
            pose = Pose.from_pdb(tag)
        
        if self.quiver:
            pose = Pose.from_pdblines(self.inquiver.get_pdb(tag))

        return pose