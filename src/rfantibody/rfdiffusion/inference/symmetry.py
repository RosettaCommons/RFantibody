"""Helper class for handle symmetric assemblies."""
import functools as fn
import logging
import string

import numpy as np
import torch
from pyrsistent import v
from scipy.spatial.transform import Rotation

format_rots = lambda r: torch.tensor(r).float()

T3_ROTATIONS = [
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [-1., -0.,  0.],
        [-0.,  1.,  0.],
        [-0.,  0., -1.]]).float(),
    torch.Tensor([
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0., -1.]]).float(),
]

saved_symmetries = ['tetrahedral', 'octahedral', 'icosahedral']

class SymGen:

    def __init__(self, global_sym, recenter, radius, model_only_neibhbors=False):
        self._log = logging.getLogger(__name__)
        self._recenter = recenter
        self._radius = radius

        if global_sym.lower().startswith('c'):
            # Cyclic symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing cyclic symmetry order {global_sym[1:]}.')
            self._init_cyclic(int(global_sym[1:]))
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower().startswith('mc'):
            # 2-component cyclic symmetry
            order = global_sym[2:]
            if not order.isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing 2-component cyclic symmetry order {order}.')
            self._init_multi_cyclic(int(order), 2)
            self.apply_symmetry = self._apply_multi_cyclic

        elif global_sym.lower().startswith('d'):
            # Dihedral symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid dihedral symmetry {global_sym}')
            self._log.info(
                f'Initializing dihedral symmetry order {global_sym[1:]}.')
            self._init_dihedral(int(global_sym[1:]))
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower() == 't3':
            # Tetrahedral (T3) symmetry
            self._log.info('Initializing T3 symmetry order.')
            self.sym_rots = T3_ROTATIONS
            self.order = 4
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym == 'octahedral':
            # Octahedral symmetry
            self._log.info(
                'Initializing octahedral symmetry.')
            self._init_octahedral()
            self.apply_symmetry = self._apply_octahedral

        elif global_sym.lower() in saved_symmetries:
            # Using a saved symmetry 
            self._log.info('Initializing %s symmetry order.'%global_sym)
            self._init_from_symrots_file(global_sym)

            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic
        else:
            raise ValueError(f'Unrecognized symmetry {global_sym}')

        # self.model_only_neibhbors = model_only_neibhbors
        # if self.model_only_neibhbors:
        #     self.close_rots = self.close_neighbors()
        # self.num_subunits = len(self.close_rots) if self.model_only_neibhbors else self.order
        self.res_idx_procesing = fn.partial(
            self._lin_chainbreaks, num_breaks=self.order)

    #####################
    ## Cyclic symmetry ##
    #####################
    def _init_cyclic(self, order):
        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots
        self.order = order

    def _apply_cyclic(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            coords_out[start_i:end_i] = torch.einsum(
                'bnj,kj->bnk', coords_out[:subunit_len], self.sym_rots[i])
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    def _lin_chainbreaks(self, num_breaks, res_idx, offset=None):
        assert res_idx.ndim == 2
        res_idx = torch.clone(res_idx)
        subunit_len = res_idx.shape[-1] // num_breaks
        chain_delimiters = []
        if offset is None:
            offset = res_idx.shape[-1]
        for i in range(num_breaks):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            chain_labels = list(string.ascii_uppercase) + [str(i+j) for i in
                    string.ascii_uppercase for j in string.ascii_uppercase]
            chain_delimiters.extend(
                [chain_labels[i] for _ in range(subunit_len)]
            )
            res_idx[:, start_i:end_i] = res_idx[:, start_i:end_i] + offset * (i+1)
        return res_idx, chain_delimiters

    #####################################
    ## Multi-component cyclic symmetry ##
    #####################################
    def _init_multi_cyclic(self, order, num_subunits):
        subunit_rots = []
        subunit_order = order // num_subunits
        for i in range(subunit_order):
            deg = i * 360.0 / subunit_order
            r = Rotation.from_euler('z', deg, degrees=True)
            subunit_rots.append(format_rots(r.as_matrix()))
        self.subunit_rots = subunit_rots

        inter_rots = []
        for i in range(num_subunits):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            inter_rots.append(format_rots(r.as_matrix()))
        self.inter_rots = inter_rots

        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots

        self.order = order
        self.num_subunits = num_subunits
    
    def _apply_multi_cyclic(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisible by {self.order}')

        subunit_len = seq_out.shape[0] // self.order
        rot_indices = []
        for i in range(len(self.subunit_rots)):
            for _ in range(self.num_subunits):
                rot_indices.append(i)
        
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i, rot in enumerate(range(self.order)):
            subunit_idx = i % self.num_subunits
            inter_rot = self.inter_rots[subunit_idx]
            rot = self.subunit_rots[rot_indices[i]]

            subunit_chain = coords_in[
                (subunit_len * subunit_idx):(subunit_len * (subunit_idx+1))]
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', subunit_chain, inter_rot)
            subunit_seq = seq_out[
                (subunit_len * subunit_idx):(subunit_len * (subunit_idx+1))]
            
            
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', subunit_chain, rot)

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = subunit_seq

        return coords_out, seq_out

    #######################
    ## Dihedral symmetry ##
    #######################
    def _init_dihedral(self, order):
        sym_rots = []
        flip = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        for i in range(order):
            deg = i * 360.0 / order
            rot = Rotation.from_euler('z', deg, degrees=True).as_matrix()
            sym_rots.append(format_rots(rot))
            rot2 = flip @ rot
            sym_rots.append(format_rots(rot2))
        self.sym_rots = sym_rots
        self.order = order * 2

    #########################
    ## Octahedral symmetry ##
    #########################
    def _init_octahedral(self):
        sym_rots = np.load("./inference/sym_rots.npz")
        self.sym_rots = [
            torch.tensor(v_i, dtype=torch.float32)
            for v_i in sym_rots['octahedral']
        ]
        self.order = len(self.sym_rots)

    def _apply_octahedral(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', coords_in[:subunit_len], self.sym_rots[i])

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    #######################
    ## symmetry from file #
    #######################
    def _init_from_symrots_file(self, name):
        """ _init_from_symrots_file initializes using 
        ./inference/sym_rots.npz

        Args:
            name: name of symmetry (of tetrahedral, octahedral, icosahedral)

        sets self.sym_rots to be a list of torch.tensor of shape [3, 3]
        """
        assert name in saved_symmetries, name + " not in " + str(saved_symmetries)

        # Load in list of rotation matrices for `name`
        fn = "./inference/sym_rots.npz"
        obj = np.load(fn)
        symms = None
        for k, v in obj.items():
            if str(k) == name: symms = v
        assert symms is not None, "%s not found in %s"%(name, fn)

        
        self.sym_rots =  [torch.tensor(v_i, dtype=torch.float32) for v_i in symms]
        self.order = len(self.sym_rots)

        # Return if identity is the first rotation  
        if not np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0):

            # Move identity to be the first rotation
            for i, rot in enumerate(self.sym_rots):
                if np.isclose(((rot-np.eye(3))**2).sum(), 0):
                    self.sym_rots = [self.sym_rots.pop(i)]  + self.sym_rots

            assert len(self.sym_rots) == self.order
            assert np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0)

    def close_neighbors(self):
        """close_neighbors finds the rotations within self.sym_rots that
        correspond to close neighbors.

        Returns:
            list of rotation matrices corresponding to the identity and close neighbors
        """
        # set of small rotation angle rotations
        rel_rot = lambda M: np.linalg.norm(Rotation.from_matrix(M).as_rotvec())
        rel_rots = [(i+1, rel_rot(M)) for i, M in enumerate(self.sym_rots[1:])]
        min_rot = min(rel_rot_val[1] for rel_rot_val in rel_rots)
        close_rots = [np.eye(3)] + [
                self.sym_rots[i] for i, rel_rot_val in rel_rots if
                np.isclose(rel_rot_val, min_rot)
                ]
        return close_rots
