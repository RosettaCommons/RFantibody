import copy

import numpy as np
import torch

from rfantibody.proteinmpnn.model.protein_mpnn_utils import (
    ProteinMPNN,
    _S_to_seq,
    _scores,
    tied_featurize,
)

#################################
# Function Definitions
#################################

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

# PDB Parse Util Functions

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def AA_to_N(x):
  # ["ARND"] -> [[0,1,2,3]]
  x = np.array(x);
  if x.ndim == 0: x = x[None]
  return [[aa_1_N.get(a, states-1) for a in y] for y in x]

def N_to_AA(x):
  # [[0,1,2,3]] -> ["ARND"]
  x = np.array(x);
  if x.ndim == 1: x = x[None]
  return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

# End PDB Parse Util Functions

def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''
  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha():
            resa,resn = resn[-1],int(resn[:-1])-1
        else:
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn:
            min_resn = resn
        if resn > max_resn:
            max_resn = resn
        if resn not in xyz:
            xyz[resn] = {}
        if resa not in xyz[resn]:
            xyz[resn][resa] = {}
        if resn not in seq:
            seq[resn] = {}
        if resa not in seq[resn]:
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_PDB(x, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
  for line in open(x,"rb"):
    line = line.decode("utf-8","ignore").rstrip()

    if line[:6] == "HETATM" and line[17:17+3] == "MSE":
      line = line.replace("HETATM","ATOM  ")
      line = line.replace("MSE","MET")

    if line[:4] == "ATOM":
      ch = line[21:22]
      if ch == chain or chain is None:
        atom = line[12:12+4].strip()
        resi = line[17:17+3]
        resn = line[22:22+5].strip()
        x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]

        if resn[-1].isalpha():
            resa,resn = resn[-1],int(resn[:-1])-1
        else:
            resa,resn = "",int(resn)-1
#         resn = int(resn)
        if resn < min_resn:
            min_resn = resn
        if resn > max_resn:
            max_resn = resn
        if resn not in xyz:
            xyz[resn] = {}
        if resa not in xyz[resn]:
            xyz[resn][resa] = {}
        if resn not in seq:
            seq[resn] = {}
        if resa not in seq[resn]:
            seq[resn][resa] = resi

        if atom not in xyz[resn][resa]:
          xyz[resn][resa][atom] = np.array([x,y,z])

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  for resn in range(min_resn,max_resn+1):
    if resn in seq:
      for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
    else: seq_.append(20)
    if resn in xyz:
      for k in sorted(xyz[resn]):
        for atom in atoms:
          if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
          else: xyz_.append(np.full(3,np.nan))
    else:
      for atom in atoms: xyz_.append(np.full(3,np.nan))
  return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))

def generate_seqopt_features( pdbfile, chains ): # multichain
    my_dict = {}
    concat_seq = ''
    concat_N = []
    concat_CA = []
    concat_C = []
    concat_O = []
    concat_mask = []
    coords_dict = {}

    for letter in chains:
        xyz, seq = parse_PDB_biounits(pdbfile, atoms=['N','CA','C','O'], chain=letter)

        concat_seq += seq[0]
        my_dict['seq_chain_'+letter]=seq[0]
        coords_dict_chain = {}
        coords_dict_chain['N_chain_'+letter]=xyz[:,0,:].tolist()
        coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
        coords_dict_chain['C_chain_'+letter]=xyz[:,2,:].tolist()
        coords_dict_chain['O_chain_'+letter]=xyz[:,3,:].tolist()
        my_dict['coords_chain_'+letter]=coords_dict_chain

    my_dict['name']=my_rstrip( pdbfile, '.pdb' )
    my_dict['num_of_chains'] = len( chains )
    my_dict['seq'] = concat_seq

    return my_dict

def get_seq_from_pdb( pdb_fn, slash_for_chainbreaks ):
    to1letter = {
      "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
      "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
      "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
      "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

    seq = ''
    with open(pdb_fn) as fp:
      for line in fp:
        if line.startswith("TER"):
          if not slash_for_chainbreaks: continue
          seq += "/"
        if not line.startswith("ATOM"):
          continue
        if line[12:16].strip() != "CA":
          continue
        resName = line[17:20]
        #
        seq += to1letter[resName]
    return my_rstrip( seq, '/' )

def init_seq_optimize_model(device, hidden_dim, num_layers, backbone_noise, num_connections, checkpoint_path):

   model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=num_connections)
   model.to(device)
   checkpoint = torch.load(checkpoint_path, map_location=device)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()

   return model

#def set_default_args( seq_per_target, omit_AAs=['X'], decoding_order='forward' ):
def set_default_args( seq_per_target, omit_AAs=['X'] ):

    #global DECODING_ORDER
    #DECODING_ORDER = decoding_order

    if not 'X' in omit_AAs: omit_AAs.append('X') # We don't want any unknown residue assignments

    retval = {}
    retval['BATCH_COPIES'] = min( 1, seq_per_target )
    retval['NUM_BATCHES'] = seq_per_target // retval['BATCH_COPIES']
    retval['temperature'] = 0.1

    omit_AAs_list = omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    retval['omit_AAs_np'] = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    retval['omit_AA_dict'] = None
    retval['pssm_dict'] = None
    retval['bias_AA_dict'] = None
    retval['tied_positions_dict'] = None
    retval['bias_by_res_dict'] = None
    retval['bias_AAs_np'] = np.zeros(len(alphabet))

    return retval

def generate_sequences( model, device, feature_dict, arg_dict, masked_chains, visible_chains, fixed_positions_dict=None ):
    seqs_scores = []

    with torch.no_grad():

        batch_clones = [copy.deepcopy( feature_dict ) for i in range(arg_dict['BATCH_COPIES'])]
        chain_id_dict = { feature_dict['name'] : ( masked_chains, visible_chains ) } # Masked, visible is the order, I think - Nate

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta= tied_featurize(
                batch_clones, 
                device, 
                chain_id_dict, 
                fixed_positions_dict, 
                arg_dict['omit_AA_dict'], 
                arg_dict['tied_positions_dict'], 
                arg_dict['pssm_dict'],
                arg_dict['bias_by_res_dict']
        )

        pssm_threshold = 0 # Nate is hardcoding this
        pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false

        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
        mask_for_loss = mask*chain_M*chain_M_pos
        scores = _scores(S, log_probs, mask_for_loss)
        native_score = scores.cpu().data.numpy()

        for j in range(arg_dict['NUM_BATCHES']):
            randn_2 = torch.randn(chain_M.shape).to(device)

            sample_dict = model.sample(
                    X,
                    randn_2,
                    S,
                    chain_M,
                    chain_encoding_all,
                    residue_idx,
                    mask=mask,
                    temperature=arg_dict['temperature'],
                    omit_AAs_np=arg_dict['omit_AAs_np'],
                    bias_AAs_np=arg_dict['bias_AAs_np'],
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef,
                    pssm_bias=pssm_bias,
                    pssm_multi=0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=pssm_log_odds_mask,
                    pssm_bias_flag=False,
                    bias_by_res=bias_by_res_all
            )

            S_sample = sample_dict["S"]

            # Compute scores
            log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2)
            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S_sample, log_probs, mask_for_loss)
            scores = scores.cpu().data.numpy()

            for b_ix in range(arg_dict['BATCH_COPIES']):
                seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                score = scores[b_ix]

                seqs_scores.append((seq,score))

    return seqs_scores
