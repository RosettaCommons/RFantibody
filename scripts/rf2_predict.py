import hydra
import random
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig

import rfantibody.rf2.modules.pose_util as pu
import rfantibody.rf2.modules.util as util
from rfantibody.rf2.modules.model_runner import AbPredictor
from rfantibody.rf2.modules.preprocess import Preprocess, pose_to_inference_RFinput


@hydra.main(version_base=None, config_path='../src/rfantibody/rf2/config', config_name='base')
def main(conf: HydraConfig) -> None:
    """
    Main function
    """
    print(f'Running RF2 with the following configs: {conf}')
    
    # Set up deterministic mode if seed is provided
    if hasattr(conf.inference, 'seed') and conf.inference.seed is not None:
        seed = conf.inference.seed
        print(f"Setting up deterministic mode with seed {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    done_list=util.get_done_list(conf)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    preprocessor=Preprocess(pose_to_inference_RFinput, conf)
    predictor=AbPredictor(conf, preprocess_fn=preprocessor, device=device)
    for pose, tag in pu.pose_generator(conf):
        if tag in done_list and conf.inference.cautious:
            print(f'Skipping {tag} as output already exists')
            continue
        predictor(pose, tag)

if __name__ == '__main__':
    main()
