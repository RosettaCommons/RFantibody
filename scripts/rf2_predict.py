import torch

import hydra
from hydra.core.hydra_config import HydraConfig

import rfantibody.rf2.modules.util as util
import rfantibody.rf2.modules.pose_util as pu
from rfantibody.rf2.modules.model_runner import AbPredictor
from rfantibody.rf2.modules.preprocess import pose_to_inference_RFinput, Preprocess

@hydra.main(version_base=None, config_path='/opt/RFantibody/src/rfantibody/rf2/config', config_name='base')
def main(conf: HydraConfig) -> None:
    """
    Main function
    """
    print(f'Running RF2 with the following configs: {conf}')
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
