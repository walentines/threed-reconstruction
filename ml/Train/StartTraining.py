
import sys
sys.path.append('..')

import hydra
from omegaconf import DictConfig
import torch
from AccelerateControlStableVideoDiffusion import TrainControlStableVideoDiffusion

@hydra.main(version_base=None, config_path="hydra_config", config_name="config_hydra")
def main(config: DictConfig):
    trainer = TrainControlStableVideoDiffusion(config)
    trainer.train()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()