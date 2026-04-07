import hydra
from omegaconf import OmegaConf, DictConfig
from boltz.data.module.training import DataConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    from hydra.utils import instantiate
    data = instantiate(cfg.data)
    data_config = DataConfig(**data)
    print("DataConfig instance datasets type:", type(data_config.datasets[0]))
    print("DataConfig instance cropper type:", type(data_config.datasets[0].cropper))

if __name__ == "__main__":
    main()
