import logging
from configs import CN

# from .find_data_path import find_data_path
# from .models_default_config import build_model_default_config

__all__ = [
    'build_config',
]


def build_config(cfg, name=None):
    """
    Built the config, defined by `cfg.config.name`.
    """

    # data_dir = find_data_path(cfg.dataset, cfg.cluster_name)
    # cfg.data_dir = data_dir
    logging.info("Data dir is {}".format(cfg.data_dir))
    # build_model_default_config(cfg, cfg.model)




