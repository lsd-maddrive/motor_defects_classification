import logging
import os

import hydra
from omegaconf import DictConfig


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DPATH = os.path.join(PROJECT_ROOT, "configs")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_multiclass")


@hydra.main(config_path=CONFIG_DPATH, config_name="train_multiclass")
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)  # noqa: F841
    criterion = hydra.utils.instantiate(cfg.criterion)  # noqa: F841

    CHECKPOINTS_DPATH = os.path.join(PROJECT_ROOT, "train_checkpoints", "multiclass")
    os.makedirs(CHECKPOINTS_DPATH, exist_ok=True)


if __name__ == "__main__":
    main()
