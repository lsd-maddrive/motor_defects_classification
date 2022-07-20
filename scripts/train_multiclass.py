import logging
import os

import hydra
from omegaconf import DictConfig

from motor_defects_classification.datasets import DefectDataset, FileSystemDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIG_DPATH = os.path.join(PROJECT_ROOT, "configs")
DATA_DPATH = os.path.join(PROJECT_ROOT, "data")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_multiclass")


def get_loaders(cfg: DictConfig, dataset: DefectDataset):
    train_df = dataset.get_data(mode="train")
    valid_df = dataset.get_data(mode="valid")

    train_dataset = FileSystemDataset(  # noqa: F841
        data_dpath=DATA_DPATH,
        label_df=train_df,
        fname_col_name="fpath",
        label_col_name="target"
    )

    valid_dataset = FileSystemDataset(  # noqa: F841
        data_dpath=DATA_DPATH,
        label_df=valid_df,
        fname_col_name="fpath",
        label_col_name="target"
    )


@hydra.main(config_path=CONFIG_DPATH, config_name="train_multiclass")
def main(cfg: DictConfig):
    data_dpath = os.path.join(DATA_DPATH, cfg.data_version)
    files_dataset = DefectDataset(data_dpath)  # noqa: F841

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)  # noqa: F841
    criterion = hydra.utils.instantiate(cfg.criterion)  # noqa: F841

    CHECKPOINTS_DPATH = os.path.join(PROJECT_ROOT, "train_checkpoints", "multiclass")
    os.makedirs(CHECKPOINTS_DPATH, exist_ok=True)


if __name__ == "__main__":
    main()
