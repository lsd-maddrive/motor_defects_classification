import logging
import os

import pandas as pd

from .utils.hash import dir_hash
from .utils.fs import read_data


class DefectDataset:
    """Dataset class for files control functions."""

    DEFAULT_EXTENSION = "xlsx"

    def __init__(self, dpath: str):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(dpath):
            raise ValueError("Dataset is not initialized - create it first!")

        self.dir_hash = dir_hash(dpath)
        self._dpath = dpath

        self._logger.info(f"Dataset directory: {self._dpath}")
        self._logger.info(f"Computed dataset hash: {self.dir_hash}")

        self._train_fpath = os.path.join(self._dpath, f"train.{self.DEFAULT_EXTENSION}")
        self._logger.info(f"Train data path: {self._train_fpath}")

        self._val_fpath = os.path.join(self._dpath, f"val.{self.DEFAULT_EXTENSION}")
        self._logger.info(f"Val data path: {self._val_fpath}")

        self._test_fpath = os.path.join(self._dpath, f"test.{self.DEFAULT_EXTENSION}")
        self._logger.info(f"Test data path: {self._test_fpath}")

    def _read_file(self, fpath: str) -> pd.DataFrame:
        if fpath.lower().endswith(".csv"):
            return pd.read_csv(fpath, index_col=0)
        elif fpath.lower().endswith(".xlsx"):
            return pd.read_excel(fpath)

    @property
    def hash(self) -> str:
        return self.dir_hash

    def get_data(self, mode: str) -> pd.DataFrame:
        if mode == "train":
            df = self._read_file(self.train_fpath)
            self._logger.info(f"Train Data Shape: {df.shape}")
            return df
        elif mode == "valid":
            df = self._read_file(self.valid_fpath)
            self._logger.info(f"Valid Data Shape: {df.shape}")
            return df
        elif mode == "test":
            df = self._read_file(self.test_fpath)
            self._logger.info(f"Test Data Shape: {df.shape}")
            return df
        raise ValueError(f"Invalid mode value: {mode}")


class FileSystemDataset:
    """Dataset class for reading motor sygnal data from file system."""

    def __init__(
        self,
        data_dpath: str,
        label_df: pd.DataFrame,
        fname_col_name: str,
        label_col_name: str
    ):
        self._data_dpath = data_dpath
        self._label_df = label_df

        self._fname_col = fname_col_name
        self._label_col = label_col_name

    def __len__(self):
        return len(self._label_df)

    def __getitem__(self, idx):
        idx_series = self._label_df.iloc[idx]

        fname = idx_series[self._fname_col]
        label = idx_series[self._label_col]

        fpath = os.path.join(self._data_dpath, fname)
        data = read_data(fpath)

        return data, label

    def get_labels(self):
        return self._labels_df[self._label_col_name].values
