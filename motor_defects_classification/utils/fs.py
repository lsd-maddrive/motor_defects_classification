from datetime import date

import numpy as np
import pandas as pd


def get_date_string() -> str:
    return date.today().strftime("%Y-%m-%d")


def read_data(fpath: str) -> np.ndarray:
    df = pd.read_csv(fpath)
    df_np = df.drop("Time", axis=1).to_numpy()
    return df_np
