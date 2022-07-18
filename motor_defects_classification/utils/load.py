import os


import pandas as pd
from tqdm import tqdm


def load_class_names():
    return [
        "no defect",
        "bearing ball defect",
        "bearing outer ring defect",
        "bearing inner ring defect",
        "intercoil defect",
        "rotor defect",
        "static eccentricity",
    ]


def load_statistics(dpath: str):
    df = pd.DataFrame()
    # TODO: multiple folders
    stream = tqdm(os.listdir(dpath), desc="Files Processing")
    for fname in stream:
        raw_df = pd.read_csv(fname)
        # get the last time value
        duration = df.Time.iat[-1]

        df = df.append(
            other={"duration": duration, "n_samples": raw_df.shape[0], "fname": fname},
            ignore_index=True
        )

    return df
