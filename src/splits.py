import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from src.config import DATA_PROCESSED, FOLDS, N_SPLITS, SEED, TEST_START


def save(dev, name, splits):
    """Save fold indices (row numbers of the cleaned file) as JSON."""
    out = [
        {"fold": i, "train": dev.index[tr].tolist(), "val": dev.index[va].tolist()}
        for i, (tr, va) in enumerate(splits)
    ]
    with open(FOLDS / name, "w") as f:
        json.dump(out, f)


def make_folds():
    df = pd.read_csv(DATA_PROCESSED)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.normalize()

    # the future test period never enters any fold
    dev = df[df["datetime_utc"] < TEST_START]
    FOLDS.mkdir(exist_ok=True)

    # temporal: split on the sorted unique dates, 1-day gap
    dates = np.sort(dev["datetime_utc"].unique())
    temporal = []
    for tr_d, va_d in TimeSeriesSplit(n_splits=N_SPLITS, gap=1).split(dates):
        tr = np.flatnonzero(dev["datetime_utc"].isin(dates[tr_d]))
        va = np.flatnonzero(dev["datetime_utc"].isin(dates[va_d]))
        temporal.append((tr, va))
    save(dev, "folds_temporal.json", temporal)

    # spatial: whole stations are held out (station id is only the group)
    save(dev, "folds_spatial.json", GroupKFold(n_splits=N_SPLITS).split(dev, groups=dev["id"]))

    # standard random K-fold (diagnostic only)
    save(dev, "folds_standard.json", KFold(N_SPLITS, shuffle=True, random_state=SEED).split(dev))
    print("Folds saved in", FOLDS)


if __name__ == "__main__":
    make_folds()