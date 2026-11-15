import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED

LAG_COLS = {"pm25": [1, 2, 3], "AOD": [1, 2, 3], "NO2": [1]}
ROLL_COLS = ["pm25", "AOD"]
DIFF_COLS = ["AOD", "temperature_celsius", "NO2"]


def station_features(g):
    """Features for one station. All of them use past days only (shift(1) first)."""
    g = g.set_index("datetime_utc")
    # regular daily timeline: a missing day is a NaN row, so a lag never jumps over a gap
    full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    out = pd.DataFrame(index=full.index)

    for c, lags in LAG_COLS.items():
        for k in lags:
            out[f"{c}_lag{k}"] = full[c].shift(k)

    for c in ROLL_COLS:
        past = full[c].shift(1)
        for w in (3, 7):
            out[f"{c}_roll{w}_mean"] = past.rolling(w, min_periods=w).mean()
            out[f"{c}_roll{w}_std"] = past.rolling(w, min_periods=w).std()
    out["AOD_roll7_max"] = full["AOD"].shift(1).rolling(7, min_periods=7).max()

    for c in DIFF_COLS:
        past = full[c].shift(1)
        out[f"{c}_diff1"] = past - past.shift(1)
    past_aod = full["AOD"].shift(1)
    out["AOD_rate3"] = (past_aod - past_aod.shift(3)) / 3

    doy = full.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # keep only the days that really exist in the data
    out = out.loc[g.index]
    out.index.name = "datetime_utc"
    return out


def build_features(df):
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.normalize()
    df = df.drop_duplicates(subset=["id", "datetime_utc"]).sort_values(["id", "datetime_utc"])
    parts = []
    for sid, g in df.groupby("id"):
        f = station_features(g)
        f["id"] = sid
        parts.append(f.reset_index())
    feats = pd.concat(parts)
    return df.merge(feats, on=["id", "datetime_utc"], how="left")


if __name__ == "__main__":
    data = build_features(pd.read_csv(DATA_PROCESSED))
    out_path = DATA_PROCESSED.with_name("features.csv")
    data.to_csv(out_path, index=False)
    print(data.shape, "->", out_path)