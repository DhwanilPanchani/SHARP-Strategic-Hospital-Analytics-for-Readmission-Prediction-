import numpy as np
import pandas as pd

def _tam(df: pd.DataFrame) -> float:
    stressed = df[df["payment_ratio"] < 0.3].groupby("DRG_Code").agg(
        Average_Total_Payments=("Average_Total_Payments","mean"),
        Total_Discharges=("Total_Discharges","sum")
    )
    normal = df[df["payment_ratio"] > 0.5].groupby("DRG_Code").agg(
        Average_Total_Payments=("Average_Total_Payments","mean"),
        Total_Discharges=("Total_Discharges","sum")
    )
    inter = stressed.join(normal, lsuffix="_s", rsuffix="_n", how="inner")
    s = (inter["Average_Total_Payments_s"] - inter["Average_Total_Payments_n"]) * inter["Total_Discharges_s"]
    return float(s.sum())

def bootstrap_tam(df: pd.DataFrame, n_boot: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    drgs = df["DRG_Code"].dropna().unique()
    vals = []
    for _ in range(n_boot):
        sample_drgs = rng.choice(drgs, size=len(drgs), replace=True)
        b = df[df["DRG_Code"].isin(sample_drgs)]
        vals.append(_tam(b))
    arr = np.array(vals)
    return pd.DataFrame({
        "mean": [arr.mean()],
        "p2_5": [np.percentile(arr, 2.5)],
        "p97_5": [np.percentile(arr, 97.5)],
    })

def bootstrap_readmit_ratio(df: pd.DataFrame, n_boot: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    providers = df["Provider_Id"].unique()
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(providers, size=len(providers), replace=True)
        b = df[df["Provider_Id"].isin(samp)]
        low = b[b["payment_ratio"] < 0.3]
        high = b[b["payment_ratio"] >= 0.3]
        x1 = low[low["is_readmit_prone"]]["Total_Discharges"].sum()
        x2 = high[high["is_readmit_prone"]]["Total_Discharges"].sum()
        vals.append(x1 / (x2 + 1e-6))
    arr = np.array(vals)
    return pd.DataFrame({
        "mean": [arr.mean()],
        "p2_5": [np.percentile(arr, 2.5)],
        "p97_5": [np.percentile(arr, 97.5)],
    })

def bootstrap_did(df: pd.DataFrame, n_boot: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    states = df["Provider_State"].unique()
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(states, size=len(states), replace=True)
        b = df[df["Provider_State"].isin(samp)]
        treated = b["Provider_State"].isin(["CA","NY","IL","PA","OH","MI","NJ","WA","MA","MD"])
        pre = b[b["year"] < 2014]
        post = b[b["year"] >= 2014]
        pre_g = pre.groupby(treated)["payment_ratio"].mean()
        post_g = post.groupby(treated)["payment_ratio"].mean()
        did = (post_g.get(True, np.nan) - pre_g.get(True, np.nan)) - (post_g.get(False, np.nan) - pre_g.get(False, np.nan))
        vals.append(did)
    arr = np.array(vals)
    return pd.DataFrame({
        "mean": [np.nanmean(arr)],
        "p2_5": [np.nanpercentile(arr, 2.5)],
        "p97_5": [np.nanpercentile(arr, 97.5)],
    })