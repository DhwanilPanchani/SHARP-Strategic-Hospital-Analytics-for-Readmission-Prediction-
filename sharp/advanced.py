import numpy as np
import pandas as pd

def spatial_autocorrelation(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby(["Provider_Zip_Code", "year"]).agg(payment_ratio=("payment_ratio","mean")).reset_index()
    c["zip_int"] = c["Provider_Zip_Code"].astype(str).str.slice(0,3).astype(int)
    c = c.sort_values(["year","zip_int"]) 
    c["lag_ratio"] = c.groupby("year")["payment_ratio"].shift(1)
    c["diff"] = (c["payment_ratio"] - c["lag_ratio"]).abs()
    return c

def anomaly_hospitals(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["Provider_Id"]).agg(
        mean_ratio=("payment_ratio","mean"),
        std_ratio=("payment_ratio","std"),
    ).reset_index()
    g["z"] = (g["mean_ratio"] - g["mean_ratio"].mean()) / (g["mean_ratio"].std() + 1e-6)
    return g.sort_values("z")

def network_metrics(df: pd.DataFrame) -> pd.DataFrame:
    e = df.groupby(["Provider_Zip_Code", "Provider_Id"]).agg(
        vol=("Total_Discharges","sum")
    ).reset_index()
    z = e.merge(e, on="Provider_Zip_Code", suffixes=("_u","_v"))
    z = z[z["Provider_Id_u"] != z["Provider_Id_v"]]
    z["weight"] = z[["vol_u","vol_v"]].min(axis=1)
    deg = z.groupby("Provider_Id_u")["weight"].sum().rename("degree")
    return deg.reset_index().rename(columns={"Provider_Id_u":"Provider_Id"})

def survival_dataset(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["is_readmit_prone"]].groupby(["Provider_Id", "year"]).agg(
        vol=("Total_Discharges","sum")
    ).reset_index()
    g = g.sort_values(["Provider_Id","year"])
    g["prev"] = g.groupby("Provider_Id")["vol"].shift(1)
    g["growth"] = (g["vol"] - g["prev"]) / (g["prev"] + 1e-6)
    th = g["growth"].quantile(0.80)
    first = g[g["growth"] >= th].groupby("Provider_Id")["year"].min().rename("event_year")
    base = g.groupby("Provider_Id")["year"].min().rename("start_year")
    last = g.groupby("Provider_Id")["year"].max().rename("last_year")
    s = pd.concat([first, base, last], axis=1).reset_index()
    s["duration"] = (s["event_year"].fillna(s["last_year"]) - s["start_year"]) 
    s["event"] = (~s["event_year"].isna()).astype(int)
    return s