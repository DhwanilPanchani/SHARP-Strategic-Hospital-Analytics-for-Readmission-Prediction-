import pandas as pd

def build_temporal(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby(["year", "Provider_State", "is_readmit_prone"]).agg(
        Total_Discharges=("Total_Discharges", "sum"),
        payment_ratio=("payment_ratio", "mean"),
        Average_Total_Payments=("Average_Total_Payments", "mean"),
    ).reset_index()
    return t

def yoy_readmit_growth(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["is_readmit_prone"]].groupby(["Provider_Id", "year"]).agg(
        readmit_discharges=("Total_Discharges", "sum")
    ).reset_index()
    g = g.sort_values(["Provider_Id", "year"]) 
    g["prev"] = g.groupby("Provider_Id")["readmit_discharges"].shift(1)
    g["yoy_growth"] = (g["readmit_discharges"] - g["prev"]) / g["prev"]
    return g