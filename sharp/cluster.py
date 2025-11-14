import pandas as pd

def build_zip_metrics(df: pd.DataFrame) -> pd.DataFrame:
    z = df.groupby(["Provider_Zip_Code", "year"]).agg(
        payment_ratio=("payment_ratio", "mean"),
        Total_Discharges=("Total_Discharges", "sum"),
        Average_Total_Payments=("Average_Total_Payments", "mean"),
        financial_stress_index=("financial_stress_index", "mean"),
    ).reset_index()
    q = z["payment_ratio"].quantile(0.25)
    z["is_stressed_area"] = z["payment_ratio"] < q
    return z

def readmit_concentration(df: pd.DataFrame) -> pd.DataFrame:
    r = df[df["is_readmit_prone"]].groupby("Provider_Zip_Code").agg(
        readmit_discharges=("Total_Discharges", "sum")
    ).reset_index()
    return r