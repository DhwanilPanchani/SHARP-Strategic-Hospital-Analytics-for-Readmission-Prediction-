import pandas as pd

def build_system_perf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hospital_system"] = df["Provider_Name"].str.extract(
        r"(BAPTIST|MERCY|ADVENTIST|PRESBYTERIAN|METHODIST|CATHOLIC|KAISER|HCA|TENET|ASCENSION)"
    )
    s = df.groupby(["hospital_system", "is_readmit_prone"]).agg(
        Total_Discharges=("Total_Discharges", "sum"),
        payment_ratio=("payment_ratio", "mean"),
        Average_Total_Payments=("Average_Total_Payments", "mean"),
    ).reset_index()
    return s