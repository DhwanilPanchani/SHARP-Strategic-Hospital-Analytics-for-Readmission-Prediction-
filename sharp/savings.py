import pandas as pd

def tam_and_top_hospitals(df: pd.DataFrame) -> tuple:
    stressed = df[df["payment_ratio"] < 0.3].groupby("DRG_Code").agg(
        Average_Total_Payments=("Average_Total_Payments", "mean"),
        Total_Discharges=("Total_Discharges", "sum"),
    )
    normal = df[df["payment_ratio"] > 0.5].groupby("DRG_Code").agg(
        Average_Total_Payments=("Average_Total_Payments", "mean"),
        Total_Discharges=("Total_Discharges", "sum"),
    )
    s = (stressed["Average_Total_Payments"] - normal["Average_Total_Payments"]) * stressed["Total_Discharges"]
    tam = s.sum()
    by_hospital = df.copy()
    by_hospital["delta_cost"] = by_hospital["Average_Total_Payments"] - by_hospital.groupby("DRG_Code")["Average_Total_Payments"].transform("median")
    by_hospital["opportunity"] = by_hospital["delta_cost"] * by_hospital["Total_Discharges"]
    ranking = (
        by_hospital.groupby(["Provider_Id", "Provider_Name", "Provider_State"]).agg(
            opportunity=("opportunity", "sum")
        ).reset_index().sort_values("opportunity", ascending=False)
    )
    top100 = ranking.head(100)
    return tam, top100