import numpy as np
import pandas as pd

HIGH_READMIT_DRGS = {
    "291": "Heart Failure",
    "292": "Heart Failure CC",
    "293": "Heart Failure MCC",
    "190": "COPD",
    "191": "COPD CC",
    "192": "COPD MCC",
    "193": "Simple Pneumonia",
    "194": "Pneumonia CC",
    "195": "Pneumonia MCC",
}

def build_features(cms: pd.DataFrame) -> pd.DataFrame:
    df = cms.copy()
    df["payment_ratio"] = df["Average_Total_Payments"] / df["Average_Covered_Charges"]
    df["medicare_coverage_ratio"] = df["Average_Medicare_Payments"] / df["Average_Total_Payments"]
    df["financial_stress_index"] = 1 - df["payment_ratio"]
    df["DRG_Code"] = df["DRG_Definition"].astype(str).str.extract(r"(\d{3})")
    df["is_readmit_prone"] = df["DRG_Code"].isin(list(HIGH_READMIT_DRGS.keys()))
    df["avg_charges_log"] = np.log1p(df["Average_Covered_Charges"])
    state_ratio = (
        df.groupby(["Provider_State", "year"])["payment_ratio"].mean().rename("state_avg_payment_ratio")
    )
    df = df.merge(state_ratio.reset_index(), on=["Provider_State", "year"], how="left")
    size = df.groupby(["Provider_Id", "year"])['Total_Discharges'].sum().rename('year_discharge_total')
    df = df.merge(size.reset_index(), on=["Provider_Id", "year"], how="left")
    bins = [0, 500, 2000, np.inf]
    labels = ["small", "medium", "large"]
    df["hospital_size_category"] = pd.cut(df["year_discharge_total"], bins=bins, labels=labels)
    diversity = df.groupby(["Provider_Id", "year"])['DRG_Code'].nunique().rename('drg_diversity_index')
    df = df.merge(diversity.reset_index(), on=["Provider_Id", "year"], how="left")
    return df