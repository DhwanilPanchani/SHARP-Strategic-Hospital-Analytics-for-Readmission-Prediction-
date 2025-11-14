import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
from pathlib import Path

def build_provider_year(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["Provider_Id", "Provider_Name", "Provider_State", "year"]).agg(
        payment_ratio=("payment_ratio", "mean"),
        medicare_coverage_ratio=("medicare_coverage_ratio", "mean"),
        financial_stress_index=("financial_stress_index", "mean"),
        avg_charges_log=("avg_charges_log", "mean"),
        state_avg_payment_ratio=("state_avg_payment_ratio", "mean"),
        hospital_size_category=("hospital_size_category", "max"),
        drg_diversity_index=("drg_diversity_index", "max"),
        readmit_discharges=("Total_Discharges", "sum"),
    ).reset_index()
    return agg

def add_next_year_target(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.sort_values(["Provider_Id", "year"]) 
    agg["next_readmit_discharges"] = agg.groupby("Provider_Id")["readmit_discharges"].shift(-1)
    agg["target_growth"] = (agg["next_readmit_discharges"] - agg["readmit_discharges"]) / agg["readmit_discharges"]
    q = agg["target_growth"].quantile(0.80)
    agg["high_risk"] = (agg["target_growth"] >= q).astype(int)
    return agg.dropna(subset=["next_readmit_discharges"]) 

def train_models(agg: pd.DataFrame):
    feats = [
        "payment_ratio",
        "medicare_coverage_ratio",
        "financial_stress_index",
        "avg_charges_log",
        "state_avg_payment_ratio",
        "drg_diversity_index",
        "year",
    ]
    d = agg.copy()
    d["hospital_size_category"] = d["hospital_size_category"].astype(str)
    d = pd.get_dummies(d, columns=["hospital_size_category"], dummy_na=True)
    X = d[feats + [c for c in d.columns if c.startswith("hospital_size_category_")]]
    y_reg = d["next_readmit_discharges"]
    y_cls = d["high_risk"]
    train = d[d["year"] <= 2013]
    val = d[d["year"] == 2014]
    test = d[d["year"] == 2015]
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(train[X.columns], train["next_readmit_discharges"])
    val_pred = rf.predict(val[X.columns])
    test_pred = rf.predict(test[X.columns])
    mae_val = mean_absolute_error(val["next_readmit_discharges"], val_pred)
    mae_test = mean_absolute_error(test["next_readmit_discharges"], test_pred)
    prob = (test_pred - test["readmit_discharges"]) / (test["readmit_discharges"] + 1e-6)
    thr = d["target_growth"].quantile(0.80)
    score = roc_auc_score(test["high_risk"], np.clip(prob / (thr + 1e-6), 0, 1))
    return {
        "model": rf,
        "features": X.columns.tolist(),
        "mae_val": mae_val,
        "mae_test": mae_test,
        "auc_test": float(score),
        "pred_test": test.assign(pred_next=test_pred),
    }

def save_model(bundle: dict, path_dir: str):
    p = Path(path_dir)
    p.mkdir(exist_ok=True)
    joblib.dump(bundle["model"], p/"rf.pkl")
    (p/"features.json").write_text(json.dumps(bundle["features"]))