from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]))
from sharp.data import load_ipps_data
from sharp.features import build_features
from sharp.cluster import build_zip_metrics, readmit_concentration
from sharp.temporal import build_temporal, yoy_readmit_growth
from sharp.system_perf import build_system_perf
from sharp.model import build_provider_year, add_next_year_target, train_models, save_model
from sharp.savings import tam_and_top_hospitals
from sharp.causal import label_medicaid_expansion, did_effect, estimate_dml_tlearner
from sharp.bootstrap import bootstrap_tam, bootstrap_readmit_ratio, bootstrap_did

def main():
    d = load_ipps_data()
    d = build_features(d)
    try:
        import duckdb as _duck
        _duck.register("cms", d)
        z_duck = _duck.sql(
            """
            select Provider_Zip_Code, year,
                   avg(payment_ratio) as payment_ratio,
                   sum(Total_Discharges) as Total_Discharges,
                   avg(Average_Total_Payments) as Average_Total_Payments,
                   avg(financial_stress_index) as financial_stress_index
            from cms
            group by 1,2
            """
        ).to_df()
        r_duck = _duck.sql(
            """
            select Provider_Zip_Code,
                   sum(Total_Discharges) as readmit_discharges
            from cms
            where is_readmit_prone
            group by 1
            """
        ).to_df()
        t_duck = _duck.sql(
            """
            select year, Provider_State, is_readmit_prone,
                   sum(Total_Discharges) as Total_Discharges,
                   avg(payment_ratio) as payment_ratio,
                   avg(Average_Total_Payments) as Average_Total_Payments
            from cms
            group by 1,2,3
            """
        ).to_df()
        a_duck = _duck.sql(
            """
            select Provider_Id, Provider_Name, Provider_State, year,
                   avg(payment_ratio) as payment_ratio,
                   avg(medicare_coverage_ratio) as medicare_coverage_ratio,
                   avg(financial_stress_index) as financial_stress_index,
                   avg(avg_charges_log) as avg_charges_log,
                   avg(state_avg_payment_ratio) as state_avg_payment_ratio,
                   max(hospital_size_category) as hospital_size_category,
                   max(drg_diversity_index) as drg_diversity_index,
                   sum(Total_Discharges) as readmit_discharges
            from cms
            group by 1,2,3,4
            """
        ).to_df()
    except Exception:
        z_duck = None
        r_duck = None
        t_duck = None
        a_duck = None
    z = build_zip_metrics(d)
    r = readmit_concentration(d)
    t = build_temporal(d)
    g = yoy_readmit_growth(d)
    s = build_system_perf(d)
    a = a_duck if a_duck is not None else build_provider_year(d)
    a = add_next_year_target(a)
    m = train_models(a)
    save_model(m, "models")
    d2 = label_medicaid_expansion(d)
    did = did_effect(d2)
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    z.to_csv(out/"zip_metrics.csv", index=False)
    r.to_csv(out/"readmit_concentration.csv", index=False)
    t.to_csv(out/"temporal.csv", index=False)
    if z_duck is not None:
        z_duck.to_csv(out/"zip_metrics_duck.csv", index=False)
    if r_duck is not None:
        r_duck.to_csv(out/"readmit_concentration_duck.csv", index=False)
    if t_duck is not None:
        t_duck.to_csv(out/"temporal_duck.csv", index=False)
    g.to_csv(out/"yoy_growth.csv", index=False)
    s.to_csv(out/"system_performance.csv", index=False)
    a.to_csv(out/"provider_year.csv", index=False)
    m["pred_test"].to_csv(out/"predictions_2016.csv", index=False)
    did.to_csv(out/"did_payment_ratio.csv", index=False)
    est = estimate_dml_tlearner(d2)
    if est is not None:
        import numpy as np
        np.savetxt(out/"dml_te.csv", est["dml_te"], delimiter=",")
        np.savetxt(out/"tlearner_te.csv", est["tlearner_te"], delimiter=",")
    bt = bootstrap_tam(d)
    br = bootstrap_readmit_ratio(d)
    bd = bootstrap_did(d2)
    bt.to_csv(out/"tam_bootstrap.csv", index=False)
    br.to_csv(out/"readmit_ratio_bootstrap.csv", index=False)
    bd.to_csv(out/"did_bootstrap.csv", index=False)
    tam, top100 = tam_and_top_hospitals(d)
    (out/"tam.txt").write_text(f"{tam}")
    top100.to_csv(out/"top100_hospitals.csv", index=False)
    low = d[d["payment_ratio"] < 0.3]
    high = d[d["payment_ratio"] >= 0.3]
    x1 = low[low["is_readmit_prone"]]["Total_Discharges"].sum()
    x2 = high[high["is_readmit_prone"]]["Total_Discharges"].sum()
    ratio = x1 / (x2 + 1e-6)
    (out/"readmit_ratio.txt").write_text(f"{ratio}")

if __name__ == "__main__":
    main()