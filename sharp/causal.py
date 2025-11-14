import pandas as pd

EXPANSION_STATES = [
    "CA","NY","IL","PA","OH","MI","NJ","WA","MA","MD"
]

def label_medicaid_expansion(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["medicaid_expansion"] = d["Provider_State"].isin(EXPANSION_STATES) & (d["year"] >= 2014)
    return d

def did_effect(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["treated"] = d["Provider_State"].isin(EXPANSION_STATES)
    pre = d[d["year"] < 2014]
    post = d[d["year"] >= 2014]
    pre_g = pre.groupby("treated")["payment_ratio"].mean().rename("pre")
    post_g = post.groupby("treated")["payment_ratio"].mean().rename("post")
    m = pd.concat([pre_g, post_g], axis=1).reset_index()
    m["did"] = (m.loc[m["treated"]==True, "post"].values[0] - m.loc[m["treated"]==True, "pre"].values[0]) - (
        m.loc[m["treated"]==False, "post"].values[0] - m.loc[m["treated"]==False, "pre"].values[0]
    )
    return m

def estimate_dml_tlearner(df: pd.DataFrame):
    try:
        from econml.dml import LinearDML
        from xgboost import XGBRegressor
        from causalml.inference.meta import TLearner
    except Exception:
        return None
    d = df.copy()
    d["treated"] = d["Provider_State"].isin(EXPANSION_STATES) & (d["year"] >= 2014)
    y = d["payment_ratio"].values
    T = d["treated"].astype(int).values
    X = d[["year","financial_stress_index","state_avg_payment_ratio","avg_charges_log"]].values
    dm = LinearDML(model_y=XGBRegressor(), model_t=XGBRegressor())
    dm.fit(y, T, X=X)
    tl = TLearner()
    te = tl.fit_predict(X, T, y)
    return {"dml_te": dm.effect(X), "tlearner_te": te}