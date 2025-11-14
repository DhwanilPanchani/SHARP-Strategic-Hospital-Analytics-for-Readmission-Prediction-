import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import joblib
from pathlib import Path
from sharp.data import load_ipps_data
from sharp.features import build_features
from sharp.cluster import build_zip_metrics
from sharp.savings import tam_and_top_hospitals
from sharp.model import build_provider_year, add_next_year_target, train_models

st.set_page_config(page_title="SHARP Dashboard", layout="wide")
@st.cache_data
def load_data():
    d = load_ipps_data()
    d = build_features(d)
    return d

data = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["National Heatmap","Hospital Deep Dive","Opportunity Finder","What-If Simulator","Model Performance"])

with tab1:
    st.sidebar.title("Filters")
    sel_state = st.sidebar.multiselect("State", sorted(data["Provider_State"].unique()))
    sel_years = st.sidebar.slider("Years", 2011, 2016, (2011, 2016))
    df = data[(data["year"]>=sel_years[0]) & (data["year"]<=sel_years[1])]
    if sel_state:
        df = df[df["Provider_State"].isin(sel_state)]
    s = df.groupby(["Provider_State","year"]).agg(
        readmit_discharges=("Total_Discharges","sum"),
        stress=("financial_stress_index","mean")
    ).reset_index()
    fig_anim = px.choropleth(s, locations="Provider_State", locationmode="USA-states", color="readmit_discharges", scope="usa", animation_frame="year")
    st.plotly_chart(fig_anim, use_container_width=True)
    corr = df[["payment_ratio","medicare_coverage_ratio","financial_stress_index","avg_charges_log"]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto"), use_container_width=True)

with tab2:
    pid = st.selectbox("Hospital", options=sorted(data["Provider_Name"].unique()))
    h = data[data["Provider_Name"]==pid]
    ts = h.groupby("year").agg(
        discharges=("Total_Discharges","sum"),
        payment_ratio=("payment_ratio","mean"),
    ).reset_index()
    st.metric("Avg Payment Ratio", f"{ts['payment_ratio'].mean():.2f}")
    st.metric("Total Readmit-prone Discharges", int(h[h["is_readmit_prone"]]["Total_Discharges"].sum()))
    st.plotly_chart(px.line(ts, x="year", y="discharges"), use_container_width=True)
    st.plotly_chart(px.line(ts, x="year", y="payment_ratio"), use_container_width=True)

with tab3:
    st.subheader("Filters")
    f_state = st.multiselect("State", sorted(data["Provider_State"].unique()))
    f_drg = st.multiselect("DRG", sorted(data["DRG_Code"].dropna().unique()))
    f_size = st.multiselect("Size", ["small","medium","large"])
    df = data.copy()
    if f_state:
        df = df[df["Provider_State"].isin(f_state)]
    if f_drg:
        df = df[df["DRG_Code"].isin(f_drg)]
    if f_size:
        df = df[df["hospital_size_category"].astype(str).isin(f_size)]
    tam, top100 = tam_and_top_hospitals(df)
    st.metric("Total Addressable Market", f"${tam:,.0f}")
    st.dataframe(top100)
    st.download_button("Download Top 100 CSV", top100.to_csv(index=False), file_name="top100_hospitals.csv")

with tab4:
    ratio = st.slider("Payment ratio", 0.1, 0.8, 0.3, 0.01)
    df = data.copy()
    df["scenario_ratio"] = ratio
    df["scenario_stress"] = 1 - df["scenario_ratio"]
    base_tam, _ = tam_and_top_hospitals(df)
    st.metric("Projected Savings TAM", f"${base_tam:,.0f}")
    st.write(df[["Provider_Name","Provider_State","scenario_ratio","scenario_stress"]].head(50))

with tab5:
    try:
        pred = pd.read_csv("outputs/predictions_2016.csv")
        panel = pd.read_csv("outputs/provider_year.csv")
        thr = panel["target_growth"].quantile(0.80)
        score_raw = (pred["pred_next"] - pred["readmit_discharges"]) / (pred["readmit_discharges"] + 1e-6)
        score = np.clip(score_raw / (thr + 1e-6), 0, 1)
        y = pred["high_risk"].astype(int).values
        fpr, tpr, _ = roc_curve(y, score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y, score)
        pr_auc = auc(recall, precision)
        prev = float(y.mean())
        df_lift = pd.DataFrame({"score": score, "y": y}).sort_values("score", ascending=False)
        df_lift["rank"] = np.arange(1, len(df_lift)+1)
        df_lift["decile"] = pd.qcut(df_lift["rank"], 10, labels=False)
        lift = df_lift.groupby("decile")["y"].sum().cumsum() / (prev * ((df_lift.groupby("decile").size().cumsum())))
        lift = lift.rename("lift")
        cal = pd.DataFrame({"score": score, "y": y})
        cal["bin"] = pd.qcut(cal["score"], 10, duplicates="drop")
        cal_plot = cal.groupby("bin").agg(pred=("score","mean"), obs=("y","mean")).reset_index()
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
        st.plotly_chart(fig_roc, use_container_width=True)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC={pr_auc:.3f}"))
        st.plotly_chart(fig_pr, use_container_width=True)
        fig_lift = px.line(lift.reset_index(), x="decile", y="lift")
        st.plotly_chart(fig_lift, use_container_width=True)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=cal_plot["pred"], y=cal_plot["obs"], mode="markers+lines", name="calibration"))
        fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="perfect", line=dict(dash="dash")))
        st.plotly_chart(fig_cal, use_container_width=True)
        total_pos = df_lift["y"].sum()
        df_gains = df_lift.copy()
        df_gains["target_frac"] = df_gains["rank"] / len(df_gains)
        df_gains["cum_positives"] = df_gains["y"].cumsum()
        df_gains["cum_capture_rate"] = df_gains["cum_positives"] / (total_pos + 1e-6)
        fig_gains = go.Figure()
        fig_gains.add_trace(go.Scatter(x=df_gains["target_frac"], y=df_gains["cum_capture_rate"], mode="lines", name="cumulative gains"))
        fig_gains.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
        st.plotly_chart(fig_gains, use_container_width=True)
        frac = st.slider("Target fraction", 0.05, 0.5, 0.2, 0.05)
        top_k = int(frac * len(df_lift))
        cap = df_lift.iloc[:top_k]["y"].sum() / (total_pos + 1e-6)
        st.metric("Top-K Capture Rate", f"{cap:.2%}")
        sel_states = st.multiselect("States for breakdown", sorted(pred["Provider_State"].unique()))
        perf_rows = []
        for stt, g in pred.groupby("Provider_State"):
            yy = g["high_risk"].astype(int).values
            ss = np.clip(((g["pred_next"] - g["readmit_discharges"]) / (g["readmit_discharges"] + 1e-6)) / (thr + 1e-6), 0, 1)
            try:
                fpr_s, tpr_s, _ = roc_curve(yy, ss)
                roc_s = auc(fpr_s, tpr_s)
                pr_s = auc(*precision_recall_curve(yy, ss)[1::-1])
            except Exception:
                roc_s = np.nan
                pr_s = np.nan
            perf_rows.append({"Provider_State": stt, "ROC_AUC": roc_s, "PR_AUC": pr_s, "Prevalence": float(np.mean(yy))})
        perf = pd.DataFrame(perf_rows)
        if sel_states:
            perf = perf[perf["Provider_State"].isin(sel_states)]
        perf = perf.sort_values("ROC_AUC", ascending=False)
        st.dataframe(perf)
        st.plotly_chart(px.bar(perf.dropna(subset=["ROC_AUC"]).head(20), x="Provider_State", y="ROC_AUC"), use_container_width=True)
        try:
            rf = joblib.load("models/rf.pkl")
            feats = json.loads(Path("models/features.json").read_text())
            imp = pd.DataFrame({"feature": feats, "importance": rf.feature_importances_})
            imp = imp.sort_values("importance", ascending=False)
            st.plotly_chart(px.bar(imp.head(20), x="feature", y="importance"), use_container_width=True)
        except Exception:
            st.write("Model feature importances unavailable.")
        st.metric("Baseline Prevalence", f"{prev:.2%}")
        st.metric("ROC AUC", f"{roc_auc:.3f}")
        st.metric("PR AUC", f"{pr_auc:.3f}")
    except Exception as e:
        st.write(f"Performance data unavailable: {e}")