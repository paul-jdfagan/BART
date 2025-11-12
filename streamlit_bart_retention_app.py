import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import pymc as pm
import pymc_bart as pmb
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

from io import StringIO
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from pymc_bart.split_rules import ContinuousSplitRule, SubsetSplitRule

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="BART Retention Analysis", layout="wide")
st.title("BART Retention Analysis")
st.caption("Guided workflow based on the shared BART retention notebook.")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce")

def compute_retention(df: pd.DataFrame, kept_col: str, active_col: str) -> pd.Series:
    denom = df[active_col].replace(0, np.nan)
    return (df[kept_col] / denom).clip(0, 1)

def line_ci(ax, x, y_mean, y_low, y_high, label=None):
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_low, y_high, alpha=0.2)

def prepare_design_matrix(df, period_col, cohort_col):
    t = (df[period_col] - df[period_col].min()).dt.days.astype(float)
    le = LabelEncoder()
    c = le.fit_transform(df[cohort_col].astype(str)).astype(float)
    X = np.column_stack([t, c])
    return X, t, c, le

# ------------------------------
# Sidebar: Data Input
# ------------------------------
st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
example = st.sidebar.checkbox("Use synthetic example", value=False)

if uploaded is not None:
    df = load_csv(uploaded)
elif example:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    cohorts = ["A", "B", "C"]
    rows = []
    for ci, c in enumerate(cohorts):
        base = 0.45 + 0.05 * ci
        trend = np.linspace(0, 0.1, len(dates))
        for i, d in enumerate(dates):
            n = rng.integers(80, 160)
            p = np.clip(base + trend[i] + rng.normal(0, 0.03), 0.01, 0.99)
            kept = rng.binomial(n, p)
            rows.append({"period": d, "cohort": c, "n_active_users": n, "n_retained": kept})
    df = pd.DataFrame(rows)
else:
    st.info("Upload a CSV or tick 'Use synthetic example' to proceed.")
    st.stop()

st.write("**Preview**")
st.dataframe(df.head(20), use_container_width=True)

# ------------------------------
# Sidebar: Column Mapping
# ------------------------------
st.sidebar.header("2) Map Columns")
all_cols = df.columns.tolist()

default_period = "period" if "period" in all_cols else all_cols[0]
default_cohort = "cohort" if "cohort" in all_cols else all_cols[1] if len(all_cols) > 1 else all_cols[0]
default_n_active = "n_active_users" if "n_active_users" in all_cols else all_cols[-2] if len(all_cols) >= 2 else all_cols[0]
default_n_kept = "n_retained" if "n_retained" in all_cols else all_cols[-1]

period_col = st.sidebar.selectbox("Period (date)", options=all_cols, index=all_cols.index(default_period))
cohort_col = st.sidebar.selectbox("Cohort (group)", options=all_cols, index=all_cols.index(default_cohort))
n_active_col = st.sidebar.selectbox("Active (denominator)", options=all_cols, index=all_cols.index(default_n_active))
n_kept_col = st.sidebar.selectbox("Retained (successes)", options=all_cols, index=all_cols.index(default_n_kept))

# Clean types
df = df.copy()
df[period_col] = ensure_datetime(df[period_col])
df = df.dropna(subset=[period_col])
df[n_active_col] = pd.to_numeric(df[n_active_col], errors="coerce")
df[n_kept_col] = pd.to_numeric(df[n_kept_col], errors="coerce")
df = df.dropna(subset=[n_active_col, n_kept_col])

# Derived retention
df["retention"] = compute_retention(df, kept_col=n_kept_col, active_col=n_active_col)
st.write("**Derived columns:** `retention = n_retained / n_active_users` (clipped to [0,1]).")
st.dataframe(df[[period_col, cohort_col, n_active_col, n_kept_col, "retention"]].head(20), use_container_width=True)

# ------------------------------
# Sidebar: Train/Test Split
# ------------------------------
st.sidebar.header("3) Train/Test Split")
min_date = df[period_col].min()
max_date = df[period_col].max()
split_date = st.sidebar.date_input(
    "Train/Test split date",
    value=min_date + (max_date - min_date) // 2,
    min_value=min_date.date(),
    max_value=max_date.date()
)

# Partition
split_dt = pd.to_datetime(split_date)
train_df = df[df[period_col] <= split_dt].copy()
test_df = df[df[period_col] > split_dt].copy()
st.write(f"**Train rows:** {len(train_df)}  |  **Test rows:** {len(test_df)}")

# ------------------------------
# Exploratory Plots (match notebook) — moved AFTER split
# ------------------------------
st.header("Exploratory Views")

# Heatmap: Retention by Cohort x Period (train only)
if not train_df.empty:
    heat_df = (
        train_df.assign(
            cohort_fmt=lambda d: (
                d[cohort_col].dt.strftime("%Y-%m")
                if np.issubdtype(train_df[cohort_col].dtype, np.datetime64)
                else d[cohort_col].astype(str)
            ),
            period_fmt=lambda d: d[period_col].dt.strftime("%Y-%m"),
        )
        .query("retention.notnull()")
        .filter(items=["cohort_fmt", "period_fmt", "retention"])
        .pivot(index="cohort_fmt", columns="period_fmt", values="retention")
    )
    fig_h, ax_h = plt.subplots(figsize=(17, 9))
    fmt = lambda y, _: f"{y:0.0%}"
    sns.heatmap(
        heat_df,
        cmap="viridis_r",
        linewidths=0.2,
        linecolor="black",
        annot=True,
        fmt="0.0%",
        cbar_kws={"format": mtick.FuncFormatter(fmt)},
        ax=ax_h,
    )
    ax_h.set_title("Retention by Cohort and Period (Train)")
    st.pyplot(fig_h)

# Line plot: Retention by Cohort over Period (train)
if not train_df.empty:
    fig_l, ax_l = plt.subplots(figsize=(12, 7))
    plot_df = train_df.assign(
        cohort_fmt=lambda d: (
            d[cohort_col].dt.strftime("%Y-%m")
            if np.issubdtype(train_df[cohort_col].dtype, np.datetime64)
            else d[cohort_col].astype(str)
        )
    )
    sns.lineplot(
        x=period_col, y="retention", hue="cohort_fmt",
        palette="viridis_r", alpha=0.8, data=plot_df, ax=ax_l
    )
    ax_l.legend(title="cohort", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7.5)
    ax_l.set(title="Retention by Cohort and Period")
    st.pyplot(fig_l)

# ------------------------------
# Sidebar: Model Settings
# ------------------------------
st.sidebar.header("4) Model Settings (BART)")
m_trees = st.sidebar.slider("Number of trees (m)", min_value=50, max_value=200, value=100, step=10)
draws = st.sidebar.slider("Draws", min_value=500, max_value=3000, value=1000, step=100)
tune = st.sidebar.slider("Tune", min_value=500, max_value=3000, value=1000, step=100)
chains = st.sidebar.slider("Chains", min_value=2, max_value=6, value=2, step=1)
target_accept = st.sidebar.slider("target_accept", min_value=0.80, max_value=0.99, value=0.9, step=0.01)
hdi_prob = st.sidebar.slider("HDI probability", min_value=0.50, max_value=0.99, value=0.94, step=0.01)

# ------------------------------
# Build Design Matrices
# ------------------------------
train_df = train_df.sort_values([cohort_col, period_col])
test_df = test_df.sort_values([cohort_col, period_col])

X_train, t_train, c_train, le = prepare_design_matrix(train_df, period_col, cohort_col)
X_test, t_test, c_test, _ = prepare_design_matrix(test_df, period_col, cohort_col)

# logit of observed retention for training
eps = 1e-6
train_retention = np.clip(train_df["retention"].values, eps, 1 - eps)
train_retention_logit = np.log(train_retention / (1 - train_retention))

n_active_train = train_df[n_active_col].values.astype(int)
n_kept_train = train_df[n_kept_col].values.astype(int)

# ------------------------------
# Model (with MutableData/Data compatibility)
# ------------------------------
st.header("Model Fit")

DataVar = pm.MutableData if hasattr(pm, "MutableData") else pm.Data

coords = {
    "obs": np.arange(len(train_df)),
    "features": np.arange(X_train.shape[1]),
}

with st.spinner("Sampling..."):
    with pm.Model(coords=coords) as model:
        x = DataVar("x", X_train, dims=("obs", "features"))
        n_obs = DataVar("n", n_active_train, dims="obs")

        mu = pmb.BART(
            name="mu",
            X=x,
            Y=train_retention_logit,
            m=m_trees,
            response="mix",
            split_rules=[ContinuousSplitRule(), SubsetSplitRule()],
            dims="obs",
        )

        p = pm.Deterministic("p", pm.math.invlogit(mu), dims="obs")
        p = pt.clip(p, eps, 1 - eps)

        retained = pm.Binomial("retained", n=n_obs, p=p, observed=n_kept_train, dims="obs")

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=target_accept,
            progressbar=True, random_seed=42
        )

st.success("Sampling complete.")

# Diagnostics
st.subheader("Diagnostics")
st.write(az.summary(idata, var_names=["~retained"]).round(3))

# ESS ECDF and R-hat histogram for 'mu'
fig_d, ax_d = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), layout="constrained")
ess_data = az.ess(idata, var_names=["mu"], method="bulk")
rhat_data = az.rhat(idata, var_names=["mu"])

ess_values = ess_data["mu"].values.flatten()
rhat_values = rhat_data["mu"].values.flatten()

# ESS ECDF
ess_sorted = np.sort(ess_values)
ess_ecdf = np.arange(1, len(ess_sorted) + 1) / len(ess_sorted)
ax_d[0].plot(ess_sorted, ess_ecdf, linewidth=2)
ax_d[0].set(title="ESS ECDF (mu)", xlabel="ESS", ylabel="ECDF")

# R-hat histogram
ax_d[1].hist(rhat_values, bins=30, edgecolor="black", alpha=0.7)
ax_d[1].axvline(1.01, color="red", linestyle="--", label="1.01")
ax_d[1].set(title="R-hat Histogram (mu)", xlabel="R-hat", ylabel="Frequency")
ax_d[1].legend()

fig_d.suptitle("Diagnostics of the BART Component", y=1.06, fontsize=16)
st.pyplot(fig_d)

# Posterior Predictive Check (cumulative)
with model:
    posterior_predictive = pm.sample_posterior_predictive(idata, random_seed=42, progressbar=False)
ax_ppc = az.plot_ppc(
    data=posterior_predictive, kind="cumulative", observed_rug=True, random_seed=42
)
ax_ppc.set(
    title="Posterior Predictive Check",
    xscale="log",
    xlabel="likelihood (n_active_users) - log scale",
)
st.pyplot(ax_ppc.figure)

# ------------------------------
# Posterior predictions for Train & Test
# ------------------------------
# Train: use posterior 'p' directly
p_train = az.extract(idata, var_names=["p"]).to_array().squeeze().values

# Test: update data and predict 'p' via posterior predictive
with model:
    pm.set_data({"x": X_test})
    if len(test_df):
        pp_test = pm.sample_posterior_predictive(
            idata, var_names=["p"], predictions=True, extend_inferencedata=False, progressbar=False
        )
        p_test = pp_test["p"]
    else:
        p_test = np.empty((p_train.shape[0], 0))

# ------------------------------
# Build plot data by cohort and period
# ------------------------------
def build_plot_frame(df_sub, p_samples, label: str):
    if p_samples.ndim == 3:
        samples_flat = p_samples.reshape(-1, p_samples.shape[-1])
    else:
        samples_flat = p_samples
    if samples_flat.size == 0:
        return pd.DataFrame(columns=[period_col, cohort_col, "p_mean", "p_low", "p_high", "phase"])
    p_mean = samples_flat.mean(axis=0)
    hdi = az.hdi(samples_flat, hdi_prob=hdi_prob)
    plot_df = df_sub[[period_col, cohort_col]].copy()
    plot_df["p_mean"] = p_mean
    plot_df["p_low"] = hdi[:, 0]
    plot_df["p_high"] = hdi[:, 1]
    plot_df["phase"] = label
    return plot_df

plot_train = build_plot_frame(train_df, p_train, "train")
plot_test = build_plot_frame(test_df, p_test, "test")
plot_all = pd.concat([plot_train, plot_test], ignore_index=True)

# ------------------------------
# Plots
# ------------------------------
st.header("Retention Predictions")
cohorts_list = list(plot_all[cohort_col].astype(str).unique())
selected_cohorts = st.multiselect(
    "Select cohorts to display",
    options=cohorts_list,
    default=cohorts_list[:min(4, len(cohorts_list))]
)

if selected_cohorts:
    ncols = 2
    nrows = int(np.ceil(len(selected_cohorts) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for i, cohort in enumerate(selected_cohorts):
        ax = axes[i // ncols, i % ncols]
        sub = plot_all[plot_all[cohort_col].astype(str) == str(cohort)].sort_values(period_col)
        train_sub = sub[sub["phase"] == "train"]
        if not train_sub.empty:
            line_ci(ax, train_sub[period_col], train_sub["p_mean"], train_sub["p_low"], train_sub["p_high"], label="train")
        test_sub = sub[sub["phase"] == "test"]
        if not test_sub.empty:
            line_ci(ax, test_sub[period_col], test_sub["p_mean"], test_sub["p_low"], test_sub["p_high"], label="test")
        ax.axvline(x=pd.to_datetime(split_dt), linestyle="--")
        ax.set_title(f"Cohort: {cohort}")
        ax.set_ylabel("Retention probability")
        ax.set_xlabel("Period")
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Select at least one cohort to visualize.")

# ------------------------------
# Export
# ------------------------------
st.header("Export")
csv_buf = StringIO()
plot_all.to_csv(csv_buf, index=False)
st.download_button(
    "Download predictions CSV",
    data=csv_buf.getvalue(),
    file_name="bart_retention_predictions.csv",
    mime="text/csv"
)

st.success("Done.")
