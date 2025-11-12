
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

def hdi_df(samples: np.ndarray, hdi_prob: float = 0.94) -> pd.DataFrame:
    az_idata = az.convert_to_inference_data({"posterior_predictive": {"p": samples}})
    hdi = az.hdi(az_idata.posterior_predictive["p"], hdi_prob=hdi_prob).to_dataframe().reset_index()
    hdi = hdi.rename(columns={0: "hdi_low", 1: "hdi_high"})
    return hdi[["draw", "hdi_low", "hdi_high"]] if "draw" in hdi.columns else hdi

def line_ci(ax, x, y_mean, y_low, y_high, label=None):
    ax.plot(x, y_mean, label=label)
    ax.fill_between(x, y_low, y_high, alpha=0.2)


def prepare_design_matrix(df, period_col, cohort_col):
    # numeric time index (days since min period) for ContinuousSplitRule
    t = (df[period_col] - df[period_col].min()).dt.days.astype(float)
    # label-encode cohorts for SubsetSplitRule
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
    # Minimal synthetic example consistent with the notebook structure
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    cohorts = ["A", "B", "C"]
    rows = []
    for c in cohorts:
        base = 0.45 + 0.05 * (cohorts.index(c))   # different baseline per cohort
        trend = np.linspace(0, 0.1, len(dates))   # slow drift
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
split_date = st.sidebar.date_input("Train/Test split date", value=min_date + (max_date - min_date) // 2,
                                   min_value=min_date.date(), max_value=max_date.date())

# Partition
split_dt = pd.to_datetime(split_date)
train_df = df[df[period_col] <= split_dt].copy()
test_df = df[df[period_col] > split_dt].copy()

st.write(f"**Train rows:** {len(train_df)}  |  **Test rows:** {len(test_df)}")


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

# logit of observed retention for training (as in the notebook pattern)
eps = 1e-6
train_retention_logit = np.log(np.clip(train_df["retention"].values, eps, 1 - eps) / np.clip(1 - train_df["retention"].values, eps, 1))

n_active_train = train_df[n_active_col].values.astype(int)
n_kept_train = train_df[n_kept_col].values.astype(int)

# ------------------------------
# Model
# ------------------------------
st.header("Model Fit")
with st.spinner("Sampling..."):
    with pm.Model(coords={"obs": np.arange(len(train_df))}) as model:
        x = pm.MutableData("x", X_train, dims=("obs", "features"))
        pm.MutableData("n_active_users", n_active_train, dims="obs")

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

        retained = pm.Binomial("retained", n=pm.MutableData("n", n_active_train), p=p, observed=n_kept_train, dims="obs")

        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, progressbar=True, random_seed=42)

st.success("Sampling complete.")

# Diagnostics
st.subheader("Diagnostics")
st.write(az.summary(idata, var_names=["~retained"]).round(3))

# ------------------------------
# Posterior predictions for Train & Test
# ------------------------------
with model:
    pm.set_data({"x": X_train})
    post_p_train = pm.draws_to_array(idata.posterior["p"]).mean(axis=(0,1))
    # HDI for train
    post_p_train_draws = pm.draws_to_array(idata.posterior["p"])

with model:
    pm.set_data({"x": X_test})
    # BART posterior for mu → p for test
    mu_test_pp = pm.sample_posterior_predictive(idata, var_names=["mu"], predictions=True, extend_inferencedata=False, progressbar=False)
    mu_test_array = mu_test_pp["mu"]
    p_test_array = 1 / (1 + np.exp(-mu_test_array))
    p_test_array = np.clip(p_test_array, eps, 1 - eps)

# ------------------------------
# Build plot data by cohort and period
# ------------------------------
def build_plot_frame(df_sub, p_samples, label: str):
    # p_samples: (draws*chains, n_obs_sub) or (draws, chains, n_obs_sub)
    if p_samples.ndim == 3:
        samples_flat = p_samples.reshape(-1, p_samples.shape[-1])
    else:
        samples_flat = p_samples

    p_mean = samples_flat.mean(axis=0)
    hdi = az.hdi(samples_flat, hdi_prob=hdi_prob)
    plot_df = df_sub[[period_col, cohort_col]].copy()
    plot_df["p_mean"] = p_mean
    plot_df["p_low"] = hdi[:, 0]
    plot_df["p_high"] = hdi[:, 1]
    plot_df["phase"] = label
    return plot_df

plot_train = build_plot_frame(train_df, post_p_train_draws, "train")
plot_test = build_plot_frame(test_df, p_test_array, "test")
plot_all = pd.concat([plot_train, plot_test], ignore_index=True)

# ------------------------------
# Plots
# ------------------------------
st.header("Retention Predictions")
cohorts_list = list(plot_all[cohort_col].astype(str).unique())
selected_cohorts = st.multiselect("Select cohorts to display", options=cohorts_list, default=cohorts_list[:min(4, len(cohorts_list))])

if selected_cohorts:
    ncols = 2
    nrows = int(np.ceil(len(selected_cohorts) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6 * nrows), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    for i, cohort in enumerate(selected_cohorts):
        ax = axes[i // ncols, i % ncols]
        sub = plot_all[plot_all[cohort_col].astype(str) == str(cohort)].sort_values(period_col)

        # Train
        train_sub = sub[sub["phase"] == "train"]
        if not train_sub.empty:
            line_ci(ax, train_sub[period_col], train_sub["p_mean"], train_sub["p_low"], train_sub["p_high"], label="train")
        # Test
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
st.download_button("Download predictions CSV", data=csv_buf.getvalue(), file_name="bart_retention_predictions.csv", mime="text/csv")

st.success("Done.")
