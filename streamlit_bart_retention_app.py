import streamlit as st
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import pymc as pm
import pymc_bart as pmb
import pytensor.tensor as pt
import seaborn as sns
from scipy.special import expit, logit
from sklearn.preprocessing import LabelEncoder
from pymc_bart.split_rules import ContinuousSplitRule, SubsetSplitRule
from datetime import datetime

# Page config
st.set_page_config(page_title="BART Retention Analysis", layout="wide", page_icon="📊")

# Style settings
az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

# App title
st.title("📊 BART Retention Analysis")
st.markdown("""
This app performs Bayesian retention modeling using BART (Bayesian Additive Regression Trees).
Upload your own retention data or use the synthetic dataset.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Use Synthetic Data", "Upload Custom Data"]
)

# Session state initialization
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False

# Function to load data
@st.cache_data
def load_synthetic_data():
    """Load the synthetic retention dataset"""
    url = "https://raw.githubusercontent.com/juanitorduz/website_projects/master/data/retention_data.csv"
    df = pd.read_csv(url, parse_dates=["cohort", "period"])
    return df

def validate_uploaded_data(df):
    """Validate uploaded data has required columns"""
    required_cols = ['cohort', 'period', 'n_users', 'n_active_users', 'retention', 'cohort_age', 'age']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, "Data validated successfully!"

# Load or upload data
if data_source == "Use Synthetic Data":
    with st.spinner("Loading synthetic data..."):
        data_df = load_synthetic_data()
        st.sidebar.success("✅ Synthetic data loaded!")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should contain: cohort, period, n_users, n_active_users, retention, cohort_age, age"
    )
    
    if uploaded_file is not None:
        data_df = pd.read_csv(uploaded_file, parse_dates=["cohort", "period"])
        is_valid, msg = validate_uploaded_data(data_df)
        
        if is_valid:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)
            st.stop()
    else:
        st.info("👆 Please upload a CSV file to continue")
        st.stop()

# Data preview
with st.expander("📋 View Raw Data", expanded=False):
    st.dataframe(data_df.head(20), use_container_width=True)
    st.write(f"**Shape:** {data_df.shape[0]} rows × {data_df.shape[1]} columns")

# Model configuration
st.sidebar.header("🎯 Model Settings")

# Train/test split
split_date = st.sidebar.date_input(
    "Train/Test Split Date",
    value=pd.to_datetime("2022-11-01").date(),
    min_value=data_df['period'].min().date(),
    max_value=data_df['period'].max().date()
)
period_train_test_split = pd.to_datetime(split_date)

# BART parameters
n_trees = st.sidebar.slider("Number of Trees (m)", 50, 200, 100, 10)
n_draws = st.sidebar.slider("MCMC Draws", 500, 3000, 2000, 500)
n_chains = st.sidebar.slider("MCMC Chains", 2, 8, 5, 1)

# Fixed features as in original code
features = ["age", "cohort_age", "month"]
split_rules = [ContinuousSplitRule(), ContinuousSplitRule(), SubsetSplitRule()]

# Data preprocessing
@st.cache_data
def preprocess_data(df, split_date):
    """Preprocess data for modeling - exactly as in original code"""
    train_df = df.query("period <= @split_date")
    test_df = df.query("period > @split_date")
    test_df = test_df[test_df["cohort"].isin(train_df["cohort"].unique())]
    
    # Train data
    train_red_df = train_df.query("cohort_age > 0").reset_index(drop=True)
    train_red_df["month"] = train_red_df["period"].dt.strftime("%m").astype(int)
    
    # Test data
    test_red_df = test_df.query("cohort_age > 0")
    test_red_df = test_red_df[test_red_df["cohort"].isin(train_red_df["cohort"].unique())].reset_index(drop=True)
    test_red_df["month"] = test_red_df["period"].dt.strftime("%m").astype(int)
    
    return train_red_df, test_red_df

train_data_red_df, test_data_red_df = preprocess_data(data_df, period_train_test_split)

# EDA Section
st.header("📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Samples", len(train_data_red_df))
    st.metric("Test Samples", len(test_data_red_df))
with col2:
    st.metric("Unique Cohorts", train_data_red_df['cohort'].nunique())
    st.metric("Date Range", f"{data_df['period'].min().date()} to {data_df['period'].max().date()}")

# Retention heatmap (from original code)
if st.checkbox("Show Retention Heatmap", value=True):
    fig, ax = plt.subplots(figsize=(17, 9))
    fmt = lambda y, _: f"{y:0.0%}"
    
    (
        train_data_red_df.assign(
            cohort=lambda df: df["cohort"].dt.strftime("%Y-%m"),
            period=lambda df: df["period"].dt.strftime("%Y-%m"),
        )
        .query("cohort_age != 0")
        .filter(["cohort", "period", "retention"])
        .pivot(index="cohort", columns="period", values="retention")
        .pipe(
            (sns.heatmap, "data"),
            cmap="viridis_r",
            linewidths=0.2,
            linecolor="black",
            annot=True,
            fmt="0.0%",
            cbar_kws={"format": mtick.FuncFormatter(fmt)},
            ax=ax,
        )
    )
    ax.set_title("Retention by Cohort and Period")
    st.pyplot(fig)
    plt.close()

# Retention trends (from original code)
if st.checkbox("Show Retention Trends", value=True):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        x="period",
        y="retention",
        hue="cohort",
        palette="viridis_r",
        alpha=0.8,
        data=train_data_red_df.query("cohort_age > 0").assign(
            cohort=lambda df: df["cohort"].dt.strftime("%Y-%m")
        ),
        ax=ax,
    )
    ax.legend(title="cohort", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7.5)
    ax.set(title="Retention by Cohort and Period")
    st.pyplot(fig)
    plt.close()

# Model fitting section
st.header("🤖 Model Training")

if st.button("🚀 Fit BART Model", type="primary"):
    with st.spinner("Fitting BART model... This may take several minutes."):
        try:
            # Prepare data exactly as in original code
            seed = sum(map(ord, "retention"))
            rng = np.random.default_rng(seed=seed)
            
            eps = np.finfo(float).eps
            train_obs_idx = train_data_red_df.index.to_numpy()
            train_n_users = train_data_red_df["n_users"].to_numpy()
            train_n_active_users = train_data_red_df["n_active_users"].to_numpy()
            train_retention = train_data_red_df["retention"].to_numpy()
            train_retention_logit = logit(train_retention + eps)
            
            train_cohort = train_data_red_df["cohort"].to_numpy()
            train_cohort_encoder = LabelEncoder()
            train_cohort_idx = train_cohort_encoder.fit_transform(train_cohort).flatten()
            
            train_period = train_data_red_df["period"].to_numpy()
            train_period_encoder = LabelEncoder()
            train_period_idx = train_period_encoder.fit_transform(train_period).flatten()
            
            x_train = train_data_red_df[features]
            
            # Build model exactly as in original code
            with pm.Model(coords={"feature": features}) as model:
                model.add_coord(name="obs", values=train_obs_idx)
                x = pm.Data(name="x", value=x_train, dims=("obs", "feature"))
                n_users = pm.Data(name="n_users", value=train_n_users, dims="obs")
                n_active_users = pm.Data(name="n_active_users", value=train_n_active_users, dims="obs")
                
                mu = pmb.BART(
                    name="mu",
                    X=x,
                    Y=train_retention_logit,
                    m=n_trees,
                    response="mix",
                    split_rules=split_rules,
                    dims="obs",
                )
                p = pm.Deterministic(name="p", var=pm.math.invlogit(mu), dims="obs")
                p = pt.switch(pt.eq(p, 0), eps, p)
                p = pt.switch(pt.eq(p, 1), 1 - eps, p)
                
                pm.Binomial(name="likelihood", n=n_users, p=p, observed=n_active_users, dims="obs")
            
            # Sample
            progress_bar = st.progress(0, text="Sampling from posterior...")
            with model:
                idata = pm.sample(draws=n_draws, chains=n_chains, random_seed=rng)
                progress_bar.progress(80, text="Generating posterior predictive...")
                posterior_predictive = pm.sample_posterior_predictive(trace=idata, random_seed=rng)
            
            progress_bar.progress(100, text="Complete!")
            
            # Store in session state
            st.session_state.model_fitted = True
            st.session_state.idata = idata
            st.session_state.posterior_predictive = posterior_predictive
            st.session_state.model = model
            st.session_state.train_data_red_df = train_data_red_df
            st.session_state.train_n_users = train_n_users
            st.session_state.train_cohort_encoder = train_cohort_encoder
            st.session_state.train_cohort_idx = train_cohort_idx
            st.session_state.train_period = train_period
            st.session_state.train_period_idx = train_period_idx
            st.session_state.train_cohort = train_cohort
            st.session_state.train_retention = train_retention
            st.session_state.x_train = x_train
            st.session_state.features = features
            st.session_state.eps = eps
            st.session_state.seed = seed
            st.session_state.rng = rng
            st.session_state.mu = mu
            st.session_state.test_data_red_df = test_data_red_df
            
            st.success("✅ Model fitted successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error fitting model: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Display results if model is fitted
if st.session_state.model_fitted:
    st.header("📊 Model Diagnostics")
    
    idata = st.session_state.idata
    posterior_predictive = st.session_state.posterior_predictive
    
    # ESS and Rhat (exactly as in original code)
    tab1, tab2, tab3 = st.tabs(["ESS & R-hat", "Posterior Predictive Check", "In-Sample Predictions"])
    
    with tab1:
        fig, ax = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 4), sharex=False, sharey=False, layout="constrained"
        )
        
        # Get ESS and R-hat values
        ess_data = az.ess(idata, var_names=["mu"], method="bulk")
        rhat_data = az.rhat(idata, var_names=["mu"])
        
        # Extract values
        ess_values = ess_data["mu"].values.flatten()
        rhat_values = rhat_data["mu"].values.flatten()
        
        # Plot ESS ECDF
        ess_sorted = np.sort(ess_values)
        ess_ecdf = np.arange(1, len(ess_sorted) + 1) / len(ess_sorted)
        ax[0].plot(ess_sorted, ess_ecdf, linewidth=2, color='C0')
        ax[0].axvline(400, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (400)')
        ax[0].set_xlabel("ESS (Bulk)", fontsize=11)
        ax[0].set_ylabel("Cumulative Probability", fontsize=11)
        ax[0].set_title("ESS Distribution (ECDF)")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Plot R-hat ECDF
        rhat_sorted = np.sort(rhat_values)
        rhat_ecdf = np.arange(1, len(rhat_sorted) + 1) / len(rhat_sorted)
        ax[1].plot(rhat_sorted, rhat_ecdf, linewidth=2, color='C1')
        ax[1].axvline(1.01, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (1.01)')
        ax[1].set_xlabel("R-hat", fontsize=11)
        ax[1].set_ylabel("Cumulative Probability", fontsize=11)
        ax[1].set_title("R-hat Distribution (ECDF)")
        
        # Fix R-hat axis
        ax[1].set_xlim(1.000, rhat_sorted.max() * 1.01)
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        fig.suptitle("Diagnostics of the BART Component", y=1.06, fontsize=16)
        st.pyplot(fig)
        plt.close()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min ESS", f"{ess_values.min():.0f}")
            st.metric("Mean ESS", f"{ess_values.mean():.0f}")
        with col2:
            st.metric("Max R-hat", f"{rhat_values.max():.4f}")
            st.metric("Mean R-hat", f"{rhat_values.mean():.4f}")
    
    with tab2:
        # Posterior Predictive Check (exactly as in original code)
        fig, ax = plt.subplots(figsize=(10, 6))
        az.plot_ppc(
            data=posterior_predictive,
            kind="cumulative",
            observed_rug=True,
            random_seed=st.session_state.seed,
            ax=ax
        )
        ax.set(
            title="Posterior Predictive Check",
            xscale="log",
            xlabel="likelihood (n_active_users) - log scale",
        )
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        # In-sample predictions (exactly as in original code)
        train_posterior_retention = (
            posterior_predictive.posterior_predictive / st.session_state.train_n_users[np.newaxis, None]
        )
        train_posterior_retention_mean = az.extract(
            data=train_posterior_retention, var_names=["likelihood"]
        ).mean("sample")
        
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.scatterplot(
            x="retention",
            y="posterior_retention_mean",
            data=st.session_state.train_data_red_df.assign(
                posterior_retention_mean=train_posterior_retention_mean
            ),
            hue="age",
            palette="viridis_r",
            size="n_users",
            ax=ax,
        )
        ax.axline(xy1=(0, 0), slope=1, color="black", linestyle="--", label="diagonal")
        ax.legend()
        ax.set(title="Posterior Predictive - Retention Mean")
        st.pyplot(fig)
        plt.close()
    
    # HDI Plots for subset of cohorts (exactly as in original code)
    st.header("🎯 In-Sample Retention HDI")
    
    train_retention_hdi = az.hdi(ary=train_posterior_retention)["likelihood"]
    
    def plot_train_retention_hdi_cohort(cohort_index, ax):
        mask = st.session_state.train_cohort_idx == cohort_index
        train_period = st.session_state.train_period
        train_period_idx = st.session_state.train_period_idx
        train_retention = st.session_state.train_retention
        train_cohort_encoder = st.session_state.train_cohort_encoder
        
        ax.fill_between(
            x=train_period[train_period_idx[mask]],
            y1=train_retention_hdi[mask, :][:, 0],
            y2=train_retention_hdi[mask, :][:, 1],
            alpha=0.3,
            color="C0",
            label="94% HDI (train)",
        )
        sns.lineplot(
            x=train_period[train_period_idx[mask]],
            y=train_retention[mask],
            color="C0",
            marker="o",
            label="observed (train)",
            ax=ax,
        )
        cohort_name = (
            pd.to_datetime(train_cohort_encoder.classes_[cohort_index]).date().isoformat()
        )
        ax.legend(loc="upper left")
        ax.set(title=f"Retention HDI - Cohort {cohort_name}")
        return ax
    
    cohort_index_to_plot = [0, 1, 5, 10, 15, 20, 25, 30]
    # Filter to available cohorts
    max_cohort_idx = st.session_state.train_cohort_encoder.classes_.shape[0] - 1
    cohort_index_to_plot = [idx for idx in cohort_index_to_plot if idx <= max_cohort_idx]
    
    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(cohort_index_to_plot) / 2)),
        ncols=2,
        figsize=(17, 11),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    
    for cohort_index, ax in zip(cohort_index_to_plot, axes.flatten()):
        plot_train_retention_hdi_cohort(cohort_index=cohort_index, ax=ax)
    
    # Hide extra subplots
    for idx in range(len(cohort_index_to_plot), len(axes.flatten())):
        axes.flatten()[idx].set_visible(False)
    
    fig.suptitle("In-Sample Retention HDI", y=1.03, fontsize=20, fontweight="bold")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close()
    
    # PDP Plots (exactly as in original code)
    st.header("📉 Partial Dependence Plots (PDP)")
    
    with st.spinner("Computing PDP plots..."):
        try:
            axes = pmb.plot_pdp(
                bartrv=st.session_state.mu,
                X=st.session_state.x_train,
                Y=st.session_state.train_retention,
                func=expit,
                xs_interval="insample",
                samples=1_000,
                grid="wide",
                color="C2",
                color_mean="C2",
                var_discrete=[2],
                figsize=(12, 7),
                random_seed=st.session_state.seed,
            )
            axes[0].set(ylim=(0, 0.2))
            plt.gcf().suptitle(
                "Partial Dependency Plots (PDP) - Retention",
                fontsize=16,
                y=1.02,
            )
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error(f"Error computing PDP: {str(e)}")
    
    # ICE Plots (exactly as in original code)
    st.header("❄️ Individual Conditional Expectation (ICE) Plots")
    
    with st.spinner("Computing ICE plots..."):
        try:
            axes = pmb.plot_ice(
                bartrv=st.session_state.mu,
                X=st.session_state.x_train,
                Y=st.session_state.train_retention,
                func=expit,
                centered=False,
                samples=200,
                instances=20,
                grid="wide",
                color="C2",
                color_mean="C2",
                var_discrete=[2],
                figsize=(12, 7),
                random_seed=st.session_state.seed,
            )
            axes[0].set(ylim=(0, 0.2))
            plt.gcf().suptitle(
                "Individual Conditional Expectation (ICE) Plots - Retention",
                fontsize=16,
                y=1.02,
            )
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.error(f"Error computing ICE: {str(e)}")
    
    # Variable Importance (exactly as in original code)
    st.header("🔍 Variable Importance")
    
    with st.spinner("Computing variable importance..."):
        try:
            # Compute variable importance
            vi_results = pmb.compute_variable_importance(
                idata=idata,
                bartrv=st.session_state.mu,
                X=st.session_state.x_train,
                random_seed=st.session_state.seed
            )
            
            # Get variable inclusion
            vi_inclusion_values, vi_inclusion_labels = pmb.get_variable_inclusion(
                idata=idata,
                X=st.session_state.x_train
            )
            
            # Create figure with 2 subplots
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Variable inclusion
            axes[0].plot(range(len(vi_inclusion_labels)), vi_inclusion_values, marker='o', linewidth=2, color='C0')
            axes[0].set_xticks(range(len(vi_inclusion_labels)))
            axes[0].set_xticklabels(vi_inclusion_labels)
            axes[0].set_ylabel("importance", fontsize=11)
            axes[0].set_xlabel("covariables", fontsize=11)
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: R² contribution
            plt.sca(axes[1])
            pmb.plot_variable_importance(
                vi_results=vi_results,
                labels=st.session_state.features,
                ax=axes[1]
            )
            
            fig.suptitle("Variable Importance", fontsize=16, y=0.995)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.warning(f"Could not compute variable importance: {str(e)}")
    
    # Out-of-Sample Predictions
    st.header("🔮 Out-of-Sample Predictions")
    
    if st.button("Generate Test Predictions"):
        with st.spinner("Generating out-of-sample predictions..."):
            try:
                # Prepare test data
                test_data_red_df = st.session_state.test_data_red_df
                test_obs_idx = test_data_red_df.index.to_numpy()
                test_n_users = test_data_red_df["n_users"].to_numpy()
                test_n_active_users = test_data_red_df["n_active_users"].to_numpy()
                test_retention = test_data_red_df["retention"].to_numpy()
                
                test_cohort = test_data_red_df["cohort"].to_numpy()
                test_cohort_idx = st.session_state.train_cohort_encoder.transform(test_cohort).flatten()
                
                x_test = test_data_red_df[st.session_state.features]
                
                # Out-of-sample predictions
                with st.session_state.model:
                    pm.set_data(
                        new_data={
                            "x": x_test,
                            "n_users": test_n_users,
                            "n_active_users": np.ones_like(test_n_active_users),
                        },
                        coords={"obs": test_obs_idx},
                    )
                    idata.extend(
                        pm.sample_posterior_predictive(
                            trace=idata,
                            var_names=["likelihood", "p", "mu"],
                            idata_kwargs={"coords": {"obs": test_obs_idx}},
                        )
                    )
                
                # Store test predictions
                st.session_state.test_predictions_ready = True
                st.session_state.test_data_red_df = test_data_red_df
                st.session_state.test_cohort_idx = test_cohort_idx
                st.session_state.test_retention = test_retention
                st.session_state.test_n_users = test_n_users
                
                st.success("✅ Test predictions generated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating test predictions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display test predictions if ready
    if hasattr(st.session_state, 'test_predictions_ready') and st.session_state.test_predictions_ready:
        st.subheader("Test Set Retention Predictions")
        
        test_posterior_retention = (
            idata.posterior_predictive["likelihood"] / st.session_state.test_n_users[np.newaxis, None]
        )
        test_retention_hdi = az.hdi(ary=test_posterior_retention)["likelihood"]
        
        def plot_test_retention_hdi_cohort(cohort_index, ax):
            mask = st.session_state.test_cohort_idx == cohort_index
            
            test_period_range = st.session_state.test_data_red_df.query(
                f"cohort == '{st.session_state.train_cohort_encoder.classes_[cohort_index]}'"
            )["period"]
            
            ax.fill_between(
                x=test_period_range,
                y1=test_retention_hdi[mask, :][:, 0],
                y2=test_retention_hdi[mask, :][:, 1],
                alpha=0.3,
                color="C1",
                label="94% HDI (test)",
            )
            sns.lineplot(
                x=test_period_range,
                y=st.session_state.test_retention[mask],
                color="C1",
                marker="o",
                label="observed (test)",
                ax=ax,
            )
            return ax
        
        # Combined train/test plots
        fig, axes = plt.subplots(
            nrows=len(cohort_index_to_plot),
            ncols=1,
            figsize=(15, 16),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        
        for cohort_index, ax in zip(cohort_index_to_plot, axes.flatten()):
            plot_train_retention_hdi_cohort(cohort_index=cohort_index, ax=ax)
            plot_test_retention_hdi_cohort(cohort_index=cohort_index, ax=ax)
            ax.axvline(
                x=pd.to_datetime(period_train_test_split),
                color="black",
                linestyle="--",
                label="train/test split",
            )
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        fig.suptitle("Retention Predictions", y=1.03, fontsize=20, fontweight="bold")
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown("""
**About**: This app uses Bayesian Additive Regression Trees (BART) for retention modeling.
Based on PyMC-BART implementation with full Bayesian inference.
""")
