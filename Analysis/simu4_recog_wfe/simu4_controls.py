# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: cmr
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import math
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
import time
import pprint
import seaborn as sns
import statsmodels.formula.api as smf
import CMR_IA as cmr
import scipy as sp
import json
import statsmodels.formula.api as smf
from scipy import stats

from CMR_IA.fitting import make_boundary, get_wmse

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load study and test data
df_study = pd.read_parquet("data/simu4_study.parquet")
df_test = pd.read_parquet("data/simu4_test.parquet")

# %%
# Inspect study data
df_study

# %%
# Inspect test data
df_test

# %%
# Load semantic matrix
sem_mat = np.load("data/simu4_smat.npy")

# %%
# Get word frequency quantile data
df = pd.read_parquet("data/simu4_word_freq.parquet")
df

# %%
# Check mean frequency of each group
freq_mean = df.groupby("quantile").freq.mean().to_numpy()
freq_mean = np.around(freq_mean, decimals=0)
freq_mean

# %%
# Load ground truth
with open("data/simu4_gt.json") as f:
    gt = json.load(f)
hr_gt = np.array(gt["hr"])
hr_std_gt = np.array(gt["hr_std"])
far_gt = np.array(gt["far"])
far_std_gt = np.array(gt["far_std"])
hr_gt, far_gt

# %%
for which in ["base", "attn", "shift"]:
    print(f"\n========== Start control experiment: {which} ==========")

    # Define parameters and load PSO results
    params = cmr.load_params(f"4{which}", fixed_params={"use_new_context": True, "use_flexible_thresh": True})

    # Run model or load saved results
    if SAVERES:
        df_simu = cmr.run_norm_recog_multi_sess(params, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
        df_simu.to_parquet(f"data/simu4_result_{which}.parquet")
    else:
        df_simu = pd.read_parquet(f"data/simu4_result_{which}.parquet")

    # Session-wise, get yes rate for each condition
    df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # Collapse across session
    df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

    # Format df for plotting
    df_plot = pd.pivot_table(df_q, values="yes_rate", index="quantile", columns="old").reset_index()
    df_plot.rename(columns={False: "far", True: "hr"}, inplace=True)
    df_plot["freq_mean"] = freq_mean

    # Plot HR and FAR by word frequency
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 9))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)
    fig.subplots_adjust(hspace=0.03)

    sns.lineplot(data=df_plot, y="hr", x="freq_mean", ax=ax1, marker="o", color="C0", markersize=10, linewidth=2)
    sns.lineplot(data=df_plot, y="far", x="freq_mean", ax=ax2, marker="s", color="C0", markersize=10, linewidth=2)

    ax1.set_ylim(0.77, 0.95)
    ax1.set_yticks(np.arange(0.80, 0.96, 0.05))
    ax2.set_ylim(0.1, 0.28)
    ax2.set_yticks(np.arange(0.1, 0.30, 0.05))
    ax1.tick_params(axis="y", direction="in")
    ax1.tick_params(axis="x", direction="in")
    ax2.tick_params(axis="y", direction="in")
    ax2.tick_params(axis="x", direction="in")
    plt.xscale("log")
    plt.xlim(5, 14000)

    ax1.spines.bottom.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
    ax2.minorticks_off()

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
    ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
    ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

    ax1.set_ylabel("HR")
    ax2.set_ylabel("FAR")
    ax2.set_xlabel("Word Frequency")

    if SAVEFIG:
        ax1.set_ylabel(None)
        ax1.set_xlabel(None)
        ax2.set_ylabel(None)
        ax2.set_xlabel(None)
        ax1.tick_params(labelleft=False)
        ax2.tick_params(labelleft=False)
        plt.savefig(f"figures/simu4_WFE_{which}.pdf")
    plt.show()

    # Error check
    hr = df_q.query("old == True")["yes_rate"].to_numpy()
    far = df_q.query("old == False")["yes_rate"].to_numpy()
    err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)
    print(f"Error: {err}")

    # Slope check
    # Add log frequency to session-wise data
    sessions = df_sess_q.session.unique()
    df_sess_q["freq_mean"] = df_sess_q["quantile"].apply(lambda x: freq_mean[x])
    df_sess_q["log_freq_mean"] = np.log(df_sess_q["freq_mean"])

    # Fit OLS slopes per session for HR and FAR
    slopes_hr, slopes_far = [], []
    for sess in sessions:
        # slope for HR
        df_tmp = df_sess_q.query("session == @sess and old == True").copy()
        res = smf.ols(formula="yes_rate ~ log_freq_mean", data=df_tmp).fit()
        slopes_hr.append(res.params["log_freq_mean"])

        # slope for FAR
        df_tmp = df_sess_q.query("session == @sess and old == False").copy()
        res = smf.ols(formula="yes_rate ~ log_freq_mean", data=df_tmp).fit()
        slopes_far.append(res.params["log_freq_mean"])

    # T test
    t_stat_hr, p_val_hr = stats.ttest_1samp(slopes_hr, 0)
    t_stat_far, p_val_far = stats.ttest_1samp(slopes_far, 0)
    print(f"HR slope: mean {np.mean(slopes_hr)}, sem {sem(slopes_hr)}, T {t_stat_hr}, p {p_val_hr}")
    print(f"FAR slope: mean {np.mean(slopes_far)}, sem {sem(slopes_far)}, T {t_stat_far}, p {p_val_far}")
