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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import CMR_IA as cmr
import scipy as sp
from matplotlib.lines import Line2D
import json

from CMR_IA.utils import wmse

cmr.analysis.setup_notebook()

SAVEFIG = False
RUNCMR = False
SAVERES = False
if SAVERES and not RUNCMR:
    print("Warning: SAVERES is ignored when RUNCMR is False; existing results are loaded instead.")

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load study and test data
df_study = pd.read_parquet("data/simu2b_study.parquet")
df_test = pd.read_parquet("data/simu2b_test.parquet")

# %%
# Inspect study data
df_study

# %%
# Inspect test data
df_test

# %%
# Load semantic matrix
sem_mat = np.load("data/simu2b_smat.npy")

# %% [markdown]
# ## Run CMR-IA

# %%
# Read PSO results and set parameters
params = cmr.load_params("2b")
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu = cmr.run_norm_recog_multi_sess(params, df_study, df_test, sem_mat, design="Osth")
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
    if SAVERES:
        df_simu.to_parquet("data/simu2b_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu2b_result.parquet")
df_simu

# %%
# Plot csim and threshold by trial position
old_csim_pos = df_simu.query("type == 'intact'").groupby("trial").csim.mean()
old_pos = np.unique(df_simu.query("type == 'intact'").trial.to_numpy())
new_csim_pos = df_simu.query("type == 'rearranged'").groupby("trial").csim.mean()
new_pos = np.unique(df_simu.query("type == 'rearranged'").trial.to_numpy())
cthresh_pos = df_simu.groupby("trial").thresh.mean()
pos = np.unique(df_simu.trial.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %% [markdown]
# ## Analysis

# %%
# Add old/new and correct columns
df_simu["old"] = df_simu.apply(lambda x: 1 if x["type"] == "intact" else 0, axis=1)
df_simu["correct"] = df_simu["s_resp"] == df_simu["old"]
df_simu

# %% [markdown]
# ### Overall HR and FAR

# %%
# Get yes rate and compute HR/FAR
df_hrfar = df_simu.groupby(["session", "type"]).correct.mean().to_frame(name="yes_rate").reset_index()
df_hrfar = df_hrfar.pivot(index="session", columns="type", values="yes_rate").reset_index()
df_hrfar["hr"] = df_hrfar["intact"]
df_hrfar["far"] = 1 - df_hrfar["rearranged"]
df_hrfar

# %%
# Melt HR/FAR for plotting
df_hrfar_plot = pd.melt(df_hrfar, id_vars=["session"], value_vars=["hr", "far"], var_name="type", value_name="yes_rate")
df_hrfar_plot

# %%
# Plot HR and FAR bar chart
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.barplot(
    data=df_hrfar_plot,
    x="type",
    y="yes_rate",
    width=0.5,
    errorbar="se",
    ax=ax,
    err_kws={"lw": 2, "color": "black"},
    facecolor="C0",
    edgecolor="C0",
    lw=2,
)

ax.set_xticks(ticks=[0, 1], labels=["HR", "FAR"])
ax.set_yticks(ticks=np.arange(0, 1.1, 0.2))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

ax.spines[["right", "top"]].set_visible(False)
ax.set(xlabel=None, ylabel="P(yes)")

if SAVEFIG:
    ax.set(ylabel=None)
    plt.tick_params(labelleft=False, labelbottom=False)
    plt.savefig(f"figures/simu2b_hrfar.pdf")
plt.show()

# %% [markdown]
# ### Far with Lag

# %%
# Compute FAR by lag for lure items
df_lure = df_simu.query("type == 'rearranged'").copy()
df_farlag = df_lure.groupby(["session", "lag"]).correct.mean().to_frame(name="yes_rate").reset_index()
df_farlag["far"] = 1 - df_farlag["yes_rate"]
df_farlag

# %%
# Plot FAR by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.lineplot(
    data=df_farlag,
    x="lag",
    y="far",
    marker="o",
    markersize=10,
    errorbar="se",
    err_style="band",
    ax=ax,
    linewidth=2,
    linestyle="-",
)

ax.set_xticks(ticks=np.arange(1, 6))
ax.set_yticks(ticks=np.arange(0, 0.41, 0.1))
ax.set_xlim(0.8, 5.2)
ax.set_ylim(0, 0.4)
ax.set(xlabel="Lag", ylabel="FAR")
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.spines[["right", "top"]].set_visible(False)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig(f"figures/simu2b_lag.pdf")
plt.show()

# %%
# Compute mean csim by lag for lure items
df_farlag_csim = df_lure.groupby(["session", "lag"]).csim.mean().to_frame(name="csim").reset_index()
df_farlag_csim

# %%
# Compute lag-csim correlation per session
sessions = df_farlag_csim.session.unique()
corrs = []
for sess in sessions:
    df_sess = df_farlag_csim.query("session == @sess")
    lag = df_sess["lag"].to_numpy()
    csim = df_sess["csim"].to_numpy()
    # do correlation
    corr = sp.stats.pearsonr(lag, csim)
    corrs.append(corr)
np.mean(corrs), np.std(corrs)

# %%
# Plot mean csim by lag
sns.lineplot(
    data=df_farlag_csim,
    x="lag",
    y="csim",
    color="blue",
    marker="o",
    err_style="bars",
)
# plt.ylim(0, 0.1)
plt.xlim(0.8, 5.2)
plt.xticks(np.arange(1, 6))
# plt.yticks(np.arange(0, 0.11, 0.01))
plt.xlabel("Lag")
plt.ylabel("False alarm rate")
plt.show()

# %% [markdown]
# ## Error Check

# %%
# Load ground truth
with open("data/simu2b_gt.json") as f:
    gt = json.load(f)
hr_gt = np.array(gt["hr"])
hr_std_gt = np.array(gt["hr_std"])
far_gt = np.array(gt["far"])
far_std_gt = np.array(gt["far_std"])
hr_gt, hr_std_gt, far_gt, far_std_gt

# %%
# Extract simulated HR and FAR
hr = df_hrfar_plot.query("type == 'hr'").yes_rate.mean()
far = df_farlag.groupby("lag").far.mean().to_numpy()
hr, far

# %%
# Calculate error
err = 5 * wmse(hr_gt, hr, hr_std_gt) + wmse(far_gt, far, far_std_gt)
err

# %% [markdown]
# ## Slope Check

# %%
# Import regression tools for slope analysis
import statsmodels.formula.api as smf
from scipy import stats

# %%
# Inspect FAR by lag data
df_farlag

# %%
# Fit OLS slope of far ~ lag per session
sessions = df_farlag.session.unique()
slopes = []
for sess in sessions:
    df_tmp = df_farlag.query("session == @sess").copy()
    res = smf.ols(formula="far ~ lag", data=df_tmp).fit()
    slopes.append(res.params["lag"])

# %%
# Summarize slope distribution
np.mean(slopes), np.std(slopes), stats.sem(slopes)

# %%
# T test
t_stat, p_val = stats.ttest_1samp(slopes, 0)
t_stat, p_val
