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
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import CMR_IA as cmr
import json

from CMR_IA.utils import wmse
from CMR_IA.fitting import _simu4_stats

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

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("4")
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu = cmr.run_norm_recog_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
    if SAVERES:
        df_simu.to_parquet("data/simu4_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu4_result.parquet")
df_simu

# %% [markdown]
# ## Analysis

# %%
# Check overall yes rate by old/new
df_simu.groupby(["old"]).s_resp.mean()

# %%
# Get word frequency quantile data
df_word = pd.read_parquet("data/simu4_word_freq.parquet")
df_word

# %%
# Check mean frequency of each group
freq_mean = df_word.groupby("quantile").freq.mean().to_numpy()
freq_mean = np.around(freq_mean, decimals=0)
freq_mean

# %% [markdown]
# ### Csim

# %%
# Check the recognition threshold for each group
s_mean = np.mean(sem_mat, axis=1)
c_vec_cal = params["c_s"] * s_mean + params["c_thresh_itm"]
df_word["c_vec"] = c_vec_cal
df_word.groupby("quantile").c_vec.mean()

# %%
# Separate hf and lf conditions
create_level = {0: "new lf", 1: "new hf", 2: "old lf", 3: "old hf"}
df_simu["hf"] = df_simu["quantile"] >= 5
df_simu["level"] = df_simu.apply(lambda x: create_level[x["old"] * 2 + x["hf"]], axis=1)
df_simu

# %%
# Plot the csim distribution
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(df_simu, x="csim", hue="level", alpha=0.5, ax=ax, stat="density")
plt.xlim(0, 0.1)
plt.show()

# %%
# Check the csim of each group
df_tmp = df_simu.groupby(["old", "quantile"]).csim.mean().to_frame().reset_index()
df_tmp = pd.pivot_table(data=df_tmp, values="csim", columns="old", index="quantile").reset_index()
df_tmp["True-False"] = df_tmp[True] - df_tmp[False]
# df_tmp["c_vec"] = df.groupby("quantile").c_vec.mean()
df_tmp["thresh"] = df_simu.groupby("quantile").thresh.mean()
df_tmp

# %%
# Plot csim and threshold by position
old_csim_pos = df_simu.query("old == True").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("old == True").position.to_numpy())
new_csim_pos = df_simu.query("old == False").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("old == False").position.to_numpy())
cthresh_pos = df_simu.groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.xlim(0, 200)
plt.show()

# %% [markdown]
# ### Yes Rate

# %%
# Session-wise, get yes rate for each condition
df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()
df_sess_q

# %%
# Collapse across session
df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()
df_q

# %%
# Format df for plotting
df_plot = pd.pivot_table(df_q, values="yes_rate", index="quantile", columns="old").reset_index()
df_plot.rename(columns={False: "far", True: "hr"}, inplace=True)
df_plot["freq_mean"] = freq_mean
df_plot

# %%
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
    plt.savefig("figures/simu4_WFE.pdf")
plt.show()

# %% [markdown]
# ## Error Check

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
# Inspect standard deviations
hr_std_gt, far_std_gt

# %%
# Get behavioral stats and compare with ground truth
hr = df_q.query("old == True")["yes_rate"].to_numpy()
far = df_q.query("old == False")["yes_rate"].to_numpy()
err = wmse(hr_gt, hr, hr_std_gt) + wmse(far_gt, far, far_std_gt)
err

# %%
# Verify fitting helper
_, _, err = _simu4_stats(df_simu, gt)
err
