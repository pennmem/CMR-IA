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

from CMR_IA.fitting import make_boundary

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load test data
df = pd.read_parquet("data/simu3_test.parquet")
df = df.loc[df.session < 300]
df

# %%
# Load semantic matrix
sem_mat = np.load("../wordpools/ltp_FR_similarity_matrix.npy")

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("3")
params

# %%
# Run model or load saved results
if SAVERES:
    df_simu = cmr.run_conti_recog_multi_sess(params, df, sem_mat, design="Hockley")
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])
    df_simu.to_parquet("data/simu3_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu3_result.parquet")
df_simu

# %% [markdown]
# ## Analysis

# %%
# Plot csim for single items by position
old_csim_pos = df_simu.query("type == 'single_old'").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("type == 'single_old'").position.to_numpy())
new_csim_pos = df_simu.query("type == 'single_new'").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("type == 'single_new'").position.to_numpy())
cthresh_pos = df_simu.query("type == 'single_new' or type == 'single_old'").groupby("position").thresh.mean()
pos = np.unique(df_simu.query("type == 'single_new' or type == 'single_old'").position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Plot csim for pairs by position
old_csim_pos = df_simu.query("type == 'pair_old'").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("type == 'pair_old'").position.to_numpy())
new_csim_pos = df_simu.query("type == 'pair_new'").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("type == 'pair_new'").position.to_numpy())
cthresh_pos = df_simu.query("type == 'pair_new' or type == 'pair_old'").groupby("position").thresh.mean()
pos = np.unique(df_simu.query("type == 'pair_new' or type == 'pair_old'").position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Check the csim of each condition
df_simu.groupby(["type", "lag"]).csim.mean()

# %%
# Check csim distribution
sns.histplot(
    data=df_simu,
    x="csim",
    hue="type",
    hue_order=["single_new", "single_old", "pair_new", "pair_old"],
    palette=[[0, 0, 1], [0.5, 0.5, 1], [1, 0, 0], [1, 0.5, 0.5]],
    stat="probability",
    binwidth=0.005,
    common_norm=False,
    edgecolor=None,
    alpha=0.5,
)
plt.show()

# %% [markdown]
# ### Yes Rate

# %%
# Session-wise, calculate the yes_rate for each condition
df_sess_laggp = df_simu.groupby(["session", "type", "lag"]).s_resp.agg(["count", "sum", "mean"]).reset_index()
df_sess_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
df_sess_laggp["yes_rate_adj"] = (df_sess_laggp["sum"] + 0.5) / (df_sess_laggp["count"] + 1)
df_sess_laggp["z_yes_rate"] = sp.stats.norm.ppf(df_sess_laggp["yes_rate_adj"])
df_sess_laggp

# %%
# Collapse across session to get hit rate and false alarm rate
df_laggp = df_sess_laggp.groupby(["type", "lag"]).yes_rate.mean().to_frame(name="yes_rate").reset_index()
df_laggp["no_rate"] = 1 - df_laggp["yes_rate"]
df_laggp

# %%
# Plot yes rate by lag and condition
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_laggp.loc[df_laggp.type == "single_old"], x="lag", y="yes_rate", linewidth=2, marker="o", markersize=10, label="I-Hits")
sns.lineplot(data=df_laggp.loc[df_laggp.type == "pair_old"], x="lag", y="yes_rate", linewidth=2, marker="^", markersize=10, label="A-Hits")
sns.lineplot(data=df_laggp.loc[df_laggp.type == "pair_new"], x="lag", y="no_rate", linewidth=2, marker="^", markersize=10, label="A-CRs")
plt.ylim([0.5, 1])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")
plt.xticks(ticks=np.arange(2, 18, 2))
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="I-Hits"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="^", markersize=10, linestyle="-", label="A-Hits"),
    Line2D([0], [0], color=sns.color_palette()[2], lw=2, marker="^", markersize=10, linestyle="-", label="A-CRs"),
]
plt.legend(handles=legend_elements)

if SAVEFIG:
    plt.tick_params(labelleft=False)
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu3_hr.pdf")
plt.show()

# %% [markdown]
# ### d-prime

# %%
# Session-wise, get dprime for item for each lag
df_item = df_sess_laggp.loc[df_sess_laggp.type.isin(["single_old", "single_new"])].copy()
df_item = pd.pivot_table(df_item, index=["session", "lag"], columns="type", values="z_yes_rate").reset_index()
df_item.rename(columns={"single_old": "z_hr", "single_new": "z_far"}, inplace=True)
df_item["dprime"] = df_item["z_hr"] - df_item["z_far"]
df_item

# %%
# Session-wise, get dprime for pair for each lag
df_pair = df_sess_laggp.loc[df_sess_laggp.type.isin(["pair_old", "pair_new"])].copy()
df_pair = pd.pivot_table(df_pair, index=["session", "lag"], columns="type", values="z_yes_rate").reset_index()
df_pair.rename(columns={"pair_old": "z_hr", "pair_new": "z_far"}, inplace=True)
df_pair["dprime"] = df_pair["z_hr"] - df_pair["z_far"]
df_pair

# %%
# Plot d-prime by lag for items and pairs
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_item, x="lag", y="dprime", linewidth=2, marker="o", markersize=10, label="Items", errorbar=None)
sns.lineplot(data=df_pair, x="lag", y="dprime", linewidth=2, marker="^", markersize=10, label="Pairs", errorbar=None)
plt.xlabel("Study-Test Lag")
plt.ylabel("$d^'$")
plt.ylim([0.5, 3])
plt.xticks(ticks=np.arange(2, 18, 2))

legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Items"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="^", markersize=10, linestyle="-", label="Pairs"),
]
plt.legend(handles=legend_elements)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu3_dprime.pdf")
plt.show()

# %% [markdown]
# ## Error Check

# %%
# Load ground truth
with open("data/simu3_gt.json") as f:
    gt = json.load(f)
I_hr_gt = np.array(gt["I_hr"])
I_far_gt = np.array(gt["I_far"])
A_hr_gt = np.array(gt["A_hr"])
A_cr_gt = np.array(gt["A_cr"])
I_dprime_gt = np.array(gt["I_dprime"])
A_dprime_gt = np.array(gt["A_dprime"])
A_far_gt = 1 - A_cr_gt
I_hr_gt, I_far_gt, A_hr_gt, A_far_gt, I_dprime_gt, A_dprime_gt

# %%
# Get the simulated vectors
I_hr = df_laggp.loc[df_laggp.type == "single_old", "yes_rate"].to_numpy()
I_far = np.mean(df_laggp.loc[df_laggp.type == "single_new", "yes_rate"].astype(float))
A_hr = df_laggp.loc[df_laggp.type == "pair_old", "yes_rate"].to_numpy()
A_far = df_laggp.loc[df_laggp.type == "pair_new", "yes_rate"].to_numpy()
I_hr, I_far, A_hr, A_far

# %%
# Calculate the error (MSE)
err = np.sum((I_hr - I_hr_gt) ** 2) + np.sum((A_hr - A_hr_gt) ** 2) + (I_far - I_far_gt) ** 2 * 5 + np.sum((A_far - A_far_gt) ** 2)
err

# %%
# Check monotonicity constraints
np.any(np.diff(I_hr) > 0), np.any(np.diff(A_hr) > 0), np.any(np.diff(A_far) > 0), np.any(I_hr < A_hr)
