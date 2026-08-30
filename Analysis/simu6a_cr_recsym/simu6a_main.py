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
import scipy as sp
from matplotlib.lines import Line2D
import json

from CMR_IA.fitting import _simu6a_stats

cmr.analysis.setup_notebook()

SAVEFIG = False
RUNCMR = True
SAVERES = False
if SAVERES and not RUNCMR:
    print("Warning: SAVERES is ignored when RUNCMR is False; existing results are loaded instead.")

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load study and test data
df_study = pd.read_parquet("data/simu6a_study.parquet")
df_test = pd.read_parquet("data/simu6a_test.parquet")

# %%
# Inspect study data
df_study

# %%
# Inspect test data
df_test

# %%
# Load semantic matrix
sem_mat = np.load("../wordpools/ltp_FR_similarity_matrix.npy")

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("6a", params_path="data/6a_260817_200-200.json", fixed_params={"nitems_in_accumulator": 48})
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu, f_in, f_dif = cmr.run_norm_cr_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # Merge f_in info, for testing
    sessions = np.unique(df_simu.session)
    for sess in sessions:
        df_tmp = df_study.loc[df_study.session == sess]
        tmp1 = df_tmp.study_itemno1.to_numpy()
        tmp2 = df_tmp.study_itemno2.to_numpy()
        tmp = np.concatenate((tmp1, tmp2))
        tmp = np.sort(tmp)
        tmp_test = df_simu.loc[df_study.session == sess, "test_itemno"]
        tmp_corr = df_simu.loc[df_study.session == sess, "correct_ans"]
        testid = np.searchsorted(tmp, tmp_test)
        corrid = np.searchsorted(tmp, tmp_corr)
        df_simu.loc[df_simu.session == sess, "corr_fin"] = [f_dif[sess][i][id] for i, id in enumerate(corrid)]
        df_simu.loc[df_simu.session == sess, "omax_fin"] = [np.max(np.delete(f_dif[sess][i], id)) for i, id in enumerate(corrid)]

    if SAVERES:
        df_simu.to_parquet("data/simu6a_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu6a_result.parquet")
df_simu

# %%
np.sum(df_simu.s_resp == -2)

# %% [markdown]
# ## Analysis

# %%
# Clean first 2 lists
df_simu = df_simu.query("list > 1")
df_simu

# %%
# Plot csim and threshold by lag
cthresh_lag = df_simu.query("thresh != -1").groupby("lag").thresh.mean()
lag = np.unique(df_simu.lag.to_numpy())

plt.plot(lag, cthresh_lag, label="thresh")
plt.xlabel("Lag")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %% [markdown]
# ### f_IN

# %%
# Compute mean f_in and f_max by lag
df_fin = df_simu.groupby("lag")[["corr_fin", "omax_fin"]].mean().reset_index()
df_fin["dif"] = df_fin["corr_fin"] - df_fin["omax_fin"]
df_fin

# %%
# Plot corr_fin and omax_fin by lag
sns.lineplot(data=df_fin, x="lag", y="corr_fin", linewidth=2, marker="o", markersize=7, label="corr")
sns.lineplot(data=df_fin, x="lag", y="omax_fin", linewidth=2, marker="o", markersize=7, label="omax")
plt.xlabel("Test Lag")
plt.ylabel("f_dif")
plt.show()

# %%
# Plot f_in difference by lag
sns.lineplot(data=df_fin, x="lag", y="dif", linewidth=2, marker="o", markersize=7)
plt.xlabel("Test Lag")
plt.ylabel("f_in")
plt.show()

# %% [markdown]
# ### Yes rate

# %%
# Session-wise, calculate correct rate for each condition
df_sess_lag = df_simu.groupby(["session", "lag", "order"]).correct.mean().to_frame(name="correct_rate").reset_index()
df_sess_lag

# %%
# Collapse across sessions
df_lag = df_sess_lag.groupby(["lag", "order"]).correct_rate.mean().to_frame(name="correct_rate").reset_index()
df_lag

# %%
# Plot correct rate by lag and recall direction
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines["left"].set_bounds(0, 1)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_lag, x="lag", y="correct_rate", linewidth=2, marker="o", markersize=10, hue="order", palette="tab10")
plt.ylim([0, 1.05])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")

legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Forward"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="Backward"),
]
L = plt.legend(handles=legend_elements, title="Recall Direction", loc="upper right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu6a_recall.pdf")
plt.show()

# %% [markdown]
# ## Error Check

# %%
# Load ground truth and compute error
with open("data/simu6a_gt.json") as f:
    gt = json.load(f)
fw_gt = np.array(gt["fw"])
bw_gt = np.array(gt["bw"])

fw = df_lag.query("order == 1").correct_rate.values
bw = df_lag.query("order == 2").correct_rate.values
err = np.power(fw - fw_gt, 2).sum() + np.power(bw - bw_gt, 2).sum()
err

# %%
# Print simulated forward and backward recall rates
fw, bw

# %%
# Verify fitting helper
_, _, err = _simu6a_stats(df_simu, gt)
err
