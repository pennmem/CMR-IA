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
import json

from CMR_IA.fitting import make_boundary

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load study and test data
df_study = pd.read_parquet("data/simu5_study.parquet")
df_test = pd.read_parquet("data/simu5_test.parquet")
df_study = df_study.loc[df_study.session < 100]
df_test = df_test.loc[df_test.session < 100]

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
params = cmr.load_params("5", fixed_params={"nitems_in_accumulator": 48})
params

# %%
# Run model or load saved results
if SAVERES:
    df_simu, f_in, f_dif = cmr.run_norm_cr_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # Merge f_in info, for testing
    sessions = df_simu.session.to_numpy()
    for sess in sessions:
        df_tmp = df_study.loc[df_study.session == sess]
        tmp1 = df_tmp.study_itemno1.to_numpy()
        tmp2 = df_tmp.study_itemno2.to_numpy()
        tmp = np.concatenate((tmp1, tmp2))
        tmp = np.sort(tmp)
        this_df = df_simu.query(f"session=={sess}")
        testid = np.searchsorted(tmp, this_df.test_itemno)
        corrid = np.searchsorted(tmp, this_df.correct_ans)
        df_simu.loc[df_simu.session == sess, "corr_fin"] = [f_dif[sess][l][i] for l, i in enumerate(corrid)]
        df_simu.loc[df_simu.session == sess, "omax_fin"] = [np.delete(f_dif[sess][l], i).max() for l, i in enumerate(corrid)]

    df_simu.to_parquet("data/simu5_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu5_result.parquet")
df_simu

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### f_IN

# %%
# Compute mean f_in and f_max by lag
df_fin = df_simu.groupby("lag")[["corr_fin", "omax_fin"]].mean().reset_index()
df_fin["dif"] = df_fin["corr_fin"] - df_fin["omax_fin"]
df_fin

# %%
# Plot corr_fin and omax_fin by lag
sns.lineplot(data=df_fin, x="lag", y="corr_fin", linewidth=2, marker="o", markersize=7)
sns.lineplot(data=df_fin, x="lag", y="omax_fin", linewidth=2, marker="o", markersize=7)
plt.ylim([-0.6, 0])
plt.xlabel("Test Lag")
plt.ylabel("f_in")
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
# Session-wise, calculate correct rate for each lag
df_sess_lag = df_simu.groupby(["session", "lag"]).correct.mean().to_frame(name="correct_rate").reset_index()
df_sess_lag

# %%
# Collapse across sessions
df_lag = df_sess_lag.groupby("lag").correct_rate.mean().to_frame(name="correct_rate").reset_index()
df_lag

# %%
# Plot correct rate by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_lag, x="lag", y="correct_rate", linewidth=2, marker="o", markersize=10)
plt.ylim([0, 1])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu5_recall.pdf")
plt.show()

# %% [markdown]
# ## Error Check

# %%
# Load ground truth and compute WLS error
with open("data/simu5_gt.json") as f:
    gt = json.load(f)
hr_gt = np.array(gt["hr"])
sem_gt = np.array(gt["hr_std"])

hr = df_lag.correct_rate.to_numpy()
wls = np.sum((hr - hr_gt) ** 2 / sem_gt**2)
wls
