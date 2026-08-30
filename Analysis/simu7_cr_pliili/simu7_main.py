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

from CMR_IA.utils import wmse
from CMR_IA.fitting import _simu7_stats

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
df_study = pd.read_parquet("data/simu7_study.parquet")
df_test = pd.read_parquet("data/simu7_test.parquet")

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
params = cmr.load_params("7", params_path="data/7_260825_200-200.json", fixed_params={"nitems_in_accumulator": 96})
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu, f_in, f_dif = cmr.run_norm_cr_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
    if SAVERES:
        df_simu.to_parquet("data/simu7_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu7_result.parquet")
df_simu

# %% [markdown]
# ## Analysis

# %%
# Get the study list and study pos of response
sessions = np.unique(df_simu.session)
nlist = len(np.unique(df_simu.list))
resp_study_list = []
resp_study_pos = []
for sess in sessions:

    pres_words = df_study.loc[df_study.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy()
    pres_words = np.reshape(pres_words, (nlist, -1, 2))
    responses = df_simu.loc[df_simu.session == sess, "s_resp"]

    for r in responses:
        if r == -1 or r == -2:
            r_list = None
            r_pos = None
        else:
            r_list = np.where(pres_words == r)[0].item()
            r_pos = np.where(pres_words == r)[1].item()
        resp_study_list.append(r_list)
        resp_study_pos.append(r_pos)

df_simu["resp_study_list"] = resp_study_list
df_simu["resp_study_pos"] = resp_study_pos
df_simu

# %%
# Get intrusion type
df_simu["list_lag"] = df_simu["resp_study_list"] - df_simu["list"]
df_simu["pos_lag"] = df_simu["resp_study_pos"] - df_simu["study_pos"]


def which_intrusion(x):
    x_list_lag = x["list_lag"]
    x_pos_lag = x["pos_lag"]

    if np.isnan(x_list_lag):
        return "NoResp"
    elif x_list_lag == 0 and x_pos_lag == 0:
        return "Correct"
    elif x_list_lag < 0:
        return "PLI"
    elif x_list_lag == 0 and x_pos_lag != 0:
        return "ILI"
    else:
        return np.nan


df_simu["intrusion_type"] = df_simu.apply(lambda x: which_intrusion(x), axis=1)
df_simu["intrusion_type"] = pd.Categorical(df_simu["intrusion_type"], categories=["NoResp", "Correct", "PLI", "ILI"])
df_simu

# %%
# Clean list 1
df_simu = df_simu.query("list > 0").copy()
df_simu

# %%
# Check response types
df_simu.groupby("intrusion_type")["list"].count()

# %%
# Check no-response counts
df_simu.query("s_resp == -1").shape, df_simu.query("s_resp == -2").shape

# %% [markdown]
# ### Overall Prob

# %%
# Count responses by session and intrusion type
df_cnt = df_simu.groupby(["session", "intrusion_type"]).s_resp.count().to_frame(name="count").reset_index()

# %%
# Check correct rate
df_cnt_correct = df_cnt.query("intrusion_type == 'Correct'").copy()
df_cnt_correct["total"] = df_simu.groupby("session").test_item.count().tolist()
df_cnt_correct["p"] = df_cnt_correct["count"] / df_cnt_correct["total"]
df_cnt_correct

# %%
# Summarize correct rate
p_correct_mean = np.mean(df_cnt_correct["p"])
p_correct_mean

# %%
# Check ILI rate
df_cnt_ILI = df_cnt.query("intrusion_type == 'ILI'").copy()
df_cnt_ILI["total"] = df_simu.groupby("session").test_item.count().tolist()
df_cnt_ILI["p"] = df_cnt_ILI["count"] / df_cnt_ILI["total"]
df_cnt_ILI

# %%
# Summarize ILI rate
p_ILI_mean = np.mean(df_cnt_ILI["p"])
p_ILI_mean

# %%
# Check PLI rate
df_cnt_PLI = df_cnt.query("intrusion_type == 'PLI'").copy()
df_cnt_PLI["total"] = df_simu.groupby("session").test_item.count().tolist()
df_cnt_PLI["p"] = df_cnt_PLI["count"] / df_cnt_PLI["total"]
df_cnt_PLI

# %%
# Summarize PLI rate
p_PLI_mean = np.mean(df_cnt_PLI["p"])
p_PLI_mean

# %% [markdown]
# ### PLI

# %%
# Pick list > 5 and list_lag -5 to -1
df_PLI = df_simu.query("intrusion_type == 'PLI' and list > 5 and list_lag > -6").copy()
df_PLI["abs_list_lag"] = df_PLI["list_lag"].abs().astype(int)
df_PLI["abs_list_lag"] = pd.Categorical(df_PLI["abs_list_lag"], categories=[1, 2, 3, 4, 5], ordered=True)
df_PLI

# %%
# Session-wise, count PLI
df_PLI_sess = df_PLI.groupby(["session"]).test_item.count().to_frame(name="PLI_cnt_sess").reset_index()
df_PLI_sess

# %%
# Session-wise, count PLI by list_lag
df_PLI_sess_lag = df_PLI.groupby(["session", "abs_list_lag"]).test_item.count().to_frame(name="PLI_cnt").reset_index()
df_PLI_sess_lag

# %%
# Calculate PLI probability
df_PLI_sess_lag = pd.merge(df_PLI_sess_lag, df_PLI_sess, on="session")
df_PLI_sess_lag["PLI_prob"] = df_PLI_sess_lag["PLI_cnt"] / df_PLI_sess_lag["PLI_cnt_sess"]
df_PLI_sess_lag

# %%
# Plot PLI probability by list lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_PLI_sess_lag, x="abs_list_lag", y="PLI_prob", linewidth=2, marker="o", markersize=10, errorbar=None)
plt.ylim([0, 0.5])
plt.xlim([0.5, 5.5])
plt.xticks(ticks=np.arange(1, 6))
plt.xlabel("List Lag")
plt.ylabel("PLI Probablility")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu7_PLI.pdf")
plt.show()

# %% [markdown]
# ### ILI

# %%
# Pick out ILI
df_ILI = df_simu.query("intrusion_type == 'ILI'").copy()
df_ILI["pos_lag"] = df_ILI["pos_lag"].astype(int)
df_ILI["pos_lag"] = pd.Categorical(df_ILI["pos_lag"], categories=np.concatenate([np.arange(-11, 0), np.arange(1, 12)]), ordered=True)
df_ILI


# %%
# Session-wise, calculate ILI probability for each lag
def get_ILI_prob(df_tmp):

    # Get possible ILI count
    possible_ILI_cnt = {}
    for pair_pos in df_tmp.study_pos:
        l_bound = -pair_pos
        r_bound = 11 - pair_pos
        for i in np.arange(l_bound, r_bound + 1):
            if i in possible_ILI_cnt:
                possible_ILI_cnt[i] += 1
            else:
                possible_ILI_cnt[i] = 1

    # Get ILI count
    df_tmp_lag = df_tmp.groupby("pos_lag", observed=False)["test_item"].count().to_frame(name="ILI_cnt")

    # Merge possible ILI count
    df_tmp_lag["possible_ILI_cnt"] = df_tmp_lag.index.map(possible_ILI_cnt).astype(float)
    df_tmp_lag["ILI_prob"] = df_tmp_lag["ILI_cnt"] / df_tmp_lag["possible_ILI_cnt"]

    return df_tmp_lag


df_ILI_sess_lag = df_ILI.groupby("session").apply(get_ILI_prob, include_groups=False).reset_index()
df_ILI_sess_lag["pos_lag"] = df_ILI_sess_lag["pos_lag"].astype(int)
df_ILI_sess_lag = df_ILI_sess_lag.query("pos_lag > -6 and pos_lag < 6").copy()
df_ILI_sess_lag

# %%
# Plot ILI probability by position lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

df_ILI_sess_lag["pos_lag"] = df_ILI_sess_lag["pos_lag"].astype(int)
sns.lineplot(data=df_ILI_sess_lag.query("-6 < pos_lag < 0"), x="pos_lag", y="ILI_prob", linewidth=2, marker="o", markersize=10, color="C0", errorbar=None)
sns.lineplot(data=df_ILI_sess_lag.query("0 < pos_lag < 6"), x="pos_lag", y="ILI_prob", linewidth=2, marker="o", markersize=10, color="C0", errorbar=None)
plt.ylim([0, 0.3])
plt.xticks(ticks=np.arange(-5, 6))
plt.xlabel("Lag")
plt.ylabel("ILI Conditional Response Probablility")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu7_ILI.pdf")
plt.show()

# %% [markdown]
# ## Err Check

# %%
# Aggregate mean PLI and ILI probabilities by lag
lag_PLI_mean = df_PLI_sess_lag.groupby("abs_list_lag").PLI_prob.mean().values
df_ILI_sess_lag["pos_lag_int"] = df_ILI_sess_lag["pos_lag"].astype(int)  # avoid nan from category vairables
lag_ILI_mean = df_ILI_sess_lag.groupby("pos_lag_int").ILI_prob.mean().values
lag_PLI_mean, lag_ILI_mean

# %%
# Check PLI monotonicity
np.cov(np.arange(5), lag_PLI_mean)[0, 1] >= 0

# %%
# Check ILI monotonicity
np.any(np.diff(lag_ILI_mean[:5]) <= 0), np.any(np.diff(lag_ILI_mean[5:]) >= 0)

# %%
# Load ground truth
with open("data/simu7_gt.json") as f:
    gt = json.load(f)
p_correct_mean_gt = np.array(gt["p_correct_mean"])
p_correct_se_gt = np.array(gt["p_correct_se"])
p_PLI_mean_gt = np.array(gt["p_PLI_mean"])
p_PLI_se_gt = np.array(gt["p_PLI_se"])
p_ILI_mean_gt = np.array(gt["p_ILI_mean"])
p_ILI_se_gt = np.array(gt["p_ILI_se"])
lag_PLI_mean_gt = np.array(gt["lag_PLI_mean"])
lag_PLI_se_gt = np.array(gt["lag_PLI_se"])
lag_ILI_mean_gt = np.array(gt["lag_ILI_mean"])
lag_ILI_se_gt = np.array(gt["lag_ILI_se"])

# %%
# Inspect ground truth lag means
lag_PLI_mean_gt, lag_ILI_mean_gt

# %%
# Compute weighted mean squared error
wls_p_correct = wmse(p_correct_mean_gt, p_correct_mean, p_correct_se_gt)
wls_p_PLI = wmse(p_PLI_mean_gt, p_PLI_mean, p_PLI_se_gt)
wls_p_ILI = wmse(p_ILI_mean_gt, p_ILI_mean, p_ILI_se_gt)
wls_lag_PLI = wmse(lag_PLI_mean_gt, lag_PLI_mean, lag_PLI_se_gt) / len(lag_PLI_mean_gt)
wls_lag_ILI = wmse(lag_ILI_mean_gt, lag_ILI_mean, lag_ILI_se_gt) / len(lag_ILI_mean_gt)
err = wls_p_correct + wls_p_PLI + wls_p_ILI + wls_lag_PLI + wls_lag_ILI
err

# %%
# Print PLI and ILI lag WLS errors
wls_lag_PLI, wls_lag_ILI

# %%
# Verify fitting helper
df_simu_check = pd.read_parquet("data/simu7_result.parquet")
_, _, _, _, _, err = _simu7_stats(df_simu_check, df_study, gt)
err
