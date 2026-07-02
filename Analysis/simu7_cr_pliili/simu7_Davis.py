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

# %% [markdown]
# Experiment 1 of Davis et al. (2008) (Probably)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %% [markdown]
# ## Load Data

# %%
# Set data paths and constants
study_path = "data/original_study/"
test_path = "data/original_test/"
files = os.listdir(study_path)  # same name in test path
nitems = 24
npairs = 12

# %%
# Load and process all subject data
df = pd.DataFrame()
for f in files:

    # Load subject data
    subjnum = int(f.lstrip("subj").rstrip(".mat"))
    study_mat = scipy.io.loadmat(os.path.join(study_path, f))["studymatrix"]
    test_mat = scipy.io.loadmat(os.path.join(test_path, f))["theinfomatrix"]
    nlists = study_mat.shape[-1]

    # Get item's list and pos
    study_items_flat = study_mat.flatten(order="F")
    list_ids = np.repeat(np.arange(1, nlists + 1), nitems)
    list_pos = np.tile(np.arange(1, npairs + 1), nlists * 2)
    item2list = {}
    item2pos = {}
    for i, item in enumerate(study_items_flat):
        item2list[item] = list_ids[i]
        item2pos[item] = list_pos[i]

    # Reshape test mat
    for i in range(nlists):
        if i == 0:
            test_2d = test_mat[:, :, 0]
        else:
            test_2d = np.concatenate((test_2d, test_mat[:, :, i]), axis=0)
    tmp_df = pd.DataFrame(test_2d)
    tmp_df.columns = ["test_item", "response", "pair_pos", "lag", "forward", "correct", "intrusion_type", "rt"]
    tmp_df["subjnum"] = subjnum
    tmp_df["list"] = np.repeat(np.arange(1, nlists + 1), npairs)

    # Get response type
    def get_intrusion_type(x):
        correctness = x["correct"]
        response = x["response"]
        if correctness == 1:
            return "Correct"
        else:
            if response == 99999 or response < 0 or (response not in study_items_flat) or response == x["test_item"]:
                return "Out"
            else:
                resp_list = item2list[response]
                if resp_list == x["list"]:
                    return "ILI"
                elif resp_list < x["list"]:
                    return "PLI"
                else:
                    return "Out"

    tmp_df["type"] = tmp_df.apply(get_intrusion_type, axis=1)

    # Get response pos
    def get_resp_pos(x):
        resp_type = x["type"]
        if resp_type == "Correct":
            return item2pos[x["response"]]
        elif resp_type == "ILI":
            return item2pos[x["response"]]
        else:
            return None

    tmp_df["resp_pos"] = tmp_df.apply(get_resp_pos, axis=1)

    # Get response list
    def get_resp_list(x):
        resp_type = x["type"]
        if resp_type == "Correct" or resp_type == "ILI":
            return x["list"]
        elif resp_type == "PLI":
            return item2list[x["response"]]
        else:
            return None

    tmp_df["resp_list"] = tmp_df.apply(get_resp_list, axis=1)

    df = pd.concat([df, tmp_df])
df

# %% [markdown]
# Note: lag is the lag between the previous test item and the current test item

# %%
# Compute pos_lag and list_lag
df["pos_lag"] = df["resp_pos"] - df["pair_pos"]
df["list_lag"] = df["resp_list"] - df["list"]
df

# %%
# Clean list 1
df = df.query("list > 1").copy()
df

# %%
# Check unique subjects - same as paper
np.unique(df.subjnum).shape

# %%
# Check list nums
df.groupby("subjnum")["list"].max()

# %%
# Check response types
df.groupby("type").response.count()

# %% [markdown]
# ## Overall Prob

# %%
# Count responses by subject and type
df_cnt = df.groupby(["subjnum", "type"]).response.count().unstack(fill_value=0).stack().to_frame(name="count").reset_index()

# %%
# Check correct rate
df_cnt_correct = df_cnt.query("type == 'Correct'").copy()
df_cnt_correct["total"] = df.groupby("subjnum").test_item.count().tolist()
df_cnt_correct["p"] = df_cnt_correct["count"] / df_cnt_correct["total"]
df_cnt_correct

# %%
# Summarize correct rate
p_correct_mean = np.mean(df_cnt_correct["p"])
p_correct_se = np.std(df_cnt_correct["p"])
p_correct_mean, p_correct_se

# %%
# Check ILI rate
df_cnt_ILI = df_cnt.query("type == 'ILI'").copy()
df_cnt_ILI["total"] = df.groupby("subjnum").test_item.count().tolist()
df_cnt_ILI["p"] = df_cnt_ILI["count"] / df_cnt_ILI["total"]
df_cnt_ILI

# %%
# Summarize ILI rate
p_ILI_mean = np.mean(df_cnt_ILI["p"])
p_ILI_se = np.std(df_cnt_ILI["p"])
p_ILI_mean, p_ILI_se

# %%
# Check PLI rate
df_cnt_PLI = df_cnt.query("type == 'PLI'").copy()
df_cnt_PLI["total"] = df.groupby("subjnum").test_item.count().tolist()
df_cnt_PLI["p"] = df_cnt_PLI["count"] / df_cnt_PLI["total"]
df_cnt_PLI

# %%
# Summarize PLI rate
p_PLI_mean = np.mean(df_cnt_PLI["p"])
p_PLI_se = np.std(df_cnt_PLI["p"])
p_PLI_mean, p_PLI_se

# %% [markdown]
# ## PLI

# %%
# Check all PLIs
df.query("type == 'PLI'")

# %%
# Pick list > 5 and list_lag -5 to -1
df_PLI = df.query("type == 'PLI' and list > 5 and list_lag > -6").copy()
df_PLI["abs_list_lag"] = df_PLI["list_lag"].abs().astype(int)
df_PLI["abs_list_lag"] = pd.Categorical(df_PLI["abs_list_lag"], categories=[1, 2, 3, 4, 5], ordered=True)
df_PLI

# %%
# Check unique subjects
len(np.unique(df_PLI.subjnum))

# %%
# Subject-wise, count PLI
df_PLI_sess = df_PLI.groupby(["subjnum"]).test_item.count().to_frame(name="PLI_cnt_sess").reset_index()
df_PLI_sess

# %%
# Subject-wise, count PLI by list_lag
df_PLI_subj_lag = df_PLI.groupby(["subjnum", "abs_list_lag"], observed=False).test_item.count().to_frame(name="PLI_cnt").reset_index()
df_PLI_subj_lag

# %%
# Calculate PLI probability
df_PLI_subj_lag = pd.merge(df_PLI_subj_lag, df_PLI_sess, on="subjnum")
df_PLI_subj_lag["PLI_prob"] = df_PLI_subj_lag["PLI_cnt"] / df_PLI_subj_lag["PLI_cnt_sess"]
df_PLI_subj_lag

# %%
# Plot PLI probability by list lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_PLI_subj_lag, x="abs_list_lag", y="PLI_prob", linewidth=2, marker="o", markersize=10, linestyle="--", errorbar=None)
plt.ylim([0, 0.5])
plt.xlim([0.5, 5.5])
plt.xticks(ticks=np.arange(1, 6))
plt.yticks(ticks=np.arange(0, 0.6, 0.1), labels=np.arange(0, 0.6, 0.1).round(decimals=2))
plt.xlabel("List Lag")
plt.ylabel("PLI Probablility")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu7_Davis_PLI.pdf")
plt.show()

# %% [markdown]
# ## ILI

# %%
# Exclude subjects with fewer than 4 ILI responses
df_ILI = df.query("type == 'ILI'").copy()
df_ILI_subjcnt = df_ILI.groupby("subjnum").response.count().to_frame(name="count").reset_index()
df_ILI_subjcnt = df_ILI_subjcnt.query("count >= 4").copy()
df_ILI = df_ILI[df_ILI.subjnum.isin(df_ILI_subjcnt.subjnum)].copy()
df_ILI["pos_lag"] = df_ILI["pos_lag"].astype(int)
df_ILI["pos_lag"] = pd.Categorical(df_ILI["pos_lag"], categories=np.concatenate([np.arange(-11, 0), np.arange(1, 12)]), ordered=True)
df_ILI

# %%
# Check unique subjects in ILI
len(np.unique(df_ILI.subjnum))


# %%
# Session-wise, calculate ILI probability for each lag
def get_ILI_prob(df_tmp):

    # get possible ILI count
    possible_ILI_cnt = {}
    for pair_pos in df_tmp.pair_pos:  # notice! 1 to 12
        l_bound = -pair_pos + 1
        r_bound = npairs - pair_pos
        for i in np.arange(l_bound, r_bound + 1):
            if i in possible_ILI_cnt:
                possible_ILI_cnt[i] += 1
            else:
                possible_ILI_cnt[i] = 1

    # get ILI count
    df_tmp_lag = df_tmp.groupby("pos_lag", observed=False)["test_item"].count().to_frame(name="ILI_cnt")

    # merge possible ILI count
    df_tmp_lag["possible_ILI_cnt"] = df_tmp_lag.index.map(possible_ILI_cnt).astype(float)
    df_tmp_lag["ILI_prob"] = df_tmp_lag["ILI_cnt"] / df_tmp_lag["possible_ILI_cnt"]

    return df_tmp_lag


df_ILI_subj_lag = df_ILI.groupby("subjnum", observed=False).apply(get_ILI_prob).reset_index()
df_ILI_subj_lag = df_ILI_subj_lag.query("pos_lag > -6 and pos_lag < 6").copy()
df_ILI_subj_lag["pos_lag_int"] = df_ILI_subj_lag["pos_lag"].astype(int)  # avoid nan from category vairables
df_ILI_subj_lag

# %%
# Plot ILI probability by position lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

sns.lineplot(data=df_ILI_subj_lag.query("-6 < pos_lag_int < 0"), x="pos_lag", y="ILI_prob", linewidth=2, marker="o", markersize=10, color="C0", linestyle="--", errorbar=None)
sns.lineplot(data=df_ILI_subj_lag.query("0 < pos_lag_int < 6"), x="pos_lag", y="ILI_prob", linewidth=2, marker="o", markersize=10, color="C0", linestyle="--", errorbar=None)
plt.ylim([0, 0.3])
plt.xticks(ticks=np.arange(-5, 6))
plt.xlabel("Lag")
plt.ylabel("ILI Conditional Response Probablility")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu7_Davis_ILI.pdf")
plt.show()

# %% [markdown]
# ## Save GT

# %%
# Aggregate PLI probabilities by lag
df_PLI_lag = df_PLI_subj_lag.groupby("abs_list_lag", observed=False)["PLI_prob"].agg(["mean", "std"]).reset_index()
df_PLI_lag

# %%
# Aggregate ILI probabilities by lag
df_ILI_lag = df_ILI_subj_lag.groupby("pos_lag_int")["ILI_prob"].agg(["mean", "std"]).reset_index()
df_ILI_lag

# %%
# Extract mean and SE arrays
lag_PLI_mean = df_PLI_lag["mean"].values
lag_PLI_se = df_PLI_lag["std"].values
lag_ILI_mean = df_ILI_lag["mean"].values
lag_ILI_se = df_ILI_lag["std"].values
lag_PLI_mean, lag_PLI_se, lag_ILI_mean, lag_ILI_se

# %%
# Save gt
if SAVEDATA:
    with open("data/simu7_gt.json", "w") as f:
        json.dump(
            {
                "p_correct_mean": np.array(p_correct_mean).tolist(),
                "p_correct_se": np.array(p_correct_se).tolist(),
                "p_PLI_mean": np.array(p_PLI_mean).tolist(),
                "p_PLI_se": np.array(p_PLI_se).tolist(),
                "p_ILI_mean": np.array(p_ILI_mean).tolist(),
                "p_ILI_se": np.array(p_ILI_se).tolist(),
                "lag_PLI_mean": np.array(lag_PLI_mean).tolist(),
                "lag_PLI_se": np.array(lag_PLI_se).tolist(),
                "lag_ILI_mean": np.array(lag_ILI_mean).tolist(),
                "lag_ILI_se": np.array(lag_ILI_se).tolist(),
            },
            f,
            indent=4,
        )
