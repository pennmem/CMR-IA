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
from CMR_IA.fitting import _simu8_g1_stats

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
df_study = pd.read_parquet("data/simu8_study.parquet")
df_test = pd.read_parquet("data/simu8_test.parquet")

# %%
# Inspect study data
df_study

# %%
# Inspect test data
df_test

# %%
# Load semantic matrix
sem_mat = np.load("data/simu8_smat.npy")

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("8", fixed_params={"nitems_in_accumulator": 16, "ban_recall": np.arange(1, 17)})
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu, f_in, f_dif = cmr.run_norm_cr_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    if SAVERES:
        df_simu.to_parquet("data/simu8_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu8_result.parquet")
df_simu

# %%
# Check omission count
df_simu.query("s_resp == -2").shape

# %%
# Check no-response count
df_simu.query("s_resp == -1").shape

# %% [markdown]
# ## Analysis

# %% [markdown]
# Not so suitable for a session-wise analysis. I just pool them together.

# %%
# Get correction flag
df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
df_simu

# %%
# Check correct rate
correct_rate = sum(df_simu.correct) / len(df_simu.correct)
correct_rate

# %% [markdown]
# ### Neighborhood Effect

# %%
# Load face distance matrix and compute neighbour counts
face_distance = np.load("data/simu8_distance.npy")
thresh = 3.0


def get_distance(df_tmp):
    faces = np.unique(df_tmp.test_itemno)
    face_dist = {}
    for face in faces:
        this_dist = []
        for other_face in faces:
            if face != other_face:
                this_dist.append(face_distance[face - 1, other_face - 1])
        this_dist = np.array(this_dist)
        face_dist[face] = this_dist
    y = df_tmp.apply(lambda x: face_dist[x["test_itemno"]], axis=1)
    return y


# Get number of neighbours by distance
df_simu["distance"] = df_simu.groupby("session").apply(get_distance).to_frame(name="distance").reset_index()["distance"]
df_simu["neighbour"] = df_simu.apply(lambda x: sum(x["distance"] < thresh), axis=1)
distance_lsts = df_simu["distance"].to_list()
df_simu.drop(columns=["distance"], inplace=True)
df_simu["neighbour_group"] = df_simu.apply(lambda x: 6 if x["neighbour"] == 7 else x["neighbour"], axis=1)
df_simu

# %%
# Count items per neighbour group
df_simu.groupby("neighbour_group").test_itemno.count()

# %%
# Get correct rate by neighbour group
df_neighbour_group = df_simu.query("neighbour_group > 0").groupby("neighbour_group").correct.mean().reset_index()
df_neighbour_group

# %%
# Plot neighbourhood effect
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(
    data=df_neighbour_group,
    x="neighbour_group",
    y="correct",
    linewidth=2,
    marker="o",
    markersize=10,
    label=str(thresh),
)
plt.ylim([0.4, 1])
plt.xlim([0.5, 6.5])
plt.xticks(ticks=np.arange(1, 7), labels=["1", "2", "3", "4", "5", "6-7"])
plt.xlabel("Number of Neighbours")
plt.ylabel("P(Correct)")
plt.legend().set_visible(False)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu8_correct-neighbor.pdf")
plt.show()


# %% [markdown]
# ### ILI

# %%
# Detect ILI responses
def get_ILI(df_tmp):
    resp_names = df_tmp["s_resp"].values
    study_names = df_tmp["correct_ans"].values  # all correct answers are all studied names
    is_studied = np.isin(resp_names, study_names)
    is_incorrect = df_tmp["correct"] == False
    is_ILI = is_studied & is_incorrect
    return is_ILI


df_simu["ILI"] = df_simu.groupby("session").apply(get_ILI).to_frame(name="ILI").reset_index()["ILI"].to_list()
df_ILI = df_simu.query("ILI == True").copy()
df_ILI

# %%
# Get name-face pair dict for each session
sess_name_face = {}
for sess in df_study.session.unique():
    sess_name_face[sess] = df_study.query(f"session == {sess}")[["study_itemno1", "study_itemno2"]].set_index("study_itemno2").to_dict()["study_itemno1"]

# %%
# Get distance between ILI and correct faces
df_ILI["resp_face"] = df_ILI.apply(lambda x: sess_name_face[x["session"]][x["s_resp"]], axis=1)
df_ILI["resp_corr_distance"] = df_ILI.apply(lambda x: face_distance[x["test_itemno"] - 1, x["resp_face"] - 1], axis=1)
df_ILI["distance_bin"] = df_ILI.apply(lambda x: str(0.5 * (x["resp_corr_distance"] // 0.5 + 1)) if x["resp_corr_distance"] < 3.5 else ">3.5", axis=1)
df_ILI["distance_bin"] = pd.Categorical(df_ILI["distance_bin"], categories=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", ">3.5"], ordered=True)
df_ILI

# %%
# Count possible ILI from all distances
distance_cnt = {}
for lst in distance_lsts:
    for d in lst:
        d_group = str(0.5 * (d // 0.5 + 1)) if d < 3.5 else ">3.5"
        if d_group in distance_cnt:
            distance_cnt[d_group] += 1
        else:
            distance_cnt[d_group] = 1

# %%
# Get ILI probability by distance bin
df_ILI_distance = df_ILI.groupby("distance_bin")["test_itemno"].count().to_frame(name="ILI_cnt").reset_index()
df_ILI_distance["ILI_poss"] = df_ILI_distance.apply(lambda x: distance_cnt[x["distance_bin"]], axis=1)
df_ILI_distance["ILI_prob"] = df_ILI_distance["ILI_cnt"] / df_ILI_distance["ILI_poss"]
df_ILI_distance

# %%
# Plot ILI by distance bin
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_ILI_distance, x="distance_bin", y="ILI_prob", linewidth=2, marker="o", markersize=10)
plt.ylim([0, 0.25])
plt.xlim([-0.5, 6.5])
plt.xlabel("Distance Bins")
plt.ylabel("ILI Conditional Response Probability")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu8_ILI-distance.pdf")
plt.show()

# %% [markdown]
# ## Err Check

# %%
# Load ground truth
with open("data/simu8_gt.json") as f:
    gt = json.load(f)
neighbor_mean_gt = np.array(gt["exp1_neighbor_mean"])
neighbor_se_gt = np.array(gt["exp1_neighbor_se"])
ILI_mean_gt = np.array(gt["exp1_ILI_mean"])
ILI_se_gt = np.array(gt["exp1_ILI_se"])

# %%
# Compute weighted mean squared error
neighbor_mean = df_neighbour_group["correct"].to_numpy()
ILI_mean = df_ILI_distance["ILI_prob"].to_numpy()
wls_neighbor = wmse(neighbor_mean_gt, neighbor_mean, neighbor_se_gt) / len(neighbor_mean_gt)
wls_ILI = wmse(ILI_mean_gt, ILI_mean, ILI_se_gt) / len(ILI_mean_gt)
wls_neighbor, wls_ILI

# %%
# Compute total error
err = wls_neighbor + wls_ILI
err

# %%
# Verify fitting helper
_, _, _, v_wls_neighbor, v_wls_ILI = _simu8_g1_stats(df_simu, df_study, face_distance, thresh, gt)
v_wls_neighbor, v_wls_ILI
