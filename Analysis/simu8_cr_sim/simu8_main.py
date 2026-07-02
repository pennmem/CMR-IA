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
from CMR_IA.fitting import _simu8_g1_stats, _simu8_g2_stats

cmr.analysis.setup_notebook()

SAVEFIG = True
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
# # Group 1

# %%
df_study_g1 = df_study.query("group == 1").copy()
df_test_g1 = df_test.query("group == 1").copy()
df_test_g1 = df_test_g1.rename(columns={"test_itemno1": "test_itemno", "test_item1": "test_item"})
df_test_g1.drop(columns=["test_itemno2", "test_item2"], inplace=True)

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("8", params_path="data/8_260630_200-200.json", fixed_params={"nitems_in_accumulator": 16, "ban_recall": np.arange(1, 17)})
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu_g1, f_in, f_dif = cmr.run_norm_cr_multi_sess(params, df_study_g1, df_test_g1, sem_mat)
    df_simu_g1 = df_simu_g1.merge(df_test_g1, on=["session", "list", "test_itemno"])
    if SAVERES:
        df_simu_g1.to_parquet("data/simu8_result_g1.parquet")
else:
    df_simu_g1 = pd.read_parquet("data/simu8_result_g1.parquet")
df_simu_g1

# %%
# Check omission count
df_simu_g1.query("s_resp == -2").shape

# %%
# Check no-response count
df_simu_g1.query("s_resp == -1").shape

# %% [markdown]
# ## Analysis

# %% [markdown]
# Not so suitable for a session-wise analysis. I just pool them together.

# %%
# Get correction flag
df_simu_g1["correct"] = df_simu_g1.s_resp == df_simu_g1.correct_ans
df_simu_g1

# %%
# Check correct rate
correct_rate = sum(df_simu_g1.correct) / len(df_simu_g1.correct)
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
df_simu_g1["distance"] = df_simu_g1.groupby("session").apply(get_distance).to_frame(name="distance").reset_index()["distance"]
df_simu_g1["neighbour"] = df_simu_g1.apply(lambda x: sum(x["distance"] < thresh), axis=1)
distance_lsts = df_simu_g1["distance"].to_list()
df_simu_g1.drop(columns=["distance"], inplace=True)
df_simu_g1["neighbour_group"] = df_simu_g1.apply(lambda x: 6 if x["neighbour"] == 7 else x["neighbour"], axis=1)
df_simu_g1

# %%
# Count items per neighbour group
df_simu_g1.groupby("neighbour_group").test_itemno.count()

# %%
# Get correct rate by neighbour group
df_neighbour_group = df_simu_g1.query("neighbour_group > 0").groupby("neighbour_group").correct.mean().reset_index()
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
    plt.savefig("figures/simu8_g1_correct-neighbor.pdf")
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


df_simu_g1["ILI"] = df_simu_g1.groupby("session").apply(get_ILI).to_frame(name="ILI").reset_index()["ILI"].to_list()
df_ILI = df_simu_g1.query("ILI == True").copy()
df_ILI

# %%
# Get name-face pair dict for each session
sess_name_face = {}
for sess in df_study_g1.session.unique():
    sess_name_face[sess] = df_study_g1.query(f"session == {sess}")[["study_itemno1", "study_itemno2"]].set_index("study_itemno2").to_dict()["study_itemno1"]

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
    plt.savefig("figures/simu8_g1_ILI-distance.pdf")
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
_, _, _, v_wls_neighbor, v_wls_ILI = _simu8_g1_stats(df_simu_g1, df_study_g1, face_distance, thresh, gt)
v_wls_neighbor, v_wls_ILI

# %% [markdown]
# # Group 2

# %%
df_study_g2 = df_study.query("group == 2").copy()
df_test_g2 = df_test.query("group == 2").copy()

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("8", params_path="data/8_260630_200-200.json", fixed_params={"learn_while_retrieving": True, "nitems_in_accumulator": 32, "ban_recall": np.arange(1, 17)})
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu_g2, f_in, f_dif = cmr.run_success_multi_sess(params, df_study_g2, df_test_g2, sem_mat, mode="Recog-CR", design="S1G3")
    df_simu_g2["test"] = df_test_g2["test"]
    df_simu_g2 = df_simu_g2.merge(df_test_g2, on=["session", "list", "test", "test_itemno1", "test_itemno2"])
    if SAVERES:
        df_simu_g2.to_parquet("data/simu8_result_g2.parquet")
else:
    df_simu_g2 = pd.read_parquet("data/simu8_result_g2.parquet")
df_simu_g2

# %% [markdown]
# ## Analysis

# %%
# Split recognition and cued recall responses
df_recog = df_simu_g2.query("test == 1").copy()
df_cr = df_simu_g2.query("test == 2").copy()

# %%
# Get name-face pair dict for each session
sess_name_face = {}
for sess in df_study_g2.session.unique():
    sess_name_face[sess] = df_study_g2.query(f"session == {sess}")[["study_itemno1", "study_itemno2"]].set_index("study_itemno2").to_dict()["study_itemno1"]

# %% [markdown]
# ### Neighborhood Effect

# %%
# Load face distance matrix and compute neighbour counts (all 16 faces studied every session)
face_distance = np.load("data/simu8_distance.npy")
thresh = 3.0
neighbour_count = ((face_distance < thresh) & (face_distance > 0)).sum(axis=1)

# Density group of each recognition probe face
df_recog["neighbour"] = df_recog["test_itemno1"].apply(lambda f: neighbour_count[f - 1])
df_recog["density"] = pd.cut(df_recog["neighbour"], [4, 6, 8, 10], labels=["low", "medium", "high"])
df_recog["yes"] = (df_recog["s_resp"] == 1).astype(int)
df_recog

# %%
# Hit rate and false alarm rate by density
df_hr = df_recog.query("correct_ans == 1").groupby("density", observed=True).yes.mean().reset_index(name="HR")
df_far = df_recog.query("correct_ans == 0").groupby("density", observed=True).yes.mean().reset_index(name="FAR")
df_hr, df_far

# %%
# Plot hit rate and false alarm rate by density (stacked, broken y-axis)
xpos = np.arange(3)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 7))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98, hspace=0.04)

sns.lineplot(x=xpos, y=df_hr["HR"].to_numpy(), ax=ax1, marker="o", color="C0", markersize=10, linewidth=2)
sns.lineplot(x=xpos, y=df_far["FAR"].to_numpy(), ax=ax2, marker="s", color="C0", markersize=10, linewidth=2)

ax1.set_ylim(0.73, 0.90)
ax2.set_ylim(0.20, 0.37)
for ax in (ax1, ax2):
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.spines.right.set_visible(False)
ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False)

# Slanted break marks between the two panels
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

ax2.set_xticks(xpos)
ax2.set_xticklabels(["Low", "Medium", "High"])
ax2.set_xlim(-0.5, 2.5)
ax1.set_ylabel("HR")
ax2.set_ylabel("FAR")
ax2.set_xlabel("Number of Neighbours Group")

if SAVEFIG:
    ax1.set_ylabel(None)
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    ax1.tick_params(labelleft=False)
    ax2.tick_params(labelleft=False)
    plt.savefig("figures/simu8_g2_hrfar-neighbor.pdf")
plt.show()

# %% [markdown]
# ### Probe Distance Effect

# %%
# Distance from each lure probe to the studied face of its (mismatched) name; targets sit at distance 0
def get_probe_distance(x):
    if x["correct_ans"] == 1:
        return 0.0
    resp_face = sess_name_face[x["session"]][x["test_itemno2"]]
    return face_distance[x["test_itemno1"] - 1, resp_face - 1]


lure_edges = [0.5, 1.5, 2.5, 3.5, 4.5]
lure_labels = ["1.5", "2.5", "3.5", "4.5"]
df_recog["probe_distance"] = df_recog.apply(get_probe_distance, axis=1)
df_recog["distance_bin"] = np.where(df_recog["correct_ans"] == 1, "Targets", pd.cut(df_recog["probe_distance"], lure_edges, labels=lure_labels).astype(object))
dbin_order = ["Targets"] + lure_labels
df_recog

# %%
# Yes rate by probe distance bin
df_yes_distance = df_recog.groupby("distance_bin").yes.mean().reindex(dbin_order).reset_index(name="yes_rate")
df_yes_distance

# %%
# Plot yes rate by probe distance
xpos = [0, 1.5, 2.5, 3.5, 4.5]
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_yes_distance, x=xpos, y="yes_rate", ax=ax, marker="o", color="C0", markersize=10, linewidth=0)
sns.lineplot(data=df_yes_distance.query("distance_bin != 'Targets'"), x=xpos[1:], y="yes_rate", ax=ax, marker=None, color="C0", markersize=10, linewidth=2, linestyle="-")
plt.xlim([-0.5, 5])
plt.ylim([0, 1])
plt.xticks(ticks=xpos, labels=df_yes_distance["distance_bin"])
plt.xlabel("Distance Bins")
plt.ylabel("Yes Rate")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu8_g2_yes-distance.pdf")
plt.show()

# %% [markdown]
# ### Final Cued Recall

# %%
# Correct flag and recalled-name distance for in-space responses (drop omissions and no-responses)
fin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
fin_labels = ["0", "1.5", "2.5", "3.5", "4.5"]
df_cr["correct"] = df_cr["s_resp"] == df_cr["correct_ans"]
df_recalled = df_cr.query("s_resp > 0").copy()
df_recalled["resp_face"] = df_recalled.apply(lambda x: x["test_itemno1"] if x["correct"] else sess_name_face[x["session"]][x["s_resp"]], axis=1)
df_recalled["resp_distance"] = df_recalled.apply(lambda x: face_distance[x["test_itemno1"] - 1, x["resp_face"] - 1], axis=1)
df_recalled["distance_bin"] = pd.cut(df_recalled["resp_distance"], fin_edges, labels=fin_labels)
df_recalled

# %%
# Count possible responses: distance from each cued face to all 16 faces (self = 0 = correct option)
recall_poss = pd.cut(face_distance[df_cr["test_itemno1"].to_numpy() - 1, :].ravel(), fin_edges, labels=fin_labels).value_counts()

# Conditional recall probability by distance bin
df_recall_distance = df_recalled.groupby("distance_bin", observed=False).size().to_frame(name="recall_cnt").reset_index()
df_recall_distance["recall_poss"] = df_recall_distance["distance_bin"].map(recall_poss)
df_recall_distance["recall_prob"] = df_recall_distance["recall_cnt"] / df_recall_distance["recall_poss"]
df_recall_distance

# %%
# Plot final cued recall probability by distance
xpos = [0, 1.5, 2.5, 3.5, 4.5]
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_recall_distance, x=xpos, y="recall_prob", ax=ax, marker="o", color="C0", markersize=10, linewidth=0)
sns.lineplot(data=df_recall_distance.query("distance_bin != '0'"), x=xpos[1:], y="recall_prob", ax=ax, marker=None, color="C0", markersize=10, linewidth=2, linestyle="-")
plt.xlim([-0.5, 5])
plt.ylim([0, 0.5])
plt.xticks(ticks=xpos, labels=["Targets", "1.5", "2.5", "3.5", "4.5"])
plt.xlabel("Distance Bins")
plt.ylabel("Probability of Recall")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu8_g2_cr-distance.pdf")
plt.show()

# %% [markdown]
# ## Err Check

# %%
# Load ground truth
with open("data/simu8_gt.json") as f:
    gt = json.load(f)
hr_mean_gt = np.array(gt["exp3_neighbor_hr_mean"])
hr_se_gt = np.array(gt["exp3_neighbor_hr_se"])
far_mean_gt = np.array(gt["exp3_neighbor_far_mean"])
far_se_gt = np.array(gt["exp3_neighbor_far_se"])
yesdist_mean_gt = np.array(gt["exp3_yesdist_mean"])
yesdist_se_gt = np.array(gt["exp3_yesdist_se"])
crdist_mean_gt = np.array(gt["exp3_crdist_mean"])
crdist_se_gt = np.array(gt["exp3_crdist_se"])

# %%
np.any(np.diff(df_hr["HR"].to_numpy()) > 0), np.any(np.diff(df_far["FAR"].to_numpy()) < 0)

# %%
# Compute weighted mean squared error
hr_mean = df_hr["HR"].to_numpy()
far_mean = df_far["FAR"].to_numpy()
yesdist_mean = df_yes_distance["yes_rate"].to_numpy()
crdist_mean = df_recall_distance["recall_prob"].to_numpy()
wls_hr = wmse(hr_mean_gt, hr_mean, hr_se_gt) / len(hr_mean_gt)
wls_far = wmse(far_mean_gt, far_mean, far_se_gt) / len(far_mean_gt)
wls_yesdist = wmse(yesdist_mean_gt, yesdist_mean, yesdist_se_gt) / len(yesdist_mean_gt)
wls_crdist = wmse(crdist_mean_gt, crdist_mean, crdist_se_gt) / len(crdist_mean_gt)
wls_hr, wls_far, wls_yesdist, wls_crdist

# %%
# Compute total error
err = wls_hr * 10 + wls_far * 10 + wls_yesdist + wls_crdist
err

# %%
# Verify fitting helper
_, _, _, _, v_wls_hr, v_wls_far, v_wls_yesdist, v_wls_crdist = _simu8_g2_stats(df_recog, df_cr, df_study_g2, face_distance, thresh, gt)
v_wls_hr, v_wls_far, v_wls_yesdist, v_wls_crdist

# %%
# g1 + g2 error
err = wls_neighbor + wls_ILI + wls_hr * 10 + wls_far * 10 + wls_yesdist + wls_crdist
err
