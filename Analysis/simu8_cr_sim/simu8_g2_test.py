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

from CMR_IA.fitting import make_boundary, get_wmse

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False
SAVERES2 = True

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
# # Group 2

# %%
df_study_g2 = df_study.query("group == 2").copy()
df_test_g2 = df_test.query("group == 2").copy()
df_study_g2 = df_study_g2.query("session < 2000").copy()
df_test_g2 = df_test_g2.query("session < 2000").copy()

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params_path = "/Users/bei/BeiWorld/Research/2022CMRIA/CMR_IA/Analysis/simuS1_recog_cr/data/simuS1_params.json"
params = cmr.load_params("S1", params_path=params_path, fixed_params={"learn_while_retrieving": True, "nitems_in_accumulator": 32, "ban_recall": np.arange(1, 17)})
params.update(beta_cue=0.5, s_fc=0.5)
params

# %%
# Run model or load saved results
if SAVERES2:
    df_simu_g2, f_in, f_dif = cmr.run_success_multi_sess(params, df_study_g2, df_test_g2, sem_mat, mode="Recog-CR", design="S1G3")
    df_simu_g2["test"] = df_test_g2["test"]
    df_simu_g2 = df_simu_g2.merge(df_test_g2, on=["session", "list", "test", "test_itemno1", "test_itemno2"])
#     df_simu_g2.to_parquet("data/simu8_result_g2.parquet")
# else:
#     df_simu_g2 = pd.read_parquet("data/simu8_result_g2.parquet")
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
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.08, top=0.98, hspace=0.04)

sns.lineplot(x=xpos, y=df_hr["HR"].to_numpy(), ax=ax1, marker="o", color="C0", markersize=10, linewidth=2)
sns.lineplot(x=xpos, y=df_far["FAR"].to_numpy(), ax=ax2, marker="s", color="C0", markersize=10, linewidth=2)

# ax1.set_ylim(0.73, 0.90)
# ax2.set_ylim(0.20, 0.37)
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
    plt.savefig("figures/simu8_g2_hrfar-neighbor.pdf")
plt.show()

# %% [markdown]
# ## Mechanism Check: Neighbourhood Effect on Recognition
#
# Human data (exp3 ground truth) show a distinctiveness effect: in denser neighbourhoods HR
# falls and FAR rises. The puzzle is that the model instead gives a tiny drop in *both* HR and
# FAR. The checks below show the actual reason: in this model the recognition evidence carries
# almost no neighbourhood information, so the small wiggles are essentially noise, not a real
# mechanism.
#
# Recognition evidence is `csim = dot(c_old, c_in)` between unit-normalized context vectors
# (`_core.pyx:746`). The only path from neighbourhood to recognition is `s_fc * sem_mat` in
# `M_FC` (`_core.pyx:287-292`); the `sem_mean` attention/criterion paths (`_core.pyx:364,373`)
# are off because `psi_s = 0` and `c_s = 0`. That path is weak, and the diagnostic signal for a
# face-name probe is the episodic face<->name co-occurrence, which is independent of how many
# neighbours the face has (names carry no semantic neighbours).

# %%
# csim, threshold, and yes-rate by neighbour count -- the recognition evidence is flat
df_recog["yes"] = (df_recog["s_resp"] == 1).astype(int)
for ca, lab in [(1, "target"), (0, "lure")]:
    sub = df_recog.query("correct_ans == @ca")
    g = sub.groupby("neighbour").agg(csim=("csim", "mean"), thresh=("thresh", "mean"), yes=("yes", "mean"), n=("yes", "size"))
    print(f"--- {lab} ---")
    print(g.round(4).to_string())
    print(f"corr(neighbour, csim)={np.corrcoef(sub.neighbour, sub.csim)[0, 1]:+.4f}  " f"corr(neighbour, yes)={np.corrcoef(sub.neighbour, sub.yes)[0, 1]:+.4f}\n")

# %%
# Model vs. human HR/FAR by density: the model barely moves and gets the FAR direction wrong
comp = pd.DataFrame(
    {
        "density": ["low", "medium", "high"],
        "HR_model": df_hr["HR"].to_numpy(),
        "FAR_model": df_far["FAR"].to_numpy(),
    }
)
comp

# %%
# Why the evidence is flat: re-encode a few sessions and measure, per face, the raw norm of the
# cued context M_FC.f and its normalized self-component. Both are essentially invariant across
# faces (range < ~1.5%), so neighbour count cannot move csim.
from CMR_IA._core import CMR

n_demo_sess = 50
list_num = len(np.unique(df_study_g2.list))
raw_norm_acc = np.zeros(16)
self_frac_acc = np.zeros(16)

# Loop over a subset of sessions, replicating only pretrial + encoding of run_success_single_sess
for sess in np.unique(df_study_g2.session)[:n_demo_sess]:
    pres_mat = np.reshape(df_study_g2.loc[df_study_g2.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy(), (list_num, -1, 2))
    cue_mat = np.reshape(df_test_g2.loc[df_test_g2.session == sess, ["test_itemno1", "test_itemno2"]].to_numpy(), (list_num, -1, 2))
    m = CMR(params, pres_mat, sem_mat, cue_mat=cue_mat, mode="Recog-CR", design="S1G3", seed=int(sess))

    # Pre-trial context shift (beta = 1 on the first trial)
    m.phase = "pretrial"
    m.beta = 1.0
    m.beta_source = 1.0
    m.present_item(m.distractor_idx, None, update_context=True, update_weights=False)
    m.distractor_idx += 1

    # Encode the 16 face-name pairs, learning M_FC exactly as during study
    m.phase = "encoding"
    for m.serial_position in range(m.pres_indexes.shape[1]):
        m.beta = params["beta_enc"]
        m.beta_source = 0
        m.present_item(m.pres_indexes[0, m.serial_position], None, update_context=True, update_weights=True, use_new_context=params["use_new_context"])

    # Per face: raw norm of the temporal cue context and the normalized self-component
    M_FC = np.asarray(m.M_FC)
    for f in range(1, 17):
        idx = int(np.searchsorted(m.all_nos_unique, f))
        v = M_FC[: m.ntemporal, idx]
        raw_norm = np.sqrt(np.sum(v**2))
        raw_norm_acc[f - 1] += raw_norm
        self_frac_acc[f - 1] += v[idx] / raw_norm

# Assemble per-face table; neighbour (euclidean distance < 3) barely relates to the cue representation
face_target_csim = df_recog.query("correct_ans == 1").groupby("test_itemno1").csim.mean()
sem_face = sem_mat[:16, :16].copy()
np.fill_diagonal(sem_face, 0)
df_mech = pd.DataFrame(
    {
        "face": np.arange(1, 17),
        "neighbour": neighbour_count,
        "sum_cos": sem_face.sum(axis=1),  # linear similarity sum (what neighbour count tracks, r~0.6)
        "sum_cos2": (sem_face**2).sum(axis=1),  # quadratic sum (what the cue norm sees, r~0)
        "raw_norm": raw_norm_acc / n_demo_sess,  # ||M_FC . f|| before normalization
        "self_frac": self_frac_acc / n_demo_sess,  # diagnostic self-component after unit-normalization
    }
)
df_mech["target_csim"] = df_mech["face"].map(face_target_csim)
print(f"raw_norm spread  = {df_mech.raw_norm.min():.4f} .. {df_mech.raw_norm.max():.4f}")
print(f"self_frac spread = {df_mech.self_frac.min():.4f} .. {df_mech.self_frac.max():.4f}")
print(f"corr(neighbour, sum_cos)   = {np.corrcoef(df_mech.neighbour, df_mech.sum_cos)[0, 1]:+.3f}")
print(f"corr(neighbour, sum_cos2)  = {np.corrcoef(df_mech.neighbour, df_mech.sum_cos2)[0, 1]:+.3f}")
print(f"corr(neighbour, raw_norm)  = {np.corrcoef(df_mech.neighbour, df_mech.raw_norm)[0, 1]:+.3f}")
print(f"corr(neighbour, csim)      = {np.corrcoef(df_mech.neighbour, df_mech.target_csim)[0, 1]:+.3f}")
df_mech.sort_values("neighbour")
