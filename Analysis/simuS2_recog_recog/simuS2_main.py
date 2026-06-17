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
import pandas as pd
import CMR_IA as cmr
import scipy as sp
import matplotlib.pyplot as plt

from CMR_IA.fitting import make_boundary, get_wmse, anal_perform_S2

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %%
# Load study and test data
df_study = pd.read_parquet("data/simuS2_study.parquet")
df_test = pd.read_parquet("data/simuS2_test.parquet")

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
params = cmr.load_params("S2", fixed_params={"learn_while_retrieving": True})
params

# %%
# Run model or load saved results
if SAVERES:
    df_simu, f_in_acc, f_in_dif = cmr.run_success_multi_sess(params, df_study, df_test, sem_mat, mode="Recog-Recog")
    df_simu["test"] = df_test["test"]
    df_simu = df_simu.merge(df_test, on=["session", "list", "test", "test_itemno1", "test_itemno2"])
    df_simu.to_parquet("data/simuS2_result.parquet")
else:
    df_simu = pd.read_parquet("data/simuS2_result.parquet")
df_simu

# %%
# Plot csim by position for single-item test trials
df_simu["position"] = np.tile(np.arange(360), 300)

old_csim_pos = df_simu.query("test_itemno2 == -1 and correct_ans == 1").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("correct_ans == 1").position.to_numpy())
new_csim_pos = df_simu.query("test_itemno2 == -1 and correct_ans == 0").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("correct_ans == 0").position.to_numpy())
cthresh_pos = df_simu.query("test_itemno2 == -1").groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Plot csim by position for pair test trials
df_simu["position"] = np.tile(np.arange(360), 300)

old_csim_pos = df_simu.query("test_itemno2 != -1 and correct_ans == 1").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("correct_ans == 1").position.to_numpy())
new_csim_pos = df_simu.query("test_itemno2 != -1 and correct_ans == 0").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("correct_ans == 0").position.to_numpy())
cthresh_pos = df_simu.query("test_itemno2 != -1").groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

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
# Get correctness and condition labels
df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans


def get_cond(x):
    this_type = x["type"]
    target = x["correct_ans"]
    if target == 1:
        if this_type == "Different_Item":
            return "Different_Item"
        elif this_type == "Item_Pair":
            return "Item_Pair"
        elif this_type == "Pair_Item":
            return "Pair_Item"
        elif this_type == "Same_Item":
            return "Same_Item"
        elif this_type == "Intact_Pair":
            return "Intact_Pair"
    elif target == 0:
        if this_type == "extra":
            return "NR_Lure"
        elif this_type == "Same_Item" or this_type == "Intact_Pair":
            return "Repeated_Lure"
        else:
            return "Discard"


df_simu["condition"] = df_simu.apply(get_cond, axis=1)
df_simu

# %%
# Compute performance stats per subject
subjects = np.unique(df_simu.subject)
stats = []
for subj in subjects:
    df_subj = df_simu.query(f"subject=={subj} and list % 3 != 0")  # discard first list
    stats_subj = anal_perform_S2(df_subj)
    stats.append(stats_subj)
stats_mean = np.nanmean(stats, axis=0)
stats_mean.round(2)

# %%
# Inspect mean stats array
stats_mean

# %%
# Compute SE stats per subject
stats_se = np.array(sp.stats.sem(np.array(stats), axis=0, nan_policy="omit"))
stats_se.round(3)

# %% [markdown]
# ### Err Check

# %%
# Define ground truth and compute error
ground_truth = np.array(
    [
        [0.82, 0.68, 0.26],  # diff item
        [0.82, 0.85, 0.64],  # item/pair
        [0.91, 0.85, 0.59],  # pair/item
        [0.81, 0.82, 0.86],  # same item
        [0.90, 0.92, 0.94],  # intact pair
        [0.07, 0.15, 0.54],  # repeated lure
        [0.07, 0.06, 0],  # non-repeated lure
    ]
)
ground_truth_se = np.array(
    [
        [0.020, 0.030, 0.10],  # diff item
        [0.016, 0.020, 0.12],  # item/pair
        [0.018, 0.021, 0.10],  # pair/item
        [0.017, 0.017, 0.03],  # same item
        [0.022, 0.019, 0.02],  # intact pair
        [0.014, 0.018, 0.12],  # repeated lure
        [0.009, 0.009, -1],  # non-repeated lure
    ]
)
err = np.sum(np.power(stats_mean - ground_truth, 2))
err

# %%
# Scale Q SE and compute weighted MSE
ground_truth_se_err = ground_truth_se.copy()
ground_truth_se_err[:, 2] /= 2
ground_truth_se_err

# %%
# Compute weighted MSE error
err = get_wmse(ground_truth, stats_mean, ground_truth_se_err)
err

# %% [markdown]
# ### Plot

# %%
# Define plot parameters
plot_params = {
    "color": "C0",
    "fmt": "o",
    "markersize": 10,
    "markeredgewidth": 2,
    "elinewidth": 2,
    "capsize": 5,
    "capthick": 2,
}
width = 0.15

# %%
# Plot Test 1 HR and FAR by condition
y1_low, y1_high = 0.62, 1
y2_low, y2_high = 0, 0.18
range1 = y1_high - y1_low
range2 = y2_high - y2_low

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"height_ratios": [range1, range2], "hspace": 0.03})
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(2, 7)
base_xs_far = np.arange(2)

# HR
ax1.errorbar(
    x=base_xs - width,
    y=ground_truth[:5, 0],
    yerr=ground_truth_se[:5, 0],
    markerfacecolor="white",
    **plot_params,
)
ax1.errorbar(
    x=base_xs + width,
    y=stats_mean[:5, 0],
    yerr=stats_se[:5, 0],
    **plot_params,
)

# FAR
ax2.errorbar(
    x=base_xs_far - width,
    y=ground_truth[7:4:-1, 0],
    yerr=ground_truth_se[7:4:-1, 0],
    markerfacecolor="white",
    **plot_params,
)
ax2.errorbar(
    x=base_xs_far + width,
    y=stats_mean[7:4:-1, 0],
    yerr=stats_se[7:4:-1, 0],
    **plot_params,
)

ax1.set_ylim(y1_low, y1_high)
ax2.set_ylim(y2_low, y2_high)
ax2.set_xlim(-0.5, 6.5)
ax1.set_yticks(np.arange(y1_low + 0.03, y1_high + 0.01, 0.05))
ax2.set_yticks(np.arange(y2_low, y2_high + 0.01, 0.05))
ax2.set_xticks(
    ticks=np.arange(7),
    labels=["NRep Lure", "Rep Lure", "Diff Item", "Item/Pair", "Pair/Item", "Same Item", "Intact Pair"],
    rotation=30,
    rotation_mode="anchor",
    va="top",
    ha="right",
)
ax1.set_ylabel("HR")
ax2.set_ylabel("FAR")

ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
ax2.minorticks_off()

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

if SAVEFIG:
    ax1.set(xlabel=None, ylabel=None)
    ax2.set(xlabel=None, ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig("figures/simuS2_test1.pdf")
plt.show()

# %%
# Plot Test 2 HR and FAR by condition
y1_low, y1_high = 0.62, 1
y2_low, y2_high = 0, 0.18
range1 = y1_high - y1_low
range2 = y2_high - y2_low

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"height_ratios": [range1, range2], "hspace": 0.03})
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(2, 7)
base_xs_far = np.arange(2)

# HR
ax1.errorbar(
    x=base_xs - width,
    y=ground_truth[:5, 1],
    yerr=ground_truth_se[:5, 1],
    markerfacecolor="white",
    **plot_params,
)
ax1.errorbar(
    x=base_xs + width,
    y=stats_mean[:5, 1],
    yerr=stats_se[:5, 1],
    **plot_params,
)

# FAR
ax2.errorbar(
    x=base_xs_far - width,
    y=ground_truth[7:4:-1, 1],
    yerr=ground_truth_se[7:4:-1, 1],
    markerfacecolor="white",
    **plot_params,
)
ax2.errorbar(
    x=base_xs_far + width,
    y=stats_mean[7:4:-1, 1],
    yerr=stats_se[7:4:-1, 1],
    **plot_params,
)

ax1.set_ylim(y1_low, y1_high)
ax2.set_ylim(y2_low, y2_high)
ax2.set_xlim(-0.5, 6.5)
ax1.set_yticks(np.arange(y1_low + 0.03, y1_high + 0.01, 0.05))
ax2.set_yticks(np.arange(y2_low, y2_high + 0.01, 0.05))
ax2.set_xticks(
    ticks=np.arange(7),
    labels=["NRep Lure", "Rep Lure", "Diff Item", "Item/Pair", "Pair/Item", "Same Item", "Intact Pair"],
    rotation=30,
    rotation_mode="anchor",
    va="top",
    ha="right",
)
ax1.set_ylabel("HR")
ax2.set_ylabel("FAR")

ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
ax2.minorticks_off()

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

if SAVEFIG:
    ax1.set(xlabel=None, ylabel=None)
    ax2.set(xlabel=None, ylabel=None)
    plt.tick_params(labelbottom=False)
    ax1.tick_params(labelleft=False)
    ax2.tick_params(labelleft=False)
    plt.savefig("figures/simuS2_test2.pdf")
plt.show()

# %%
# Plot Q by condition
fig, ax = plt.subplots(figsize=(6, 8))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(1, 6)

ax.errorbar(
    x=base_xs - width,
    y=ground_truth[:5, 2],
    yerr=ground_truth_se[:5, 2],
    markerfacecolor="white",
    **plot_params,
)
ax.errorbar(
    x=base_xs + width,
    y=stats_mean[:5, 2],
    yerr=stats_se[:5, 2],
    **plot_params,
)
ax.errorbar(
    x=-width,
    y=ground_truth[5, 2],
    yerr=ground_truth_se[5, 2],
    markerfacecolor="white",
    **plot_params,
)
ax.errorbar(
    x=width,
    y=stats_mean[5, 2],
    yerr=stats_se[5, 2],
    **plot_params,
)

# setting
ax.set_ylim(0, 1)
ax.set_xlim(-0.5, 5.5)
ax.set_xticks(
    ticks=np.arange(6),
    labels=["Rep Lure", "Diff Item", "Item/Pair", "Pair/Item", "Same Item", "Intact Pair"],
    rotation=30,
    rotation_mode="anchor",
    va="top",
    ha="right",
)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.set_ylabel("Q")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig("figures/simuS2_q.pdf")
plt.show()

# %%
# Plot legend
from matplotlib.lines import Line2D

fig2, ax2 = plt.subplots(figsize=(3, 2))
ax2.axis("off")
h_data = Line2D(
    [0],
    [0],
    color="C0",
    marker="o",
    markersize=10,
    markeredgewidth=2,
    markerfacecolor="white",
    linestyle="none",
    label="Data",
)
h_model = Line2D(
    [0],
    [0],
    color="C0",
    marker="o",
    markersize=10,
    markeredgewidth=2,
    linestyle="none",
    label="CMR-IA",
)
L = ax2.legend(handles=[h_data, h_model], loc="center", ncol=1, fontsize=24)
if SAVEFIG:
    plt.savefig("figures/simuS2_legend.pdf")
