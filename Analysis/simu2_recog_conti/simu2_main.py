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
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import CMR_IA as cmr
import json

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVERES = False

# %% [markdown]
# ## Load Data

# %%
# Load study, test data and semantic matrix
df_study = pd.read_parquet("data/simu2_study.parquet")
df_test = pd.read_parquet("data/simu2_test.parquet")
sem_mat = np.load("data/simu2_smat.npy")

# %%
# Inspect study data
df_study

# %%
# Inspect test data
df_test

# %% [markdown]
# ## Run CMR-IA

# %%
# Load fitted parameters
params = cmr.load_params("2", fixed_params={"c_thresh_itm": 1})
params

# %%
# Run model or load saved results
if SAVERES:
    df_simu = cmr.run_norm_recog_multi_sess(params, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
    df_simu.to_parquet("data/simu2_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu2_result.parquet")
df_simu

# %%
# Plot csim
old_csim = df_simu.query("old == True").groupby("recog_pos").csim.mean()
new_csim = df_simu.query("old == False").groupby("recog_pos").csim.mean()
thresh_mean = df_simu.groupby("recog_pos").thresh.mean()
pos = np.unique(df_simu.recog_pos)

plt.plot(np.unique(df_simu.query("old").recog_pos), old_csim, label="old")
plt.plot(np.unique(df_simu.query("~old").recog_pos), new_csim, label="new")
plt.plot(pos, thresh_mean, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()


# %% [markdown]
# ## Analysis

# %%
# Get conditions and propagate lag_cat to adjacent new items
def conditions(s):
    if s.old_lag == -999:
        return np.nan
    elif abs(s.old_lag) == 1:
        return "a"
    elif abs(s.old_lag) > 10:
        return "r"
    return np.nan


# Get lag category
df_simu["lag_cat"] = df_simu.apply(conditions, axis=1)

# Propagate lag_cat from an old item to the immediately following new item
recog_pos = df_simu.recog_pos.to_numpy()
old = df_simu.old.to_numpy()
lag_cat = df_simu.lag_cat.to_numpy()
lag_cat_with_new = [lag_cat[i - 1] if (recog_pos[i] > 1 and not old[i] and old[i - 1]) else lag_cat[i] for i in range(len(df_simu))]
df_simu["lag_cat"] = lag_cat_with_new

# new/old x lag_cat
create_level = {0: "new_r", 1: "new_a", 2: "old_r", 3: "old_a"}
df_t = df_simu.loc[pd.notna(df_simu.lag_cat)].copy()
df_t["level"] = df_t.apply(lambda x: create_level[x["old"] * 2 + (x["lag_cat"] == "a")], axis=1)
df_t

# %%
# Check overall HR and FAR
far = df_simu.query("old == False").s_resp.mean()
df_overall = df_simu.query("old == True").groupby(["subject", "lag_cat"]).s_resp.mean().reset_index()
hr_a = df_overall.groupby("lag_cat").s_resp.mean()["a"]
hr_r = df_overall.groupby("lag_cat").s_resp.mean()["r"]
print(f"FAR={far:.3f}  HR_adjacent={hr_a:.3f}  HR_remote={hr_r:.3f}")

hr_a_gt, hr_r_gt, far_gt = 0.724046, 0.693084, 0.268861
err_simple = (abs(hr_a - hr_a_gt) + abs(hr_r - hr_r_gt)) / 2 + abs(far - far_gt)
print(f"Simple error: {err_simple:.4f}")

# %% [markdown]
# ### ROC

# %%
# Compute ROC curve
df_roc = cmr.analysis.compute_roc(df_t, thresh_arr=np.arange(0, 2, 0.001))
df_roc

# %%
# Plot ROC curve
fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="both", direction="in")
ax.plot([0, 1], [0, 1], color="grey", linestyle="dashed")
sns.lineplot(data=df_roc, x="new_a", y="old_a", ax=ax, estimator=None, linewidth=2)
sns.lineplot(data=df_roc, x="new_r", y="old_r", ax=ax, estimator=None, linewidth=2)
plt.ylim([0, 1])
plt.xlim([0, 1])
ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks(ticks=ticks)
plt.yticks(ticks=ticks)
plt.xlabel("FAR")
plt.ylabel("HR")
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, label="Adjacent"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, label="Remote"),
]
plt.legend(handles=legend_elements, loc="lower right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu2_roc.pdf")
plt.show()

# %% [markdown]
# ### ZROC

# %%
# Compute and plot zROC curve
df_zroc = cmr.analysis.compute_zroc(df_roc)
df_zroc_plot_a = df_zroc.query("-2 < z_new_a < 2 and -2 < z_old_a < 2")
df_zroc_plot_r = df_zroc.query("-2 < z_new_r < 2 and -2 < z_old_r < 2")

fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="both", direction="in")
plt.axvline(x=0, color="grey", linestyle="dashed")
plt.axhline(y=0, color="grey", linestyle="dashed")
sns.lineplot(data=df_zroc_plot_a, x="z_new_a", y="z_old_a", ax=ax, estimator=None, linewidth=2)
sns.lineplot(data=df_zroc_plot_r, x="z_new_r", y="z_old_r", ax=ax, estimator=None, linewidth=2)
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.xlabel("Z(FAR)")
plt.ylabel("Z(HR)")
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, label="Adjacent"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, label="Remote"),
]
plt.legend(handles=legend_elements, loc="lower right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig("figures/simu2_zroc.pdf")
plt.show()

# %% [markdown]
# ### Error Check

# %%
# Load ground truth and compute ROC error
with open("data/simu2_gt.json") as f:
    gt = json.load(f)
far_a_gt = np.array(gt["far_a"])
hr_a_gt = np.array(gt["hr_a"])
far_r_gt = np.array(gt["far_r"])
hr_r_gt = np.array(gt["hr_r"])

hr_a_interp = cmr.analysis.interpolate_roc(df_roc["new_a"].to_numpy(), df_roc["old_a"].to_numpy(), far_a_gt)
hr_r_interp = cmr.analysis.interpolate_roc(df_roc["new_r"].to_numpy(), df_roc["old_r"].to_numpy(), far_r_gt)
err = np.sum((hr_a_interp - hr_a_gt) ** 2) + np.sum((hr_r_interp - hr_r_gt) ** 2)
print(f"ROC error: {err}")
print(f"Adjacent ordering preserved: {np.all((hr_a_interp > hr_r_interp)[1:])}")

# %% [markdown]
# ### Alternative plots

# %%
# Plot interpolated ROC at ground truth FAR points
fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
x = np.array([0, 1])
y = np.array([0, 1])
ax.plot(x, y, color="grey", linestyle="dashed")
plt.plot(far_a_gt, hr_a_interp, color=sns.color_palette()[0], marker="o", markersize=10, linestyle="-")
plt.plot(far_r_gt, hr_r_interp, color=sns.color_palette()[1], marker="o", markersize=10, linestyle="-")
plt.ylim([0, 1])
plt.xlim([0, 1])
ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks(ticks=ticks)
plt.yticks(ticks=ticks)
plt.xlabel("FAR")
plt.ylabel("HR")
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Adjacent"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="Remote"),
]
plt.legend(handles=legend_elements, loc="lower right")
plt.show()
