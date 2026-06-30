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
import json
import pickle
import pandas as pd
import CMR_IA as cmr
import scipy as sp
import pingouin as pg
import matplotlib.pyplot as plt

from CMR_IA.utils import wmse
from CMR_IA.fitting import _simuS1_subj_stats, _simuS1_stats

np.set_printoptions(suppress=True)

cmr.analysis.setup_notebook()

SAVEFIG = False
RUNCMR = False
SAVERES = False
if SAVERES and not RUNCMR:
    print("Warning: SAVERES is ignored when RUNCMR is False; existing results are loaded instead.")

# %% [markdown]
# ## Run CMR-IA

# %%
# Read PSO results and build params
params = cmr.load_params("S1", fixed_params={"learn_while_retrieving": True, "rec_time_limit": 10000, "nitems_in_accumulator": 192})
params


# %%
# Define simulation function for each task group
def simu_success(tag, params):

    # Which group
    if tag == "Item-CR":
        test1_num = 80
        i = 1
        mode = "Recog-CR"
        design = None
    elif tag == "Pair-CR":
        test1_num = 80
        i = 2
        mode = "Recog-CR"
        design = None
    elif tag == "Asso-CR":
        test1_num = 40
        i = 3
        mode = "Recog-CR"
        design = "S1G3"

    # Load stimuli
    df_study = pd.read_parquet("data/simuS1_study.parquet")
    df_test = pd.read_parquet("data/simuS1_test.parquet")
    df_study = df_study.query(f"group == {i}")
    df_test = df_test.query(f"group == {i}")

    # Load semantic matrix
    sem_mat = np.load("../wordpools/ltp_FR_similarity_matrix.npy")

    # Run CMR
    df_simu, f_in_acc, f_in_dif = cmr.run_success_multi_sess(params, df_study, df_test, sem_mat, mode=mode, design=design)
    df_simu["test"] = df_test["test"]
    df_simu = df_simu.merge(df_test, on=["session", "list", "test", "test_itemno1", "test_itemno2"])

    # Get f_in
    sessions = np.unique(df_simu.session)
    tmp_corr_fin = []
    tmp_omax_fin = []
    for sess in sessions:
        df_tmp = df_study.loc[df_study.session == sess]
        tmp1 = df_tmp.study_itemno1.to_numpy()
        tmp2 = df_tmp.study_itemno2.to_numpy()
        df_tmp2 = df_test.loc[df_test.session == sess]
        tmp3 = df_tmp2.test_itemno1[df_tmp2.test_itemno1 >= 0].to_numpy()
        tmp4 = df_tmp2.test_itemno2[df_tmp2.test_itemno2 >= 0].to_numpy()
        tmp = np.concatenate((tmp1, tmp2, tmp3, tmp4))
        tmp = np.unique(tmp)  # sort
        nlists = len(np.unique(df_simu.list))
        for lst in range(nlists):
            tmp_corr = df_simu.query(f"session == {sess} and list == {lst}")["correct_ans"][test1_num:]
            corrid = np.searchsorted(tmp, tmp_corr)
            corr_fin = [f_in_dif[sess][lst * int(test1_num / 2) + i][id] for i, id in enumerate(corrid)]
            omax_fin = [np.max(np.delete(f_in_dif[sess][lst * int(test1_num / 2) + i], id)) for i, id in enumerate(corrid)]
            tmp_corr_fin = tmp_corr_fin + [-1] * test1_num + corr_fin
            tmp_omax_fin = tmp_omax_fin + [-1] * test1_num + omax_fin
    df_simu["corr_fin"] = tmp_corr_fin
    df_simu["omax_fin"] = tmp_omax_fin

    return df_simu


# %%
# Run model or load saved results
if RUNCMR:
    df_simu_g1 = simu_success("Item-CR", params)
    df_simu_g2 = simu_success("Pair-CR", params)
    df_simu_g3 = simu_success("Asso-CR", params)
    if SAVERES:
        df_simu_g1.to_parquet("data/simuS1_result_g1.parquet")
        df_simu_g2.to_parquet("data/simuS1_result_g2.parquet")
        df_simu_g3.to_parquet("data/simuS1_result_g3.parquet")
else:
    df_simu_g1 = pd.read_parquet("data/simuS1_result_g1.parquet")
    df_simu_g2 = pd.read_parquet("data/simuS1_result_g2.parquet")
    df_simu_g3 = pd.read_parquet("data/simuS1_result_g3.parquet")

# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Item - CR (Group1)

# %%
# Plot csim by position for Group 1
df_simu = df_simu_g1.query("test == 1").copy()
df_simu["position"] = np.tile(np.arange(320), 100)

old_csim_pos = df_simu.query("correct_ans == 1").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("correct_ans == 1").position.to_numpy())
new_csim_pos = df_simu.query("correct_ans == 0").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("correct_ans == 0").position.to_numpy())
cthresh_pos = df_simu.groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Compute Group 1 performance stats
subjects = np.unique(df_simu_g1.subject)
g1_stats = []
for subj in subjects:
    df_subj = df_simu_g1.query(f"subject == {subj}").copy()
    g1_stats.append(list(_simuS1_subj_stats(df_subj)))
g1_stats = np.array(g1_stats)

# %%
# Print Group 1 mean and SE stats
print(g1_stats)
print("mean:")
print(np.mean(g1_stats, axis=0))
print("se:")
print(sp.stats.sem(g1_stats, axis=0))

# %%
# Compute d-prime for Group 1
old_num = 40
new_num = 40
g1_stats_hack = g1_stats.copy()
hacked_hr = (g1_stats_hack[:, 1] * old_num + 0.5) / (old_num + 1)
hacked_far = (g1_stats_hack[:, 2] * new_num + 0.5) / (new_num + 1)
g1_stats_hack[:, 1] = hacked_hr
g1_stats_hack[:, 2] = hacked_far
g1_ds = sp.stats.norm.ppf(g1_stats_hack[:, 1]) - sp.stats.norm.ppf(g1_stats_hack[:, 2])
print(np.mean(g1_ds))
print(sp.stats.sem(g1_ds))

# %% [markdown]
# ### Pair - CR (Group2)

# %%
# Plot csim by position for Group 2
df_simu = df_simu_g2.query("test == 1").copy()
df_simu["position"] = np.tile(np.arange(240), 100)

old_csim_pos = df_simu.query("correct_ans == 1").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("correct_ans == 1").position.to_numpy())
new_csim_pos = df_simu.query("correct_ans == 0").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("correct_ans == 0").position.to_numpy())
cthresh_pos = df_simu.groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Compute Group 2 performance stats
subjects = np.unique(df_simu_g2.subject)
g2_stats = []
for subj in subjects:
    df_subj = df_simu_g2.query(f"subject == {subj}").copy()
    g2_stats.append(list(_simuS1_subj_stats(df_subj)))
g2_stats = np.array(g2_stats)

# %%
# Print Group 2 mean and SE stats
print(g2_stats)
print("mean:")
print(np.mean(g2_stats, axis=0))
print("se:")
print(sp.stats.sem(g2_stats, axis=0))

# %%
# Compute d-prime for Group 2
old_num = 40
new_num = 40
g2_stats_hack = g2_stats.copy()
hacked_hr = (g2_stats_hack[:, 1] * old_num + 0.5) / (old_num + 1)
hacked_far = (g2_stats_hack[:, 2] * new_num + 0.5) / (new_num + 1)
g2_stats_hack[:, 1] = hacked_hr
g2_stats_hack[:, 2] = hacked_far
g2_ds = sp.stats.norm.ppf(g2_stats_hack[:, 1]) - sp.stats.norm.ppf(g2_stats_hack[:, 2])
print(np.mean(g2_ds))
print(sp.stats.sem(g2_ds))

# %% [markdown]
# ### Association - CR (Group3)

# %%
# Plot csim by position for Group 3
df_simu = df_simu_g3.query("test == 1").copy()
df_simu["position"] = np.tile(np.arange(200), 100)

old_csim_pos = df_simu.query("correct_ans == 1").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("correct_ans == 1").position.to_numpy())
new_csim_pos = df_simu.query("correct_ans == 0").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("correct_ans == 0").position.to_numpy())
cthresh_pos = df_simu.groupby("position").thresh.mean()
pos = np.unique(df_simu.position.to_numpy())

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Compute Group 3 performance stats
subjects = np.unique(df_simu_g3.subject)
g3_stats = []
for subj in subjects:
    df_subj = df_simu_g3.query(f"subject == {subj}").copy()
    g3_stats.append(list(_simuS1_subj_stats(df_subj)))
g3_stats = np.array(g3_stats)

# %%
# Print Group 3 mean and SE stats
print(g3_stats)
print("mean:")
print(np.mean(g3_stats, axis=0))
print("se:")
print(sp.stats.sem(g3_stats, axis=0))

# %%
# Compute d-prime for Group 3
old_num = 20
new_num = 20
g3_stats_hack = g3_stats.copy()
hacked_hr = (g3_stats_hack[:, 1] * old_num + 0.5) / (old_num + 1)
hacked_far = (g3_stats_hack[:, 2] * new_num + 0.5) / (new_num + 1)
g3_stats_hack[:, 1] = hacked_hr
g3_stats_hack[:, 2] = hacked_far
g3_ds = sp.stats.norm.ppf(g3_stats_hack[:, 1]) - sp.stats.norm.ppf(g3_stats_hack[:, 2])
print(np.mean(g3_ds))
print(sp.stats.sem(g3_ds))

# %% [markdown]
# ### Aggregate Three Groups

# %%
# Compute mean stats across groups
stats = []
stats.append(list(np.mean(np.array(g1_stats), axis=0)))
stats.append(list(np.mean(np.array(g2_stats), axis=0)))
stats.append(list(np.mean(np.array(g3_stats), axis=0)))
stats = np.array(stats)
stats

# %%
# Compute SE stats across groups
stats_se = []
stats_se.append(list(sp.stats.sem(np.array(g1_stats), axis=0)))
stats_se.append(list(sp.stats.sem(np.array(g2_stats), axis=0)))
stats_se.append(list(sp.stats.sem(np.array(g3_stats), axis=0)))
stats_se = np.array(stats_se)
stats_se

# %% [markdown]
# ## Err Check

# %%
# Load ground truth and compute error
with open("data/simuS1_gt.json") as f:
    gt = json.load(f)
ground_truth = np.array([gt["g1_mean"], gt["g2_mean"], gt["g3_mean"]])  # p_rc, hr, far, q
ground_truth_se = np.array([gt["g1_se"], gt["g2_se"], gt["g3_se"]])

# %%
# Compute MSE error
err = np.mean(np.power(stats - ground_truth, 2))
err

# %%
# Verify fitting helper
_, err = _simuS1_stats([df_simu_g1, df_simu_g2, df_simu_g3], gt)
err / stats.size

# %%
# Compute weighted MSE error
err = wmse(ground_truth, stats, ground_truth_se)
err

# %% [markdown]
# ## Plot

# %%
# Plot recognition HR and FAR by group
y1_low, y1_high = 0.62, 0.85
y2_low, y2_high = 0.1, 0.28
range1 = y1_high - y1_low
range2 = y2_high - y2_low

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={"height_ratios": [range1, range2], "hspace": 0.03})
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(3)
width = 0.1

# HR
plot_params = {
    "color": "C0",
    "fmt": "o",
    "markersize": 10,
    "markeredgewidth": 2,
    "elinewidth": 2,
    "capsize": 5,
    "capthick": 2,
}
ax1.errorbar(
    x=base_xs - width,
    y=ground_truth[:, 1],
    yerr=ground_truth_se[:, 1],
    markerfacecolor="white",
    **plot_params,
)
ax1.errorbar(
    x=base_xs + width,
    y=stats[:, 1],
    yerr=stats_se[:, 1],
    **plot_params,
)

# FAR
ax2.errorbar(
    x=base_xs - width,
    y=ground_truth[:, 2],
    yerr=ground_truth_se[:, 2],
    markerfacecolor="white",
    **plot_params,
)
ax2.errorbar(
    x=base_xs + width,
    y=stats[:, 2],
    yerr=stats_se[:, 2],
    **plot_params,
)

# Setting
ax1.set_ylim(y1_low, y1_high)
ax2.set_ylim(y2_low, y2_high)
ax2.set_xlim(-0.5, 2.5)
ax1.set_yticks(np.arange(y1_low + 0.03, y1_high + 0.01, 0.05))
ax2.set_yticks(np.arange(y2_low, y2_high + 0.01, 0.05))
ax2.set_xticks(ticks=[0, 1, 2], labels=["Item", "Pair", "Associative"])
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
ax1.set_ylabel("HR")
ax2.set_ylabel("FAR")

ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax2.spines.right.set_visible(False)
ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
ax2.minorticks_off()

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax1.plot(0, 0, transform=ax1.transAxes, **kwargs)
ax2.plot(0, 1, transform=ax2.transAxes, **kwargs)

if SAVEFIG:
    ax1.set(ylabel=None)
    ax2.set(ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig("figures/simuS1_recog.pdf")
plt.show()

# %%
# Plot CR probability by group
fig, ax = plt.subplots(figsize=(6, 8))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(3)
width = 0.1

plot_params = {
    "color": "C0",
    "fmt": "o",
    "markersize": 10,
    "markeredgewidth": 2,
    "elinewidth": 2,
    "capsize": 5,
    "capthick": 2,
}
ax.errorbar(
    x=base_xs - width,
    y=ground_truth[:, 0],
    yerr=ground_truth_se[:, 0],
    markerfacecolor="white",
    **plot_params,
)
ax.errorbar(
    x=base_xs + width,
    y=stats[:, 0],
    yerr=stats_se[:, 0],
    **plot_params,
)

# Setting
ax.set_ylim(0.1, 0.5)
ax.set_xlim(-0.5, 2.5)
ax.set_xticks(ticks=[0, 1, 2], labels=["Item", "Pair", "Associative"])
ax.set_yticks(np.arange(0.1, 0.51, 0.1))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.set_ylabel("P(Rc)")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

if SAVEFIG:
    ax.set(ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig("figures/simuS1_prc.pdf")
plt.show()

# %%
# Plot Q by group
fig, ax = plt.subplots(figsize=(6, 8))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

base_xs = np.arange(3)
width = 0.1

# Q
plot_params = {
    "color": "C0",
    "fmt": "o",
    "markersize": 10,
    "markeredgewidth": 2,
    "elinewidth": 2,
    "capsize": 5,
    "capthick": 2,
}
ax.errorbar(
    x=base_xs - width,
    y=ground_truth[:, 3],
    yerr=ground_truth_se[:, 3],
    markerfacecolor="white",
    **plot_params,
)
ax.errorbar(
    x=base_xs + width,
    y=stats[:, 3],
    yerr=stats_se[:, 3],
    **plot_params,
)

# Setting
ax.set_ylim(0.5, 0.9)
ax.set_xlim(-0.5, 2.5)
ax.set_xticks(ticks=[0, 1, 2], labels=["Item", "Pair", "Associative"])
ax.set_yticks(np.arange(0.5, 0.91, 0.1))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.set_ylabel("Q")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

if SAVEFIG:
    ax.set(ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig("figures/simuS1_q.pdf")
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
    plt.savefig("figures/simuS1_legend.pdf")


# %% [markdown]
# ## Performance Analysis

# %%
# Combine groups into single dataframe
def array2df(group_stats, group_ds, group_num):
    group_stats = np.array(group_stats)
    df = pd.DataFrame(group_stats, columns=["PR", "HR", "FAR", "Q"])
    df["d"] = group_ds
    df["subject"] = df.index
    df["group"] = group_num
    return df


df_group1 = array2df(g1_stats, g1_ds, 1)
df_group2 = array2df(g2_stats, g2_ds, 2)
df_group3 = array2df(g3_stats, g3_ds, 3)
df_groups = pd.concat([df_group1, df_group2, df_group3])
df_groups

# %%
# ANOVA on PR
pg.anova(data=df_groups, dv="PR", between="group", detailed=True)

# %%
# Pairwise Tukey test on PR
pg.pairwise_tukey(data=df_groups, dv="PR", between="group")

# %%
# ANOVA on d-prime
pg.anova(data=df_groups, dv="d", between="group", detailed=True)

# %%
# Pairwise Tukey test on d-prime
pg.pairwise_tukey(data=df_groups, dv="d", between="group")

# %%
# ANOVA on Q
pg.anova(data=df_groups, dv="Q", between="group", detailed=True)

# %%
# Pairwise Tukey test on Q
pg.pairwise_tukey(data=df_groups, dv="Q", between="group")


# %% [markdown]
# ## Symmetry Analysis

# %%
# Define symmetry test function
def test_sym(df_simu, testnum):
    df = df_simu.query(f"test == {testnum} and order >= 0").copy()
    df["correct"] = df["s_resp"] == df["correct_ans"]
    df_order = df.groupby(["order", "subject"]).correct.mean().to_frame(name="p_correct").reset_index()
    return df_order


# %%
# Test symmetry on recognition across all groups
df_simu_all = pd.concat([df_simu_g1, df_simu_g2, df_simu_g3])
df_order_recog = test_sym(df_simu_all, 1)
print(df_order_recog.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_recog, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on CR across all groups
df_simu_all = pd.concat([df_simu_g1, df_simu_g2, df_simu_g3])
df_order_cr = test_sym(df_simu_all, 2)
print(df_order_cr.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_cr, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on recognition for Group 1
df_order_g1_recog = test_sym(df_simu_g1, 1)
print(df_order_g1_recog.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g1_recog, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on CR for Group 1
df_order_g1_cr = test_sym(df_simu_g1, 2)
print(df_order_g1_cr.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g1_cr, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on recognition for Group 2
df_order_g2_recog = test_sym(df_simu_g2, 1)
print(df_order_g2_recog.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g2_recog, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on CR for Group 2
df_order_g2_cr = test_sym(df_simu_g2, 2)
print(df_order_g2_cr.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g2_cr, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on recognition for Group 3
df_order_g3_recog = test_sym(df_simu_g3, 1)
print(df_order_g3_recog.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g3_recog, dv="p_correct", subject="subject", within="order")

# %%
# Test symmetry on CR for Group 3
df_order_g3_cr = test_sym(df_simu_g3, 2)
print(df_order_g3_cr.groupby("order").p_correct.mean())
pg.pairwise_tests(df_order_g3_cr, dv="p_correct", subject="subject", within="order")
