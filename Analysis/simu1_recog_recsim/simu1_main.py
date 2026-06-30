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
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.cluster import KMeans
import json

from CMR_IA.utils import wmse
from CMR_IA.fitting import _simu1_stats

cmr.analysis.setup_notebook()

SAVEFIG = False
RUNCMR = False
SAVERES = False
if SAVERES and not RUNCMR:
    print("Warning: SAVERES is ignored when RUNCMR is False; existing results are loaded instead.")

# %% [markdown]
# ## Load Data

# %%
# Load data
df = pd.read_parquet("data/simu1_test.parquet")
df

# %%
# Load semantic matrix
sem_mat = np.load("data/simu1_smat.npy")

# %% [markdown]
# ## Run CMR-IA

# %%
# Define parameters and load PSO results
params = cmr.load_params("1", fixed_params={"beta_cue": 0.0})  # beta_cue does not matter here
params

# %%
# Run model or load saved results
if RUNCMR:
    df_simu = cmr.run_conti_recog_multi_sess(params, df, sem_mat, design="EXP1")
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])
    if SAVERES:
        df_simu.to_parquet("data/simu1_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu1_result.parquet")
df_simu

# %%
# Plot csim histogram by old/new status
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(df_simu.query("old == 1"), x="csim", alpha=0.5, ax=ax, stat="density", label="old")
sns.histplot(df_simu.query("old == 0"), x="csim", alpha=0.5, ax=ax, stat="density", label="new")
plt.legend()
plt.show()

# %%
# Plot csim and threshold by position
old_csim_pos = df_simu.query("old == True").groupby("position").csim.mean()
old_pos = np.unique(df_simu.query("old == True").position.to_numpy())
new_csim_pos = df_simu.query("old == False").groupby("position").csim.mean()
new_pos = np.unique(df_simu.query("old == False").position.to_numpy())
cthresh_pos = df_simu.groupby("position").thresh.mean()[20:]
pos = np.unique(df_simu.position.to_numpy())[20:]

plt.plot(old_pos, old_csim_pos, label="old")
plt.plot(new_pos, new_csim_pos, label="new")
plt.plot(pos, cthresh_pos, label="thresh")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Plot csim_diff by position for old items
df_simu["csim_diff"] = df_simu["csim"] - df_simu["thresh"]
new_csim_diff_pos = df_simu.query("old == True").groupby("position").csim_diff.mean()
new_pos = np.unique(df_simu.query("old == True").position.to_numpy())

plt.plot(new_pos, new_csim_diff_pos, label="new")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %%
# Plot FAR by position
far_pos = df_simu.query("old == False").groupby("position").s_resp.mean()
pos = np.unique(df_simu.query("old == False").position.to_numpy())

plt.plot(pos, far_pos, label="new")
plt.xlabel("Position")
plt.ylabel("Similarity Score")
plt.legend()
plt.show()

# %% [markdown]
# ## Analysis

# %%
# Calculate the rolling category length
rolling_window = 9
category_label_dummies = df_simu["category_label"].str.get_dummies()
category_label_dummies.columns = ["cl_" + col for col in category_label_dummies.columns]
category_label_dummies_events = pd.concat([df_simu, category_label_dummies], axis=1)  # record the occurrence of every cat label
cl_rolling_sum = category_label_dummies_events.groupby("session").rolling(rolling_window, min_periods=1, on="position")[category_label_dummies.columns].sum().reset_index()
df_rollcat = df_simu.merge(cl_rolling_sum, on=["session", "position"])
df_simu["roll_cat_label_length"] = df_rollcat.apply(lambda x: x["cl_" + x["category_label"]], axis=1)  # how many cat within 9 window
df_simu["roll_cat_label_length"] = df_simu["roll_cat_label_length"] - 1  # how many cat in previous 8 window, not include self

# Add rolling category length level
df_simu["roll_cat_len_level"] = pd.cut(x=df_simu.roll_cat_label_length, bins=[0, 2, np.inf], right=False, include_lowest=True, labels=["0-1", ">=2"]).astype("str")
df_simu

# %%
# Add log and log lag bin
df_simu["log_lag"] = np.log(df_simu["lag"])
df_simu["log_lag_bin"] = pd.cut(df_simu["log_lag"], np.arange(df_simu["log_lag"].max() + 1), labels=False, right=False)
df_simu["log_lag_bin"] = df_simu.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
df_simu["log_lag_bin"] = df_simu.apply(lambda x: 5 if x["log_lag_bin"] > 5 else x["log_lag_bin"], axis=1)
df_simu

# %%
# Construct local FAR
old_vec = df_simu.old.to_numpy()
log_lag_bin_vec = df_simu.log_lag_bin.to_numpy()
position_vec = df_simu.position.to_numpy()
max_position = np.max(position_vec)
log_lag_bin_newpre_lst = []
log_lag_bin_newpost_lst = []
for i in range(len(df_simu)):
    if position_vec[i] > 0:
        if old_vec[i] == False and old_vec[i - 1] == True:
            log_lag_bin_newpre_lst.append(log_lag_bin_vec[i - 1])
        else:
            log_lag_bin_newpre_lst.append("N")
    else:
        log_lag_bin_newpre_lst.append("N")

    if position_vec[i] < max_position:
        if old_vec[i] == False and old_vec[i + 1] == True:
            log_lag_bin_newpost_lst.append(log_lag_bin_vec[i + 1])
        else:
            log_lag_bin_newpost_lst.append("N")
    else:
        log_lag_bin_newpost_lst.append("N")

df_simu["log_lag_bin_newpre"] = log_lag_bin_newpre_lst
df_simu["log_lag_bin_newpost"] = log_lag_bin_newpost_lst
df_simu

# %%
# Distribute items into bins
log_lag_bins = [0, 2, 3, 4, 5]
for bin in log_lag_bins:
    col_name = "log_lag_bin_" + str(bin)
    df_simu[col_name] = (df_simu.log_lag_bin == bin) | (df_simu.log_lag_bin_newpre == bin) | (df_simu.log_lag_bin_newpost == bin)
df_simu


# %%
# Compute rolling category distance features
def get_roll_cat_distance(df_tmp):
    cat_labels = df_tmp.category_label.to_numpy()
    roll_cat_lens = df_tmp.roll_cat_label_length.to_numpy()
    res = []
    for i in range(len(cat_labels)):
        now_cat = cat_labels[i]
        now_roll_cat_len = roll_cat_lens[i]
        if now_roll_cat_len == 0:
            res.append(np.array([]))
        else:
            all_cat = np.where(cat_labels == now_cat)[0]
            all_cat_idx = np.where(all_cat == i)[0].item()
            relevant_cat_pos = all_cat[int((all_cat_idx - now_roll_cat_len)) : all_cat_idx]
            res.append(relevant_cat_pos - i)
    return res


df_simu["roll_cat_distance"] = df_simu.groupby("session").apply(get_roll_cat_distance).explode().to_numpy()
df_simu["max_roll_cat_distance"] = df_simu["roll_cat_distance"].apply(lambda x: np.max(x) if len(x) > 0 else 0)
df_simu["mean_roll_cat_distance"] = df_simu["roll_cat_distance"].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
df_simu["effect_roll_cat_distance"] = df_simu["roll_cat_distance"].apply(lambda x: np.sum(9 + x) if len(x) > 0 else 0)
df_simu

# %%
# Drop the first 20 items
df_simu = df_simu.query("position >= 20")

# %% [markdown]
# ### Recency & Similarity

# %%
# Compute yes_rate by lag bin and similarity level
df_lst = []
for bin in log_lag_bins:
    col_name = "log_lag_bin_" + str(bin)
    df_tmp = df_simu.query(col_name + " == True").groupby(["session", "old", "roll_cat_len_level"])["s_resp"].agg(["mean", "sum", "count"]).reset_index()
    df_tmp["log_lag_bin"] = bin
    df_lst.append(df_tmp)
df_rollcat_laggp = pd.concat(df_lst)
df_rollcat_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
df_rollcat_laggp["yes_rate_adj"] = (df_rollcat_laggp["sum"] + 0.5) / (df_rollcat_laggp["count"] + 1)
df_rollcat_laggp["log_lag_disp"] = np.ceil(np.e**df_rollcat_laggp.log_lag_bin)  # log lag for display
df_rollcat_laggp

# %%
# Check item counts per group
df_rollcat_laggp.groupby(["old", "log_lag_bin", "roll_cat_len_level"])["count"].sum()

# %%
# Get hr, far, dprime
df_rollcat_laggp["old"] = df_rollcat_laggp["old"].astype("str")
df_dprime = pd.pivot_table(df_rollcat_laggp, values=["yes_rate", "yes_rate_adj"], index=["session", "roll_cat_len_level", "log_lag_disp"], columns="old").reset_index()
df_dprime.columns = [" ".join(col).strip() for col in df_dprime.columns.values]
df_dprime = df_dprime.rename(columns={"yes_rate False": "far", "yes_rate True": "hr", "yes_rate_adj False": "far_adj", "yes_rate_adj True": "hr_adj"})
df_dprime["z_hr"] = norm.ppf(df_dprime["hr_adj"])
df_dprime["z_far"] = norm.ppf(df_dprime["far_adj"])
df_dprime["dprime"] = df_dprime["z_hr"] - df_dprime["z_far"]
df_dprime

# %% [markdown]
# ### Plot HR

# %%
# Plot HR by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="hr", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="-", lw=2, ax=ax, errorbar=None)
plt.ylabel('P("Yes" | Old)')
plt.xlabel("Lag")
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.52, 0.78])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig(f"figures/simu1_hr.pdf")
plt.show()

# %% [markdown]
# ### Plot FAR

# %%
# Plot FAR by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="far", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="-", lw=2, ax=ax, errorbar=None)
plt.ylabel('P("Yes" | New)')
plt.xlabel("Lag")
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.25, 0.35])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level", loc="lower right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig(f"figures/simu1_far.pdf")
plt.show()

# %% [markdown]
# ### Plot d prime

# %%
# Plot d-prime by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="dprime", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="-", lw=2, ax=ax, errorbar=None)
plt.ylabel("d prime")
plt.xlabel("Lag")
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.5, 1.7])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")
plt.show()


# %% [markdown]
# ### Get Az

# %%
# Define function to calculate Az from ROC
def calculate_Az(df_tmp1, min_thresh=0.8, max_thresh=1.2):
    log_lag_bins = [0, 2, 3, 4, 5]
    Azs = []
    for bin in log_lag_bins:

        # Get the df of this log_lag_bin
        col_name = "log_lag_bin_" + str(bin)
        df_tmp = df_tmp1.query(col_name + " == True").copy()

        # Get variables
        conf = df_tmp.csim.to_numpy()
        truth = df_tmp.old.to_numpy()
        base_thresh = df_tmp.thresh.to_numpy() / params["c_thresh_itm"]
        old_num = np.sum(truth)
        new_num = np.sum(~truth)
        is_old = truth
        is_new = ~truth

        if np.sum(truth) == 0 or np.sum(~truth) == 0:
            Azs.append(np.nan)
            continue

        # Calculate hr and far for different thresholds
        ts = np.linspace(min_thresh, max_thresh, 7)
        hrs = []
        fars = []
        old_conf = conf * is_old
        new_conf = conf * is_new
        old_base_thresh = base_thresh * is_old
        new_base_thresh = base_thresh * is_new
        for t in ts:
            hr = (np.sum(old_conf > t * old_base_thresh) + 0.5) / (old_num + 1)
            far = (np.sum(new_conf > t * new_base_thresh) + 0.5) / (new_num + 1)
            hrs.append(hr)
            fars.append(far)

        # Calculate z_hr and z_far
        z_hr = norm.ppf(hrs)
        z_far = norm.ppf(fars)

        try:
            # Linear regression on z_hr and z_far manually
            n = len(z_far)
            X = np.column_stack((np.ones(n), z_far))
            beta = np.linalg.inv(X.T @ X) @ X.T @ z_hr
            intercept, slope = beta
        except:
            print("fail")
            Azs.append(np.nan)

        # Get A_z
        Az = norm.cdf(intercept / np.sqrt(1 + slope**2))
        Azs.append(Az)

    df_return = pd.DataFrame({"log_lag_bin": log_lag_bins, "Az": Azs})
    return df_return


# %%
# Compute Az per session and similarity level
df_Az = df_simu.groupby(["session", "roll_cat_len_level"]).apply(calculate_Az, min_thresh=0.9, max_thresh=1.1).reset_index()
df_Az.drop(columns="level_2", inplace=True)
df_Az["log_lag_disp"] = np.ceil(np.e**df_Az.log_lag_bin)
df_Az

# %%
# Check Nan
df_Az.loc[df_Az.Az.isna()]

# %% [markdown]
# ### Plot Az

# %%
# Plot Az by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_Az, y="Az", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="-", lw=2, ax=ax, errorbar=None)
plt.ylabel("$A_z$")
plt.xlabel("Lag")
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.52, 0.78])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="-", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="-", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.tick_params(labelleft=False)
    plt.savefig(f"figures/simu1_Az.pdf")
plt.show()

# %%
# Extract mean Az arrays for low and high similarity
df_plot = df_Az.groupby(["roll_cat_len_level", "log_lag_bin"]).Az.mean().to_frame(name="Az").reset_index()
Az_lowsim = df_plot.query("roll_cat_len_level == '0-1'").Az.to_numpy()
Az_highsim = df_plot.query("roll_cat_len_level == '>=2'").Az.to_numpy()
Az_lowsim, Az_highsim

# %% [markdown]
# ## Error Check

# %%
# Extract mean HR and FAR arrays for error check
df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].mean().reset_index()
hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["hr"].to_numpy()
hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["hr"].to_numpy()
far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["far"].to_numpy()
far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["far"].to_numpy()
hr_lowsim, hr_highsim, far_lowsim, far_highsim

# %%
# load ground truth
with open("data/simu1_gt.json") as f:
    gt = json.load(f)
hr_lowsim_gt = np.array(gt["hr_lowsim"])
hr_lowsim_std_gt = np.array(gt["hr_lowsim_std"])
hr_highsim_gt = np.array(gt["hr_highsim"])
hr_highsim_std_gt = np.array(gt["hr_highsim_std"])
far_lowsim_gt = np.array(gt["far_lowsim"])
far_lowsim_std_gt = np.array(gt["far_lowsim_std"])
far_highsim_gt = np.array(gt["far_highsim"])
far_highsim_std_gt = np.array(gt["far_highsim_std"])
far_lowsim_overall_gt = np.array(gt["far_lowsim_overall"])
far_lowsim_overall_std_gt = np.array(gt["far_lowsim_overall_std"])
far_highsim_overall_gt = np.array(gt["far_highsim_overall"])
far_highsim_overall_std_gt = np.array(gt["far_highsim_overall_std"])
hr_lowsim_gt, hr_highsim_gt, far_lowsim_gt, far_highsim_gt, far_lowsim_overall_gt, far_highsim_overall_gt

# %%
# Calculate error
err = wmse(hr_lowsim_gt, hr_lowsim, hr_lowsim_std_gt) + wmse(hr_highsim_gt, hr_highsim, hr_highsim_std_gt) + wmse(far_lowsim_gt, far_lowsim, far_lowsim_std_gt) + wmse(far_highsim_gt, far_highsim, far_highsim_std_gt)
err

# %%
# Verify fitting helper
df_simu_check = pd.read_parquet("data/simu1_result.parquet")
_, _, _, _, err = _simu1_stats(df_simu_check, gt)
err

# %% [markdown]
# ## Extra Analysis

# %%
# Plot csim histograms by similarity level and old/new status
df_csim = df_simu.query("log_lag_bin_0 == True or log_lag_bin_2 == True or log_lag_bin_3 == True or log_lag_bin_4 == True or log_lag_bin_5 == True")

# Plot histogram of csim, group by roll_cat_len_level and old
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(df_csim.query("old == True and roll_cat_len_level == '0-1'"), x="csim", alpha=0.5, ax=ax, stat="density", label="old low")
sns.histplot(df_csim.query("old == True and roll_cat_len_level == '>=2'"), x="csim", alpha=0.5, ax=ax, stat="density", label="old high")
sns.histplot(df_csim.query("old == False and roll_cat_len_level == '0-1'"), x="csim", alpha=0.5, ax=ax, stat="density", label="new low")
sns.histplot(df_csim.query("old == False and roll_cat_len_level == '>=2'"), x="csim", alpha=0.5, ax=ax, stat="density", label="new high")
plt.legend()
plt.show()

# %%
# Print mean csim_diff per lag bin and group
tmp_lst = [0, 2, 3, 4, 5]
for i in tmp_lst:
    print(i)
    df_tmp = df_csim.query(f"log_lag_bin_{i} == True").groupby(["session", "old", "roll_cat_len_level"]).csim_diff.mean().reset_index()
    print(df_tmp.groupby(["old", "roll_cat_len_level"]).csim_diff.mean())
    # print(df_csim.query(f"log_lag_bin_{i} == True").groupby(["old", "roll_cat_len_level"]).csim_diff.agg(["mean", "std"]).round(4))

# %%
# Plot csim_diff histograms per lag bin for old items
from matplotlib.ticker import FormatStrFormatter

tmp_lst = [0, 2, 3, 4, 5]
for i in tmp_lst:
    plt.subplots(figsize=(8, 6))
    sns.histplot(
        df_csim.query(f"log_lag_bin_{i} == True and old == True"),
        x="csim_diff",
        hue="roll_cat_len_level",
        hue_order=["0-1", ">=2"],
        binwidth=0.002,
        # bins=np.arange(0.3, 0.84, 0.002),
        alpha=0.5,
        stat="probability",
        common_norm=False,
        kde=True,
    )

    sns.histplot(
        df_csim.query(f"log_lag_bin_{i} == True and old == False"),
        x="csim_diff",
        hue="roll_cat_len_level",
        hue_order=["0-1", ">=2"],
        binwidth=0.002,
        # bins=np.arange(0.3, 0.84, 0.002),
        alpha=0.2,
        stat="probability",
        common_norm=False,
        kde=True,
    )
    # plt.xlim([0.35, 0.8])
    plt.title(f"log_lag_bin_{i}")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    plt.show()


# %%
# Define function to get ROC HR/FAR pairs
def get_roc_hrfar(df_tmp, min_thresh=0.8, max_thresh=1.2):

    # Get variables
    conf = df_tmp.csim.to_numpy()
    truth = df_tmp.old.to_numpy()
    base_thresh = df_tmp.thresh.to_numpy() / params["c_thresh_itm"]
    old_num = np.sum(truth)
    new_num = np.sum(~truth)
    is_old = truth
    is_new = ~truth

    # Calculate hr and far for different thresholds
    ts = np.linspace(min_thresh, max_thresh, 7)
    hrs = []
    fars = []
    old_conf = conf * is_old
    new_conf = conf * is_new
    old_base_thresh = base_thresh * is_old
    new_base_thresh = base_thresh * is_new
    for t in ts:
        hr = (np.sum(old_conf > t * old_base_thresh) + 0.5) / (old_num + 1)
        far = (np.sum(new_conf > t * new_base_thresh) + 0.5) / (new_num + 1)
        hrs.append(hr)
        fars.append(far)

    df_return = pd.DataFrame({"hr": hrs, "far": fars})
    return df_return


# %%
# Plot ROC curves by lag bin
for i in tmp_lst:
    df_csim_log_lag_bin = df_csim.query(f"log_lag_bin_{i} == True")
    df_plot = df_csim_log_lag_bin.groupby(["session", "roll_cat_len_level"]).apply(get_roc_hrfar, min_thresh=0.9, max_thresh=1.1).reset_index()
    df_plot.rename(columns={"level_2": "level"}, inplace=True)
    df_plot = df_plot.groupby(["roll_cat_len_level", "level"])[["hr", "far"]].mean().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.lineplot(data=df_plot, x="far", y="hr", hue="roll_cat_len_level", ax=ax, marker="o", markersize=10, linestyle="-", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FAR")
    plt.ylabel("HR")
    plt.legend()
    plt.title(f"log_lag_bin_{i}")
    plt.show()
