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
from matplotlib.lines import Line2D
from scipy.stats import norm
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False
SAVECSV = False

# %% [markdown]
# ## Data Preparation

# %%
# Import dataframe with both real data and simulation data
df = pd.read_parquet("data/exp1_behav.parquet")
df

# %%
# Plot confidence by position
old_conf_pos = df.query("old == True").groupby("position").confidence.mean()
old_pos = np.unique(df.query("old == True").position.values)
new_conf_pos = df.query("old == False").groupby("position").confidence.mean()
new_pos = np.unique(df.query("old == False").position.values)

plt.plot(old_pos, old_conf_pos, label="old")
plt.plot(new_pos, new_conf_pos, label="new")
plt.xlabel("Position")
plt.ylabel("Confidence")
plt.legend()
plt.show()

# %% [markdown]
# ### Get Rolling Category Length

# %%
# Calculate the rolling category length
rolling_window = 9
category_label_dummies = df["category_label"].str.get_dummies()
category_label_dummies.columns = ["cl_" + col for col in category_label_dummies.columns]
category_label_dummies_events = pd.concat([df, category_label_dummies], axis=1)  # record the occurrence of every cat label
cl_rolling_sum = category_label_dummies_events.groupby("subject_ID").rolling(rolling_window, min_periods=1, on="position")[category_label_dummies.columns].sum().reset_index()
df_rollcat = df.merge(cl_rolling_sum, on=["subject_ID", "position"])
df["roll_cat_label_length"] = df_rollcat.apply(lambda x: x["cl_" + x["category_label"]], axis=1)  # how many cat within 9 window
df["roll_cat_label_length"] = df["roll_cat_label_length"] - 1  # how many cat in previous 8 window, not include self

# add rolling category length level
df["roll_cat_len_level"] = pd.cut(x=df.roll_cat_label_length, bins=[0, 2, np.inf], right=False, include_lowest=True, labels=["0-1", ">=2"]).astype("str")
df

# %%
# Add log and log lag bin
df["log_lag"] = np.log(df["lag"])
df["log_lag_bin"] = pd.cut(df["log_lag"], np.arange(df["log_lag"].max() + 1), labels=False, right=False)
df["log_lag_bin"] = df.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
df["log_lag_bin"] = df.apply(lambda x: 5 if x["log_lag_bin"] > 5 else x["log_lag_bin"], axis=1)
df

# %% [markdown]
# ### Get Local FAR

# %% [markdown]
# For an old item at position i, new items at position i-1 or i+1 are regarded as in the same lag bin. Similarity level of a new item is further determined by how many same-category items are within previous 8 items, independent of the similarity level of the old item.

# %%
# Construct local FAR
old_vec = df.old.to_numpy()
log_lag_bin_vec = df.log_lag_bin.to_numpy()
position_vec = df.position.to_numpy()
log_lag_bin_newpre_lst = []  # previous is old
log_lag_bin_newpost_lst = []  # next is old
for i in range(len(df)):
    if position_vec[i] > 0:
        if old_vec[i] == False and old_vec[i - 1] == True:
            log_lag_bin_newpre_lst.append(log_lag_bin_vec[i - 1])
        else:
            log_lag_bin_newpre_lst.append("N")
    else:
        log_lag_bin_newpre_lst.append("N")

    if position_vec[i] < np.max(position_vec):
        if old_vec[i] == False and old_vec[i + 1] == True:
            log_lag_bin_newpost_lst.append(log_lag_bin_vec[i + 1])
        else:
            log_lag_bin_newpost_lst.append("N")
    else:
        log_lag_bin_newpost_lst.append("N")

df["log_lag_bin_newpre"] = log_lag_bin_newpre_lst
df["log_lag_bin_newpost"] = log_lag_bin_newpost_lst
df

# %%
# Check the FAR for those following an old item
df.query("log_lag_bin_newpre != 'N'").groupby(["subject_ID", "roll_cat_len_level"])["yes"].mean().to_frame(name="far").reset_index().groupby("roll_cat_len_level")["far"].mean()

# %% [markdown]
# lag 1~7 -> log_lag_bin 0 -> plot at lag 1
#
# lag 8~20 -> log_lag_bin 2 -> plot at lag 8
#
# lag 21~54 -> log_lag_bin 3 -> plot at lag 21
#
# lag 55~148 -> log_lag_bin 4 -> plot at lag 55 
#
# lag >=149 -> log_lag_bin 5 -> plot at lag 149

# %%
# Distribute items into bins
log_lag_bins = [0, 2, 3, 4, 5]
for bin in log_lag_bins:
    col_name = "log_lag_bin_" + str(bin)
    df[col_name] = (df.log_lag_bin == bin) | (df.log_lag_bin_newpre == bin) | (df.log_lag_bin_newpost == bin)
df

# %%
# Df.groupby(["old", "log_lag_bin", "roll_cat_len_level"]).yes.count()

# %%
# Plot the distribution of lags in df
fig, ax = plt.subplots()
sns.histplot(df.query("lag > 0"), x="lag", ax=ax, bins=np.arange(1, 300))

# %% [markdown]
# ## Analyze by Lag Group and Roll Cat Group

# %%
# Discard the first 20 trials
df = df.query("position >= 20").copy()  #
df

# %%
# Calculate overall HR and FAR
df_overall = df.groupby(["session", "roll_cat_len_level", "old"])["yes"].mean().reset_index()
df_overall.groupby(["roll_cat_len_level", "old"])["yes"].mean()

# %%
# Calculate overall FAR
far_lowsim_overall, far_highsim_overall = df_overall.query("old == False").groupby("roll_cat_len_level")["yes"].mean()
far_lowsim_overall_std, far_highsim_overall_std = df_overall.query("old == False").groupby("roll_cat_len_level")["yes"].std()
far_lowsim_overall, far_highsim_overall, far_lowsim_overall_std, far_highsim_overall_std

# %%
# Get yes rate for each lag bin and similarity level
df_lst = []
for bin in log_lag_bins:
    col_name = "log_lag_bin_" + str(bin)
    df_tmp = df.query(col_name + " == True").groupby(["subject_ID", "old", "roll_cat_len_level"])["yes"].agg(["mean", "sum", "count"]).reset_index()
    df_tmp["log_lag_bin"] = bin
    df_lst.append(df_tmp)
df_rollcat_laggp = pd.concat(df_lst)
df_rollcat_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
df_rollcat_laggp["yes_rate_adj"] = (df_rollcat_laggp["sum"] + 0.5) / (df_rollcat_laggp["count"] + 1)
df_rollcat_laggp

# %%
# Log lag for display
df_rollcat_laggp["log_lag_disp"] = np.ceil(np.e**df_rollcat_laggp.log_lag_bin)
df_rollcat_laggp

# %%
df_rollcat_laggp.groupby(["old", "log_lag_bin", "roll_cat_len_level"])["count"].sum()

# %%
# Pivot for hr and far
df_rollcat_laggp["old"] = df_rollcat_laggp["old"].astype("str")
df_dprime = pd.pivot_table(df_rollcat_laggp, values=["yes_rate", "yes_rate_adj", "sum", "count"], index=["subject_ID", "roll_cat_len_level", "log_lag_disp"], columns="old").reset_index()
df_dprime.columns = [" ".join(col).strip() for col in df_dprime.columns.values]
df_dprime = df_dprime.rename(
    columns={
        "yes_rate False": "far",
        "yes_rate True": "hr",
        "yes_rate_adj False": "far_adj",
        "yes_rate_adj True": "hr_adj",
        "count False": "new num",
        "count True": "old num",
        "sum False": "new yes",
        "sum True": "old yes",
    }
)
df_dprime

# %%
# Calculate dprime
df_dprime["z_hr"] = norm.ppf(df_dprime["hr_adj"])
df_dprime["z_far"] = norm.ppf(df_dprime["far_adj"])
df_dprime["dprime"] = df_dprime["z_hr"] - df_dprime["z_far"]
df_dprime

# %% [markdown]
# ### HR

# %%
# Plot HR by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="hr", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="--", lw=2, ax=ax, errorbar="se")
plt.ylabel('P("Yes" | Old)')
plt.xlabel("Lag")
ax.set(xlabel=None, ylabel=None)
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.52, 0.78])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")

if SAVEFIG:
    plt.savefig(f"figures/exp1_hr.pdf")

# %% [markdown]
# ### FAR

# %%
# Plot FAR by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="far", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="--", lw=2, ax=ax, errorbar="se")
plt.ylabel('P("Yes" | New)')
plt.xlabel("Lag")
ax.set(xlabel=None, ylabel=None)
plt.xticks(ticks=np.arange(0, 160, 20))
plt.yticks(ticks=np.arange(0.14, 0.40, 0.02))
plt.ylim([0.25, 0.35])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level", loc="lower right")

if SAVEFIG:
    plt.savefig(f"figures/exp1_far.pdf")

# %% [markdown]
# ### d prime

# %%
# Plot d-prime by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_dprime, y="dprime", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="--", lw=2, ax=ax, errorbar="se")
plt.ylabel("d prime")
plt.xlabel("Lag")
ax.set(xlabel=None, ylabel=None)
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.5, 1.7])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")

# %% [markdown]
# ### Save ground truth

# %%
# Compute ground truth HR and FAR vectors
df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].mean().reset_index()
df_hrfar_std = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].std().reset_index()
hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"').hr.to_numpy()
hr_lowsim_std = df_hrfar_std.query('roll_cat_len_level == "0-1"').hr.to_numpy()
hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"').hr.to_numpy()
hr_highsim_std = df_hrfar_std.query('roll_cat_len_level == ">=2"').hr.to_numpy()
far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"').far.to_numpy()
far_lowsim_std = df_hrfar_std.query('roll_cat_len_level == "0-1"').far.to_numpy()
far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"').far.to_numpy()
far_highsim_std = df_hrfar_std.query('roll_cat_len_level == ">=2"').far.to_numpy()
hr_lowsim, hr_highsim, far_lowsim, far_highsim

# %%
# Inspect standard deviations
hr_lowsim_std, hr_highsim_std, far_lowsim_std, far_highsim_std

# %%
# Save gt
if SAVEDATA:
    with open("../simu1_recog_recsim/data/simu1_gt.json", "w") as f:
        json.dump(
            {
                "hr_lowsim": np.array(hr_lowsim).tolist(),
                "hr_lowsim_std": np.array(hr_lowsim_std).tolist(),
                "hr_highsim": np.array(hr_highsim).tolist(),
                "hr_highsim_std": np.array(hr_highsim_std).tolist(),
                "far_lowsim": np.array(far_lowsim).tolist(),
                "far_lowsim_std": np.array(far_lowsim_std).tolist(),
                "far_highsim": np.array(far_highsim).tolist(),
                "far_highsim_std": np.array(far_highsim_std).tolist(),
                "far_lowsim_overall": np.array(far_lowsim_overall).tolist(),
                "far_lowsim_overall_std": np.array(far_lowsim_overall_std).tolist(),
                "far_highsim_overall": np.array(far_highsim_overall).tolist(),
                "far_highsim_overall_std": np.array(far_highsim_overall_std).tolist(),
            },
            f,
            indent=4,
        )


# %% [markdown]
# ### Az

# %%
# Define calculate_Az function
def calculate_Az(df_tmp1):
    log_lag_bins = [0, 2, 3, 4, 5]
    Azs = []
    for bin in log_lag_bins:

        # get the df of this log_lag_bin
        col_name = "log_lag_bin_" + str(bin)
        df_tmp = df_tmp1.query(col_name + " == True").copy()

        # get variables
        conf = df_tmp.confidence.to_numpy()
        truth = df_tmp.old.to_numpy()
        old_num = np.sum(truth)
        new_num = np.sum(~truth)
        is_old = truth
        is_new = ~truth

        if old_num == 0 or new_num == 0:
            Azs.append(np.nan)
            continue

        min_conf = np.nanmin(conf)
        max_conf = np.nanmax(conf)

        if max_conf == min_conf:
            Azs.append(np.nan)
            continue

        # calculate HR and FAR for different thresholds
        thresholds = np.arange(1, 8)
        hrs = []
        fars = []
        old_conf = conf * is_old
        new_conf = conf * is_new
        for thresh in thresholds:
            hr = (np.sum(old_conf > thresh) + 0.5) / (old_num + 1)
            far = (np.sum(new_conf > thresh) + 0.5) / (new_num + 1)
            hrs.append(hr)
            fars.append(far)

        # calculate z_hr and z_far
        z_hr = norm.ppf(hrs)
        z_far = norm.ppf(fars)

        # linear regression on z_hr and z_far
        try:
            n = len(z_far)
            X = np.column_stack((np.ones(n), z_far))
            beta, *_ = np.linalg.lstsq(X, z_hr, rcond=None)
            intercept, slope = beta
        except np.linalg.LinAlgError:
            Azs.append(np.nan)
            continue

        # get A_z
        Az = norm.cdf(intercept / np.sqrt(1 + slope**2))
        Azs.append(Az)

    # df to return
    df_return = pd.DataFrame({"log_lag_bin": log_lag_bins, "Az": Azs})

    return df_return


# %%
# Get Az (some instability on different machines)
df_Az = df.groupby(["subject_ID", "roll_cat_len_level"]).apply(calculate_Az).reset_index()
df_Az.drop(columns="level_2", inplace=True)
df_Az["log_lag_disp"] = np.ceil(np.e**df_Az.log_lag_bin)
df_Az

# %%
# Pickup those rows where Az is Nan in df_Az
df_Az.loc[df_Az.Az.isna()]

# %%
# Plot Az by lag and similarity level
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df_Az, y="Az", x="log_lag_disp", hue="roll_cat_len_level", marker="o", markersize=10, linestyle="--", lw=2, ax=ax, errorbar="se")
plt.ylabel("$A_z$")
plt.xlabel("Lag")
ax.set(xlabel=None, ylabel=None)
plt.yticks(ticks=np.arange(0.5, 0.80, 0.05))
plt.xticks(ticks=np.arange(0, 160, 20))
plt.ylim([0.52, 0.78])
plt.xlim([0, 151])
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Low"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="High"),
]
plt.legend(handles=legend_elements, title="Similarity Level")

if SAVEFIG:
    plt.savefig(f"figures/exp1_Az.pdf")

# %%
# Summarize Az by similarity level
df_plot = df_Az.groupby(["roll_cat_len_level", "log_lag_bin"]).Az.mean().to_frame(name="Az").reset_index()
Az_lowsim = df_plot.query("roll_cat_len_level == '0-1'").Az.to_numpy()
Az_highsim = df_plot.query("roll_cat_len_level == '>=2'").Az.to_numpy()
Az_lowsim, Az_highsim

# %%
# Save for R
if SAVECSV:
    df_dprime.to_csv("data/exp1_hrfar.csv", index=False)
    df_Az.to_csv("data/exp1_Az.csv", index=False)
