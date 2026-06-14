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
from scipy import stats
import seaborn as sns
from matplotlib.lines import Line2D
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %%
# Read processed data
df = pd.read_parquet("data/simu2_Schwartz_preproc.parquet")
df

# %%
# Check overall HR and FAR
df["yes"] = df["confidence"] >= 4
df_overall = df.groupby(["subject", "old"]).yes.mean().reset_index()
df_overall.groupby(["old"]).yes.mean()

# %% [markdown]
# ## Construct Conditions

# %%
# Inspect old items with no lag info
df.query("old == True and old_lag == -999")


# %%
# Label lag categories for old items
def conditions(s):
    if s.old_lag == -999:
        return np.nan
    elif np.absolute(s.old_lag) == 1:
        return "a"
    elif np.absolute(s.old_lag) > 10:
        return "r"
    else:
        return np.nan


df["lag_cat"] = df.apply(conditions, axis=1)
df

# %%
# Check yes rates by lag category and old/new
df_overall = df.groupby(["subject", "lag_cat", "old"]).yes.mean().reset_index()
df_overall.groupby(["lag_cat", "old"]).yes.mean()

# %%
# Construct local FAR by giving new items the lag category of the previous old item
recog_pos = df.recog_pos.values
old = df.old.values
lag_cat = df.lag_cat.values
lag_cat_with_new = []
for i in range(len(df)):
    if recog_pos[i] > 1:
        if not old[i] and old[i - 1]:
            lag_cat_with_new.append(lag_cat[i - 1])
        else:
            lag_cat_with_new.append(lag_cat[i])
    else:
        lag_cat_with_new.append(lag_cat[i])
df["lag_cat_with_new"] = lag_cat_with_new
df

# %%
# Count items per lag category and old/new
df.groupby(["old", "lag_cat_with_new"]).old.count()

# %%
# Count confidence in each condition
df["confidence"] = df["confidence"].astype("category")
df_conf = df.groupby(["subject", "lag_cat_with_new", "old", "confidence"]).old.count().to_frame(name="count").reset_index()
df_conf["cumsum"] = df_conf.groupby(["subject", "lag_cat_with_new", "old"])["count"].cumsum()
df_conf["total"] = df_conf.groupby(["subject", "lag_cat_with_new", "old"])["count"].transform("sum")
df_conf["rate"] = 1 - df_conf["cumsum"] / df_conf["total"]
df_conf

# %%
# Collapse across subjects to get ROC data
df_roc = df_conf.groupby(["lag_cat_with_new", "old", "confidence"]).rate.mean().reset_index()
df_roc = pd.pivot_table(df_roc, index=["lag_cat_with_new", "confidence"], columns="old", values="rate").reset_index()
df_roc.rename(columns={False: "FAR", True: "HR"}, inplace=True)
df_roc["z_FAR"] = stats.norm.ppf(df_roc["FAR"])
df_roc["z_HR"] = stats.norm.ppf(df_roc["HR"])
df_roc["confidence"] = df_roc["confidence"].astype("int")
df_roc

# %%
# Plot ROC curve
fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)

sns.lineplot(data=df_roc.query("confidence < 6"), x="FAR", y="HR", hue="lag_cat_with_new", marker="o", markersize=10, estimator=None, linewidth=2, linestyle="--", ax=ax)

ax.plot(np.array([0, 1]), np.array([0, 1]), color="grey", linestyle="dashed")
ax.spines[["right", "top"]].set_visible(False)
plt.ylim([0, 1])
plt.xlim([0, 1])
ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks(ticks=ticks)
plt.yticks(ticks=ticks)
plt.xlabel("FAR")
plt.ylabel("HR")
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Adjacent"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="Remote"),
]
plt.legend(handles=legend_elements, loc="lower right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu2_Schwartz_roc.pdf")
plt.show()

# %%
# Plot zROC curve
fig, ax = plt.subplots(figsize=(5, 5))
fig.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.97)

sns.lineplot(data=df_roc.query("confidence < 6"), x="z_FAR", y="z_HR", hue="lag_cat_with_new", marker="o", markersize=10, estimator=None, linewidth=2, linestyle="--", ax=ax)

plt.axvline(x=0, color="grey", linestyle="dashed")
plt.axhline(y=0, color="grey", linestyle="dashed")
ax.spines[["right", "top"]].set_visible(False)
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.xlabel("Z(FAR)")
plt.ylabel("Z(HR)")
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Adjacent"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="Remote"),
]
plt.legend(handles=legend_elements, loc="lower right")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu2_Schwartz_roc.pdf")
plt.show()

# %%
# Extract ground truth FAR/HR arrays
far_a_gt = df_roc.query("lag_cat_with_new == 'a' and confidence < 6").FAR.values
hr_a_gt = df_roc.query("lag_cat_with_new == 'a' and confidence < 6").HR.values
far_r_gt = df_roc.query("lag_cat_with_new == 'r' and confidence < 6").FAR.values
hr_r_gt = df_roc.query("lag_cat_with_new == 'r' and confidence < 6").HR.values
far_a_gt, hr_a_gt, far_r_gt, hr_r_gt

# %%
# Save gt
if SAVEDATA:
    with open("data/simu2_gt.json", "w") as f:
        json.dump(
            {
                "far_a": far_a_gt.tolist(),
                "hr_a": hr_a_gt.tolist(),
                "far_r": far_r_gt.tolist(),
                "hr_r": hr_r_gt.tolist(),
            },
            f,
            indent=4,
        )
