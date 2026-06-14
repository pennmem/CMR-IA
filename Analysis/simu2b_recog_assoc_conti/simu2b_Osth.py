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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %%
# Load raw data
df = pd.read_csv("data/exp1.csv")
df

# %%
# Extract test phase trials
df_test = df.query("phase == 'test'").copy()
df_test

# %% [markdown]
# ## Data Clean

# %%
# Count unique subjects
np.unique(df.subj).size

# %%
# Exclude extreme RTs
df_test = df_test.query("0.2 <= RT <= 8").copy()

# %%
# Get yes rate and compute HR/FAR
df_yesrate = df_test.groupby(["subj", "type"]).correct.mean().to_frame(name="yes_rate").reset_index()
df_yesrate = df_yesrate.pivot(index="subj", columns="type", values="yes_rate").reset_index()
df_yesrate["hr"] = df_yesrate["intact"]
df_yesrate["far"] = 1 - df_yesrate["rearranged"]
df_yesrate

# %%
# Calculate d-prime
df_yesrate["hr_z"] = norm.ppf(df_yesrate.hr)
df_yesrate["far_z"] = norm.ppf(df_yesrate.far)
df_yesrate["d_prime"] = df_yesrate.hr_z - df_yesrate.far_z
df_yesrate

# %%
# Identify subjects with negative d-prime
df_yesrate.query("d_prime < 0")

# %%
# Two subjects are excluded
bad_subj = df_yesrate.query("d_prime < 0").subj.to_numpy()
bad_subj

# %%
# Clean RT and first two trials after excluding bad subjects
df_test = df.query("subj not in @bad_subj").copy()
df_test = df_test.query("RT > 0.5").copy()
df_test = df_test.query("trial > 0").copy()
df_test

# %% [markdown]
# ## Overall HR and FAR

# %%
# Get yes rate and compute HR/FAR per subject
df_hrfar = df_test.groupby(["subj", "type"]).correct.mean().to_frame(name="yes_rate").reset_index()
df_hrfar = df_hrfar.pivot(index="subj", columns="type", values="yes_rate").reset_index()
df_hrfar["hr"] = df_hrfar["intact"]
df_hrfar["far"] = 1 - df_hrfar["rearranged"]
df_hrfar

# %%
# Melt HR/FAR for plotting
df_hrfar_plot = pd.melt(df_hrfar, id_vars=["subj"], value_vars=["hr", "far"], var_name="type", value_name="yes_rate")
df_hrfar_plot

# %%
# Plot HR and FAR bar chart
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.barplot(
    data=df_hrfar_plot,
    x="type",
    y="yes_rate",
    width=0.5,
    errorbar="se",
    ax=ax,
    err_kws={"lw": 2, "color": "black"},
    facecolor="white",
    edgecolor="C0",
    lw=2,
)

ax.set_xticks(ticks=[0, 1], labels=["HR", "FAR"])
ax.set_yticks(ticks=np.arange(0, 1.1, 0.2))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

ax.spines[["right", "top"]].set_visible(False)
ax.set(xlabel=None, ylabel="P(yes)")

if SAVEFIG:
    ax.set(ylabel=None)
    plt.tick_params(labelbottom=False)
    plt.savefig(f"figures/simu2b_Osth_hrfar.pdf")
plt.show()

# %% [markdown]
# ## Far with Lag

# %%
# Compute FAR by lag for lure items
df_lure = df_test.query("type == 'rearranged'").copy()
df_farlag = df_lure.groupby(["subj", "lag"]).correct.mean().to_frame(name="yes_rate").reset_index()
df_farlag["far"] = 1 - df_farlag["yes_rate"]
df_farlag

# %%
# Plot FAR by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.lineplot(
    data=df_farlag,
    x="lag",
    y="far",
    marker="o",
    markersize=10,
    errorbar="se",
    err_style="band",
    ax=ax,
    linewidth=2,
    linestyle="--",
)

ax.set_xticks(ticks=np.arange(1, 6))
ax.set_yticks(ticks=np.arange(0, 0.41, 0.1))
ax.set_xlim(0.8, 5.2)
ax.set_ylim(0, 0.4)
ax.set(xlabel="Lag", ylabel="FAR")
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

ax.spines[["right", "top"]].set_visible(False)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig(f"figures/simu2b_Osth_lag.pdf")
plt.show()

# %%
# Extract ground truth HR and FAR arrays
hr_gt = df_hrfar_plot.query("type == 'hr'").yes_rate.mean()
hr_std_gt = df_hrfar_plot.query("type == 'hr'").yes_rate.std()
far_gt = df_farlag.groupby("lag").far.mean().to_numpy()
far_std_gt = df_farlag.groupby("lag").far.std().to_numpy()
hr_gt, hr_std_gt, far_gt, far_std_gt

# %%
# Save gt as json
if SAVEDATA:
    with open("data/simu2b_gt.json", "w") as f:
        json.dump(
            {
                "hr": np.array(hr_gt).tolist(),
                "hr_std": np.array(hr_std_gt).tolist(),
                "far": np.array(far_gt).tolist(),
                "far_std": np.array(far_std_gt).tolist(),
            },
            f,
            indent=4,
        )
