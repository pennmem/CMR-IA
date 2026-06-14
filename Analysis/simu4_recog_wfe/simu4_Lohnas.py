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
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %%
# Define word frequency bins
word_freq = [21, 51, 90, 141, 196, 285, 415, 632, 1163, 4332]

# %%
# Load Lohnas HR data
df_hr = pd.read_csv("data/simu4_Lohnas_HR.csv", header=None)
df_hr

# %%
# Pivot HR table
df_hr = df_hr.pivot_table(index=2, columns=3, values=1).reset_index(drop=True)
df_hr

# %%
# Add SD and mean columns
df_hr["SD1"] = df_hr[" sd+"] - df_hr[" mean"]
df_hr["SD2"] = df_hr[" mean"] - df_hr[" sd-"]
df_hr["SD"] = df_hr[["SD1", "SD2"]].mean(axis=1)
df_hr["X"] = word_freq
df_hr["Y"] = df_hr[" mean"]
df_hr

# %%
# Load Lohnas FAR data
df_far = pd.read_csv("data/simu4_Lohnas_FAR.csv", header=None)
df_far

# %%
# Pivot FAR table
df_far = df_far.pivot_table(index=2, columns=3, values=1).reset_index(drop=True)
df_far

# %%
# Add SD and mean columns to FAR
df_far["SD1"] = df_far[" sd+"] - df_far[" mean"]
df_far["SD2"] = df_far[" mean"] - df_far[" sd-"]
df_far["SD"] = df_far[["SD1", "SD2"]].mean(axis=1)
df_far["X"] = word_freq
df_far["Y"] = df_far[" mean"]
df_far

# %%
# Save gt
if SAVEDATA:
    with open("data/simu4_gt.json", "w") as f:
        json.dump(
            {
                "hr": df_hr["Y"].to_numpy().tolist(),
                "hr_std": df_hr["SD"].to_numpy().tolist(),
                "far": df_far["Y"].to_numpy().tolist(),
                "far_std": df_far["SD"].to_numpy().tolist(),
                "word_freq": np.array(word_freq).tolist(),
            },
            f,
            indent=4,
        )

# %%
# Plot HR and FAR by word frequency
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 9))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)
fig.subplots_adjust(hspace=0.03)

ax1.errorbar(x=df_hr["X"], y=df_hr["Y"], yerr=df_hr["SD"], linewidth=2, marker=None, capsize=3, capthick=2, linestyle="none")
ax2.errorbar(x=df_far["X"], y=df_far["Y"], yerr=df_far["SD"], linewidth=2, marker=None, capsize=3, capthick=2, linestyle="none")

sns.lineplot(data=df_hr, y="Y", x="X", ax=ax1, marker="o", color="C0", markersize=10, linewidth=2, linestyle="--")
sns.lineplot(data=df_far, y="Y", x="X", ax=ax2, marker="s", color="C0", markersize=10, linewidth=2, linestyle="--")

ax1.set_ylim(0.77, 0.95)
ax1.set_yticks(np.arange(0.80, 0.96, 0.05))
ax2.set_ylim(0.1, 0.28)
ax2.set_yticks(np.arange(0.1, 0.30, 0.05))
ax1.tick_params(axis="y", direction="in")
ax1.tick_params(axis="x", direction="in")
ax2.tick_params(axis="y", direction="in")
ax2.tick_params(axis="x", direction="in")
plt.xscale("log")
plt.xlim(5, 14000)

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

ax1.set_ylabel("HR")
ax2.set_ylabel("FAR")
ax2.set_xlabel("Word Frequency")

if SAVEFIG:
    ax1.set_ylabel(None)
    ax1.set_xlabel(None)
    ax2.set_ylabel(None)
    ax2.set_xlabel(None)
    plt.savefig("figures/simu4_Lohnas.pdf")

# %%
# Print rounded HR and FAR values
df_hr["Y"].to_numpy().round(3), df_far["Y"].to_numpy().round(3)
