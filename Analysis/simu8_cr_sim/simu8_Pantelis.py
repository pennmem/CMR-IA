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
# Load correct recall data
df_corr = pd.read_csv("data/simu8_Pantelis_correct.csv", header=None)
df_corr

# %%
# Pivot correct recall table
df_corr = df_corr.pivot_table(index=2, columns=3, values=1).reset_index(drop=True)
df_corr

# %%
# Compute SD and format X/Y columns
df_corr["SD1"] = df_corr[" sd+"] - df_corr[" mean"]
df_corr["SD2"] = df_corr[" mean"] - df_corr[" sd-"]
df_corr["SD"] = df_corr[["SD1", "SD2"]].mean(axis=1)
df_corr["X"] = ["1", "2", "3", "4", "5", "6-7"]
df_corr["Y"] = df_corr[" mean"]
df_corr

# %%
# Plot correct recall by neighbour group
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.errorbar(x=df_corr["X"], y=df_corr["Y"], yerr=df_corr["SD"], marker=None, capsize=3, capthick=2, linestyle="none")
sns.lineplot(data=df_corr, y="Y", x="X", ax=ax, marker="o", color="C0", markersize=10, linewidth=2, linestyle="--")

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.ylim([0.4, 1])
plt.xlim([-0.5, 5.5])
plt.xticks(ticks=np.arange(0, 6), labels=["1", "2", "3", "4", "5", "6-7"])
plt.xlabel("Number of Neighbours")
plt.ylabel("P(Correct)")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu8_Pantelis_correct.pdf")
plt.show()

# %%
# Load ILI data
df_ILI = pd.read_csv("data/simu8_Pantelis_ILI.csv", header=None)
df_ILI

# %%
# Pivot ILI table
df_ILI = df_ILI.pivot_table(index=2, columns=3, values=1).reset_index(drop=True)
df_ILI

# %%
# Compute SD and format X/Y columns for ILI
df_ILI["SD1"] = df_ILI[" sd+"] - df_ILI[" mean"]
df_ILI["SD2"] = df_ILI[" mean"] - df_ILI[" sd-"]
df_ILI["SD"] = df_ILI[["SD1", "SD2"]].mean(axis=1)
df_ILI["X"] = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", ">3.5"]
df_ILI["Y"] = df_ILI[" mean"]
df_ILI

# %%
# Plot ILI by neighbour group
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.errorbar(x=df_ILI["X"], y=df_ILI["Y"], yerr=df_ILI["SD"], marker=None, capsize=3, capthick=2, linestyle="none")
sns.lineplot(data=df_ILI, y="Y", x="X", ax=ax, marker="o", color="C0", markersize=10, linewidth=2, linestyle="--")

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.ylim([0, 0.25])
plt.xlim([-0.5, 6.5])
plt.xticks(ticks=np.arange(0, 7), labels=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", ">3.5"])
plt.xlabel("Distance Bins")
plt.ylabel("ILI Conditional Response Probability")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu8_Pantelis_ILI.pdf")
plt.show()

# %%
# Save gt as json
if SAVEDATA:
    with open("data/simu8_gt.json", "w") as f:
        json.dump(
            {
                "neighbor_mean": df_corr["Y"].to_numpy().tolist(),
                "neighbor_se": df_corr["SD"].to_numpy().tolist(),
                "ILI_mean": df_ILI["Y"].to_numpy().tolist(),
                "ILI_se": df_ILI["SD"].to_numpy().tolist(),
            },
            f,
            indent=4,
        )
