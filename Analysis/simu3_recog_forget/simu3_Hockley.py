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
from matplotlib.lines import Line2D
import scipy as sp
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %%
# Load Hockley data
df = pd.read_csv("data/simu3_Hockley.csv", header=[0, 1])
df

# %%
# Rename columns and set lag values
df.columns = ["I_Hit_X", "I_Hit_Y", "A_Hit_X", "A_Hit_Y", "A_CR_X", "A_CR_Y", "Item_X", "Item_Y", "Pair_X", "Pair_Y"]
x_range = [2, 4, 6, 8, 16]
df["I_Hit_X"] = x_range
df["A_Hit_X"] = x_range
df["A_CR_X"] = x_range
df["Item_X"] = x_range
df["Pair_X"] = x_range
df

# %%
# Infer the FAR for items
I_Fars = sp.stats.norm.cdf(sp.stats.norm.ppf(df["I_Hit_Y"]) - df["Item_Y"])
I_Far = np.mean(I_Fars)
I_Far

# %%
# Check associative d-prime
sp.stats.norm.ppf(df["A_Hit_Y"]) - sp.stats.norm.ppf(1 - df["A_CR_Y"])

# %%
# Check associative CR rate
1 - df["A_CR_Y"]

# %%
# Save gt as json
if SAVEDATA:
    with open("data/simu3_gt.json", "w") as f:
        json.dump(
            {
                "I_hr": df["I_Hit_Y"].to_numpy().tolist(),
                "I_far": np.array(I_Far).tolist(),
                "A_hr": df["A_Hit_Y"].to_numpy().tolist(),
                "A_cr": df["A_CR_Y"].to_numpy().tolist(),
                "I_dprime": df["Item_Y"].to_numpy().tolist(),
                "A_dprime": df["Pair_Y"].to_numpy().tolist(),
            },
            f,
            indent=4,
        )

# %%
# Plot HR by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.lineplot(data=df, x="I_Hit_X", y="I_Hit_Y", linewidth=2, marker="o", markersize=10, label="I-Hits", linestyle="--")
sns.lineplot(data=df, x="A_Hit_X", y="A_Hit_Y", linewidth=2, marker="^", markersize=10, label="A-Hits", linestyle="--")
sns.lineplot(data=df, x="A_CR_X", y="A_CR_Y", linewidth=2, marker="^", markersize=10, label="A-CRs", linestyle="--")
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.ylim([0.5, 1])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")
plt.xticks(ticks=np.arange(2, 18, 2))
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="I-Hits"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="^", markersize=10, linestyle="--", label="A-Hits"),
    Line2D([0], [0], color=sns.color_palette()[2], lw=2, marker="^", markersize=10, linestyle="--", label="A-CRs"),
]
plt.legend(handles=legend_elements)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu3_Hockley_hr.pdf")
plt.show()

# %%
# Plot d-prime by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
sns.lineplot(data=df, x="Item_X", y="Item_Y", linewidth=2, marker="o", markersize=10, label="Items", linestyle="--")
sns.lineplot(data=df, x="Pair_X", y="Pair_Y", linewidth=2, marker="^", markersize=10, label="Pairs", linestyle="--")
plt.xlabel("Study-Test Lag")
plt.ylabel("$d^'$")
plt.ylim([0.5, 3])
plt.yticks(np.arange(0.5, 3.5, 0.5))
plt.xticks(ticks=np.arange(2, 18, 2))
legend_elements = [
    Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Items"),
    Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="^", markersize=10, linestyle="--", label="Pairs"),
]
plt.legend(handles=legend_elements)

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu3_Hockley_dprime.pdf")
plt.show()
