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
# Load Murdock data
df = pd.read_csv("data/simu5_Murdock.csv", header=None)
df

# %%
# Pivot table
df = df.pivot_table(index=2, columns=3, values=1).reset_index(drop=True)
df

# %%
# Compute SD and format X/Y columns
df["SD1"] = df[" SD+1"] - df[" Mean"]
df["SD2"] = df[" Mean"] - df[" SD-1"]
df["SD"] = df[["SD1", "SD2"]].mean(axis=1)
df["X"] = np.arange(0, 6)
df["Y"] = df[" Mean"]
df

# %%
# Save gt
if SAVEDATA:
    with open("data/simu5_gt.json", "w") as f:
        json.dump(
            {
                "hr": df["Y"].to_numpy().tolist(),
                "hr_std": df["SD"].to_numpy().tolist(),
            },
            f,
            indent=4,
        )

# %%
# Plot HR by test lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.errorbar(x=df["X"], y=df["Y"], yerr=df["SD"], linewidth=2, marker=None, capsize=3, capthick=2, linestyle="none")
sns.lineplot(data=df, y="Y", x="X", ax=ax, marker="o", color="C0", markersize=10, linewidth=2, linestyle="--")

plt.ylim([0, 1])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu5_Murdock.pdf")
