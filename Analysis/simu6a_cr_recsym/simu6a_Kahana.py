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
import json
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False
SAVEDATA = False

# %%
# Load Kahana data
df = pd.read_csv("data/simu6a_Kahana.csv", header=[0, 1])
df

# %%
# Rename columns and set lag values
df.columns = ["Backward_X", "Backward_Y", "Forward_X", "Forward_Y"]
x_range = np.arange(6)
df["Backward_X"] = x_range
df["Forward_X"] = x_range
df

# %%
# Save gt as json
if SAVEDATA:
    with open("data/simu6a_gt.json", "w") as f:
        json.dump(
            {
                "fw": df["Forward_Y"].to_numpy().tolist(),
                "bw": df["Backward_Y"].to_numpy().tolist(),
            },
            f,
            indent=4,
        )

# %%
# Plot forward and backward recall by lag
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

sns.lineplot(data=df, x="Forward_X", y="Forward_Y", linewidth=2, marker="o", markersize=10, label="Forward", linestyle="--")
sns.lineplot(data=df, x="Backward_X", y="Backward_Y", linewidth=2, marker="o", markersize=10, label="Backward", linestyle="--")
ax.spines["left"].set_bounds(0, 1)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.ylim([0, 1.05])
plt.xlabel("Study-Test Lag")
plt.ylabel("P(Correct)")
legend_elements = [Line2D([0], [0], color=sns.color_palette()[0], lw=2, marker="o", markersize=10, linestyle="--", label="Forward"), Line2D([0], [0], color=sns.color_palette()[1], lw=2, marker="o", markersize=10, linestyle="--", label="Backward")]
plt.legend(handles=legend_elements, title="Recall Direction")

if SAVEFIG:
    ax.set(xlabel=None, ylabel=None)
    plt.savefig("figures/simu6a_Kahana.pdf")
plt.show()
