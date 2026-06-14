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
import pickle
import math
from scipy.stats import sem
import matplotlib.pyplot as plt
import pandas as pd
import time
import pprint
import seaborn as sns
import statsmodels.formula.api as smf
import CMR_IA as cmr

cmr.analysis.setup_notebook()

SAVEFIG = False

# %%
# Get word frequency quantile data
df = pd.read_parquet("data/simu4_word_freq.parquet")
df

# %%
# Load semantic matrix
s_mat = np.load("data/simu4_smat.npy")

# %%
# Compute mean semantic similarity per word
s_mat_dia = s_mat.copy()
np.fill_diagonal(s_mat_dia, 0)
df["s_mean"] = np.sum(s_mat_dia, axis=1) / (np.shape(s_mat_dia)[1] - 1)
df["log_freq"] = np.log(df["freq"])
df

# %%
# Regress s_mean on log frequency
model = smf.ols(formula="s_mean ~ log_freq", data=df).fit()
print(model.summary())

# %%
# Plot s_mean vs log frequency with regression line
b0 = model.params.iloc[0]
b1 = model.params.iloc[1]

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.scatterplot(data=df, x="log_freq", y="s_mean")
ax.axline((0, b0), slope=b1, color="k", linestyle="--")
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")

plt.xlim(-1, 11)
plt.ylim(0, 0.2)
plt.ylabel("$\mathbf{s}^{\mathrm{mean}}$")
plt.xlabel("Log Word Freqency", fontweight="bold")
plt.tight_layout()

if SAVEFIG:
    plt.savefig("figures/simu4_freq_sem.pdf")

# %%
# Reverse regression: log_freq ~ s_mean
model1 = smf.ols(formula="log_freq ~ s_mean", data=df).fit()
print(model1.summary())
