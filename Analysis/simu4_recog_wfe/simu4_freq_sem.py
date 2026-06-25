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
import statsmodels.formula.api as smf
import CMR_IA as cmr
from scipy.stats import pearsonr

cmr.analysis.setup_notebook()

SAVEFIG = False

# %%
# Get word frequency quantile data
df = pd.read_parquet("data/simu4_word_freq.parquet")
df

# %%
# Load semantic matrix
sem_mat = np.load("data/simu4_smat.npy")

# %%
# Compute mean semantic similarity per word
sem_mat_dia = sem_mat.copy()
np.fill_diagonal(sem_mat_dia, 0)
df["s_mean"] = np.sum(sem_mat_dia, axis=1) / (np.shape(sem_mat_dia)[1] - 1)
df["log_freq"] = np.log(df["freq"])
df

# %% [markdown]
# ## Semantic Mean and Frequency

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

# %% [markdown]
# ## Verify Semantic Norm

# %%
# Calculate svec norm
svec_norm = np.sqrt(np.sum(sem_mat**2, axis=1))
df["svec_norm"] = svec_norm

# Correlations with frequency
print(f"N words = {len(df)}")
print("\n=== Correlations with word frequency ===")
for col in ["s_mean", "svec_norm"]:
    r_log, p_log = pearsonr(df["log_freq"], df[col])
    print(f"{col:>10s}:  Pearson(log_freq) r={r_log:+.4f} (p={p_log:.2e})")

# s_mean vs svec_norm
r_ss, _ = pearsonr(df["s_mean"], df["svec_norm"])
print(f"\ns_mean vs svec_norm:  Pearson r={r_ss:+.4f}")

# Group means by frequency quantile
print("\n=== Means by frequency quantile ===")
grp = df.groupby("quantile").agg(
    freq_mean=("freq", "mean"),
    s_mean=("s_mean", "mean"),
    svec_norm=("svec_norm", "mean"),
).round({"freq_mean": 0, "s_mean": 5, "svec_norm": 6})
print(grp.to_string())

# %% [markdown]
# ## Verify Clustering (Monaco et al.)

# %%
# Raw value range (word2vec cosines can be negative)
N = sem_mat.shape[1]
off = sem_mat[~np.eye(N, dtype=bool)]
print(f"sem_mat shape={sem_mat.shape}  off-diag: min={off.min():.3f} max={off.max():.3f} "
      f"mean={off.mean():.3f} median={np.median(off):.3f}")

# Per-word metrics
df["s_top10"] = np.sort(sem_mat_dia, 1)[:, -10:].mean(1)  # mean of 10 closest neighbors
df["s_top50"] = np.sort(sem_mat_dia, 1)[:, -50:].mean(1)  # mean of 50 closest neighbors
for thr in (0.3, 0.4, 0.5):  # neighbors above cosine threshold
    df[f"n_above_{thr}"] = (sem_mat_dia > thr).sum(1)

metrics = ["s_mean", "s_top10", "s_top50", "n_above_0.3", "n_above_0.4", "n_above_0.5"]
print("\n=== Correlation of each metric with log word frequency ===")
for m in metrics:
    r, p = pearsonr(df["log_freq"], df[m])
    print(f"{m:>12s}:  Pearson(log_freq) r={r:+.4f} (p={p:.1e})")

print("\n=== Means by frequency quantile (0=rarest ... 9=most common) ===")
grp = df.groupby("quantile").agg(
    freq=("freq", "mean"),
    s_mean=("s_mean", "mean"),
    s_top10=("s_top10", "mean"),
    n_above_0_4=("n_above_0.4", "mean"),
).round(4)
print(grp.to_string())
