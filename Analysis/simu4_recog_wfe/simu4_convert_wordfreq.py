"""
Build word frequency table for simu4 (984-item PEERS subset).
"""

import scipy.io
import numpy as np
import pandas as pd

# Select 984 words from 1638 with positive concreteness/imageability
mat = scipy.io.loadmat("../wordpools/concreteness_imageability_norms.mat", squeeze_me=True)
selected_idx = np.where(mat["C"] > 0)[0]

# Word strings
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = [line.rstrip("\n") for line in f]

# Frequency norms
mat = scipy.io.loadmat("../wordpools/Frequency_norms.mat", squeeze_me=True)
frequency = mat["F"]

# Create df
df = pd.DataFrame({
    "item": [items[i] for i in selected_idx],
    "itemno_old": [i + 1 for i in selected_idx],
    "itemno": np.arange(1, 985),
    "freq": [frequency[i] for i in selected_idx],
})

# Frequency quantile bins from Lonhas and Kahana (2013)
bins = [1, 36, 68, 115, 163, 235, 344, 495, 816, 1575, 26215]
df["quantile"] = pd.cut(df.freq, bins, labels=np.arange(10)).astype(int)

# Save
df.to_parquet("data/simu4_word_freq.parquet", index=False)
print(f"Saved: simu4_word_freq.parquet  shape={df.shape}")
