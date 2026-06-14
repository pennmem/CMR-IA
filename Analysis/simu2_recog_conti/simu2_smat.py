"""
Generate a random symmetric semantic similarity matrix for simu2.
"""

import numpy as np
import pandas as pd

# Load test df
df_test = pd.read_parquet("data/simu2_test.parquet")

# Random semantic matrix, diag elements as 1
n = int(np.max(df_test.itemno1))
rng = np.random.default_rng(seed=42)
s_mat = 0.1 * rng.random((n, n))
s_mat = s_mat - np.diag(np.diag(s_mat)) + np.identity(n)
for i in range(n):
    for j in range(i + 1, n):
        s_mat[i, j] = s_mat[j, i]

# Save
np.save("data/simu2_smat.npy", s_mat)
print(f"Saved: simu2_smat.npy  shape={s_mat.shape}")
