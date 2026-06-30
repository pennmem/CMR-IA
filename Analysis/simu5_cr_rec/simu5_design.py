"""
Self-generated design from Murdock. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

simu_sess_num = 100
nlist = 78
n = 6
wordpool = np.arange(1, 1639)
pos_lags = np.array([0, 1, 2, 3, 4, 5])

rng = np.random.default_rng(seed=42)
study_parts = []
test_parts = []
for sess in range(simu_sess_num):
    
    all_words = rng.choice(wordpool, nlist * 2 * n, replace=False)
    for lst in range(nlist):
        
        # Study
        pres_words = all_words[lst * 2 * n : (lst + 1) * 2 * n].reshape(n, 2)
        study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "list": lst,
            "session": sess,
        }))

        # Test
        lag = rng.choice(pos_lags)
        test_probe = pres_words[5 - lag][0]
        correct_ans = pres_words[5 - lag][1]
        test_parts.append(pd.DataFrame({
            "test_itemno": [test_probe],
            "test_item": [items[test_probe - 1]],
            "correct_ans": [correct_ans],
            "lag": [lag],
            "list": [lst],
            "session": [sess],
        }))

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simu5_study.parquet", index=False)
df_test.to_parquet("data/simu5_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
