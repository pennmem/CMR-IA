"""
Self-generated design from Kahana. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

simu_sess_num = 100
nlist = 42
n = 6
wordpool = np.arange(1, 1639)

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
        order = np.arange(5, -1, -1)
        direction = rng.permutation([0] * 3 + [1] * 3)
        test_probe = pres_words[order, direction]
        correct_ans = pres_words[order, 1 - direction]
        test_parts.append(pd.DataFrame({
            "test_itemno": test_probe,
            "test_item": items[test_probe - 1],
            "correct_ans": correct_ans,
            "lag": np.arange(6),
            "list": lst,
            "session": sess,
            "order": direction + 1,  # 1=forward, 2=backward
        }))

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simu6a_study.parquet", index=False)
df_test.to_parquet("data/simu6a_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
