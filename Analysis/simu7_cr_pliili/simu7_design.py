"""
Self-generated design from Davis. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

simu_sess_num = 1000
list_num = 16
n = 12
wordpool = np.arange(1, 1639)
serial_pos = np.arange(12)

rng = np.random.default_rng(seed=42)
study_parts = []
test_parts = []
for sess in range(simu_sess_num):
    
    sess_words = rng.choice(wordpool, 2 * n * list_num, replace=False).reshape(list_num, -1)
    for lst in range(list_num):
        
        # Study
        pres_words = sess_words[lst].reshape(n, 2)
        study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "serial_pos": serial_pos,
            "list": lst,
            "session": sess,
        }))

        # Test
        study_pos_sel = rng.choice(serial_pos, 8, replace=False)
        test_dir = rng.permutation([0] * 4 + [1] * 4)
        test_probe = [pres_words[pos][d] for pos, d in zip(study_pos_sel, test_dir)]
        correct_ans = [pres_words[pos][1 - d] for pos, d in zip(study_pos_sel, test_dir)]
        test_parts.append(pd.DataFrame({
            "test_itemno": test_probe,
            "test_item": items[np.array(test_probe) - 1],
            "correct_ans": correct_ans,
            "study_pos": study_pos_sel,
            "test_dir": test_dir,
            "list": lst,
            "session": sess,
        }))

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simu7_study.parquet", index=False)
df_test.to_parquet("data/simu7_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
