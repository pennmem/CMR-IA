"""
Self-generated design from Kahana. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

simu_sess_num = 1000
nlist = 6
n = 12
wordpool = np.arange(1, 1639)

rng = np.random.default_rng(seed=42)
study_parts = []
test_parts = []
for sess in range(simu_sess_num):
    
    all_words = rng.choice(wordpool, nlist * 2 * n, replace=False)
    for lst in range(nlist):
        
        # Study
        pres_words = all_words[lst * 2 * n : (lst + 1) * 2 * n].reshape(n, 2)
        pair_idx = np.arange(1, 13) + lst * n + sess * nlist * n
        study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "list": lst,
            "session": sess,
        }))

        # Build balanced test orders for two tests
        test_order = rng.permutation([[1, 1]] * 3 + [[1, 2]] * 3 + [[2, 1]] * 3 + [[2, 2]] * 3)
        test1_order = test_order[:, 0]
        test2_order = test_order[:, 1]
        test1_probe = np.array([pres_words[i, test1_order[i] - 1] for i in range(n)])
        test1_ans = np.array([pres_words[i, 2 - test1_order[i]] for i in range(n)])
        test2_probe = np.array([pres_words[i, test2_order[i] - 1] for i in range(n)])
        test2_ans = np.array([pres_words[i, 2 - test2_order[i]] for i in range(n)])

        # Test 1
        t1 = list(zip(test1_probe, test1_ans, test1_order, pair_idx))
        t1 = rng.permutation(t1)
        t1_probe, t1_ans, t1_ord, t1_idx = zip(*t1)
        test_parts.append(pd.DataFrame({
            "test_itemno1": np.array(t1_probe),
            "test_itemno2": -1,
            "test_item": items[np.array(t1_probe) - 1],
            "correct_ans": np.array(t1_ans),
            "order": np.array(t1_ord),
            "rep": 1,
            "test": 1,
            "list": lst,
            "session": sess,
            "test_pos": np.arange(1, 13),
            "pair_idx": np.array(t1_idx),
        }))

        # Test 2
        t2 = list(zip(test2_probe, test2_ans, test2_order, pair_idx))
        t2 = rng.permutation(t2)
        t2_probe, t2_ans, t2_ord, t2_idx = zip(*t2)
        test_parts.append(pd.DataFrame({
            "test_itemno1": np.array(t2_probe),
            "test_itemno2": -1,
            "test_item": items[np.array(t2_probe) - 1],
            "correct_ans": np.array(t2_ans),
            "order": np.array(t2_ord),
            "rep": 1,
            "test": 2,
            "list": lst,
            "session": sess,
            "test_pos": np.arange(13, 25),
            "pair_idx": np.array(t2_idx),
        }))

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simu6b_study.parquet", index=False)
df_test.to_parquet("data/simu6b_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
