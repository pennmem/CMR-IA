"""
Self-generated design from exp2. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

n = 48
v = n - 8  # last 8 pairs not tested
wordpool = np.arange(1, 1639)
nsubj = 100
g1_per = 4
g2_per = 3
g3_per = 5


# --- Group 1: Item Recognition + CR ---

rng = np.random.default_rng(seed=42)
g1_study_parts, g1_test_parts = [], []
for subj in range(nsubj):
    
    all_words = rng.choice(wordpool, g1_per * (2 * n + v), replace=False)
    for lst in range(g1_per):
        
        # Study
        pres_words = all_words[lst * (2 * n + v) : lst * (2 * n + v) + 2 * n].reshape(n, 2)
        pair_order = rng.permutation([0] * (v // 2) + [1] * (v // 2))
        g1_study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "order": np.concatenate((pair_order, [-1] * 8)),
            "pair_idx": np.arange(n) + lst * n,
            "list": lst, "session": subj, "subject": subj,
        }))

        valid_words = pres_words[:v, :]

        # Test1: item recognition
        old_probe = np.array([valid_words[i, 1 - pair_order[i]] for i in range(v)])
        new_probe = all_words[lst * (2 * n + v) + 2 * n : (lst + 1) * (2 * n + v)]
        test1_probe = np.concatenate((old_probe, new_probe))
        tmp_test1 = pd.DataFrame({
            "test_itemno1": test1_probe, "test_itemno2": -1,
            "test_item1": items[test1_probe - 1],
            "correct_ans": np.array([1] * v + [0] * v),
            "order": np.concatenate((pair_order, [-1] * v)),
            "pair_idx": np.concatenate((np.arange(v) + lst * n, [-1] * v)),
            "test": 1, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test1 = tmp_test1.iloc[rng.permutation(tmp_test1.index)].reset_index(drop=True)

        # Test2: cued recall
        test2_probe = np.array([valid_words[i, pair_order[i]] for i in range(v)])
        test2_ans = np.array([valid_words[i, 1 - pair_order[i]] for i in range(v)])
        tmp_test2 = pd.DataFrame({
            "test_itemno1": test2_probe, "test_itemno2": -1,
            "test_item1": items[test2_probe - 1],
            "correct_ans": test2_ans,
            "order": pair_order,
            "pair_idx": np.arange(v) + lst * n,
            "test": 2, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test2 = tmp_test2.iloc[rng.permutation(tmp_test2.index)].reset_index(drop=True)
        g1_test_parts.extend([tmp_test1, tmp_test2])

df_g1_study = pd.concat(g1_study_parts, ignore_index=True)
df_g1_test = pd.concat(g1_test_parts, ignore_index=True)
df_g1_study["group"] = 1
df_g1_test["group"] = 1


# --- Group 2: Pair Recognition + CR ---

rng = np.random.default_rng(seed=42)
g2_study_parts, g2_test_parts = [], []
for subj in range(nsubj):
    
    all_words = rng.choice(wordpool, g2_per * 2 * (n + v), replace=False)
    for lst in range(g2_per):
        
        # Study
        pres_words = all_words[lst * 2 * (n + v) : lst * 2 * (n + v) + 2 * n].reshape(n, 2)
        pair_order = rng.permutation([0] * (v // 2) + [1] * (v // 2))
        g2_study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "order": np.concatenate((pair_order, [-1] * 8)),
            "pair_idx": np.arange(n) + lst * n,
            "list": lst, "session": subj, "subject": subj,
        }))

        valid_words = pres_words[:v, :]

        # Test1: pair recognition
        old_pairs = valid_words.copy()
        for i in range(v):
            if pair_order[i] == 1:
                old_pairs[i] = np.flip(old_pairs[i])
        new_pairs = all_words[lst * 2 * (n + v) + 2 * n : (lst + 1) * 2 * (n + v)].reshape(v, 2)
        test1_pairs = np.concatenate((old_pairs, new_pairs), axis=0)
        tmp_test1 = pd.DataFrame({
            "test_itemno1": test1_pairs[:, 0], "test_itemno2": test1_pairs[:, 1],
            "test_item1": items[test1_pairs[:, 0] - 1],
            "test_item2": items[test1_pairs[:, 1] - 1],
            "correct_ans": np.array([1] * v + [0] * v),
            "order": np.concatenate((pair_order, [-1] * v)),
            "pair_idx": np.concatenate((np.arange(v) + lst * n, [-1] * v)),
            "test": 1, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test1 = tmp_test1.iloc[rng.permutation(tmp_test1.index)].reset_index(drop=True)

        # Test2: cued recall
        test2_probe = np.array([valid_words[i, pair_order[i]] for i in range(v)])
        test2_ans = np.array([valid_words[i, 1 - pair_order[i]] for i in range(v)])
        tmp_test2 = pd.DataFrame({
            "test_itemno1": test2_probe, "test_itemno2": -1,
            "test_item1": items[test2_probe - 1], "test_item2": None,
            "correct_ans": test2_ans,
            "order": pair_order,
            "pair_idx": np.arange(v) + lst * n,
            "test": 2, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test2 = tmp_test2.iloc[rng.permutation(tmp_test2.index)].reset_index(drop=True)
        g2_test_parts.extend([tmp_test1, tmp_test2])

df_g2_study = pd.concat(g2_study_parts, ignore_index=True)
df_g2_test = pd.concat(g2_test_parts, ignore_index=True)
df_g2_study["group"] = 2
df_g2_test["group"] = 2


# --- Group 3: Association Recognition + CR ---

rng = np.random.default_rng(seed=42)
g3_study_parts, g3_test_parts = [], []
for subj in range(nsubj):
    
    all_words = rng.choice(wordpool, g3_per * 2 * n, replace=False)
    for lst in range(g3_per):
        
        # Study
        pres_words = all_words[lst * 2 * n : (lst + 1) * 2 * n].reshape(n, 2)
        pair_order = rng.permutation([0] * (v // 2) + [1] * (v // 2))
        g3_study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "order": np.concatenate((pair_order, [-1] * 8)),
            "pair_idx": np.arange(n) + lst * n,
            "list": lst, "session": subj, "subject": subj,
        }))

        valid_words = pres_words[:v, :]

        # Test1: association recognition
        intact_idx = rng.permutation([True] * (v // 2) + [False] * (v // 2))
        intact_pair_idx = np.arange(v)[intact_idx] + lst * n
        intact_pairs_raw = valid_words[intact_idx, :].copy()
        intact_pairs = valid_words[intact_idx, :].copy()
        intact_order = pair_order[intact_idx]
        for i in range(len(intact_pairs)):
            if intact_order[i] == 1:
                intact_pairs[i] = np.flip(intact_pairs[i])

        rearrange_pairs = valid_words[~intact_idx, :].copy()
        rearrange_pairs[:, 1] = rng.permutation(rearrange_pairs[:, 1])
        rearrange_order = pair_order[~intact_idx]
        for i in range(len(rearrange_pairs)):
            if rearrange_order[i] == 1:
                rearrange_pairs[i] = np.flip(rearrange_pairs[i])

        test1_pairs = np.concatenate((intact_pairs, rearrange_pairs), axis=0)
        test1_order = np.concatenate((intact_order, rearrange_order))
        tmp_test1 = pd.DataFrame({
            "test_itemno1": test1_pairs[:, 0], "test_itemno2": test1_pairs[:, 1],
            "test_item1": items[test1_pairs[:, 0] - 1],
            "test_item2": items[test1_pairs[:, 1] - 1],
            "correct_ans": np.array([1] * (v // 2) + [0] * (v // 2)),
            "order": test1_order,
            "pair_idx": np.concatenate((intact_pair_idx, [-1] * (v // 2))),
            "test": 1, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test1 = tmp_test1.iloc[rng.permutation(tmp_test1.index)].reset_index(drop=True)

        # Test2: cued recall (only intact pairs)
        cued_order = pair_order[intact_idx]
        test2_probe = np.array([intact_pairs_raw[i, cued_order[i]] for i in range(v // 2)])
        test2_ans = np.array([intact_pairs_raw[i, 1 - cued_order[i]] for i in range(v // 2)])
        tmp_test2 = pd.DataFrame({
            "test_itemno1": test2_probe, "test_itemno2": -1,
            "test_item1": items[test2_probe - 1], "test_item2": None,
            "correct_ans": test2_ans,
            "order": cued_order,
            "pair_idx": intact_pair_idx,
            "test": 2, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test2 = tmp_test2.iloc[rng.permutation(tmp_test2.index)].reset_index(drop=True)
        g3_test_parts.extend([tmp_test1, tmp_test2])

df_g3_study = pd.concat(g3_study_parts, ignore_index=True)
df_g3_test = pd.concat(g3_test_parts, ignore_index=True)
df_g3_study["group"] = 3
df_g3_test["group"] = 3


# Merge groups
df_study = pd.concat([df_g1_study, df_g2_study, df_g3_study], ignore_index=True)
df_test = pd.concat([df_g1_test, df_g2_test, df_g3_test], ignore_index=True)

# Save
df_study.to_parquet("data/simuS1_study.parquet", index=False)
df_test.to_parquet("data/simuS1_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
