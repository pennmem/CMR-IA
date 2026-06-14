"""
Self-generated design from Schulman 1967. Use self-generated PEERS 984 semantic matrix.
"""

import numpy as np
import pandas as pd

# Load word frequency data
df_freq = pd.read_parquet("data/simu4_word_freq.parquet")
itemno_list = df_freq.itemno
df_freq = df_freq.set_index("itemno")

rng = np.random.default_rng(seed=42)
simu_sess_num = 1000
simu_old_num = 100
simu_new_num = 100
study_parts = []
test_parts = []
for i in range(simu_sess_num):
    
    # Choose words
    words = rng.choice(itemno_list, simu_old_num + simu_new_num, replace=False)
    old_words = words[:simu_old_num]
    new_words = words[simu_old_num:]
    test_words = rng.permutation(np.concatenate([old_words, new_words]))

    # Study
    tmp_study = df_freq.loc[old_words].copy()
    tmp_study["session"] = i
    tmp_study["list"] = 0
    study_parts.append(tmp_study)

    # Test
    tmp_test = df_freq.loc[test_words].copy()
    tmp_test["session"] = i
    tmp_test["list"] = 0
    tmp_test["position"] = np.arange(len(test_words))
    tmp_test["old"] = np.isin(test_words, old_words)
    test_parts.append(tmp_test)

df_study = pd.concat(study_parts).reset_index().rename(columns={"itemno": "itemno1"})
df_test = pd.concat(test_parts).reset_index().rename(columns={"itemno": "itemno1"})
df_study["itemno2"] = -1
df_test["itemno2"] = -1

# Save
df_study.to_parquet("data/simu4_study.parquet", index=False)
df_test.to_parquet("data/simu4_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
