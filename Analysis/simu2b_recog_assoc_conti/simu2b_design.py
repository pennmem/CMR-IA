"""
Original design from Osth. Use self-generated w2v semantic matrix.
"""

import json
import numpy as np
import pandas as pd

# Load raw data
df = pd.read_csv("data/exp1.csv")

# Compute forward direction (serPos1 < serPos2 means A-B order)
df["forward"] = (df["serPos1"] < df["serPos2"]).astype(int)

# Add itemnos
with open("data/item2no.json") as f:
    item2no = json.load(f)
df["itemno1"] = df["word1"].str.lower().str.strip().map(item2no)
df["itemno2"] = df["word2"].str.lower().str.strip().map(item2no)

# Discard subjects with != 960 trials
trial_counts = df.groupby("subj").trial.count()
discard_subjs = trial_counts[trial_counts != 960].index.to_numpy()
df = df.query("subj not in @discard_subjs").copy()

# Clean up columns
df = df.drop(columns=["response", "RT", "correct", "serPos1", "serPos2", "intactLag", "prevResponse", "prevRT"])
df = df.rename(columns={"cycle": "list"})

# Assign session and split study / test
sess = np.unique(df.subj)
study_parts = []
test_parts = []
for i, subj in enumerate(sess):
    tmp = df.loc[df.subj == subj].copy()
    tmp["session"] = i
    study_parts.append(tmp.query("phase == 'study'").drop(columns=["phase", "type", "lag"]))
    test_parts.append(tmp.query("phase == 'test'").drop(columns=["phase"]))
df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

df_study = df_study[["session", "subj", "list", "trial", "word1", "word2", "itemno1", "itemno2"]]
df_test = df_test[["session", "subj", "list", "trial", "word1", "word2", "itemno1", "itemno2", "type", "lag", "forward"]]

# Save
df_study.to_parquet("data/simu2b_study.parquet", index=False)
df_test.to_parquet("data/simu2b_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
