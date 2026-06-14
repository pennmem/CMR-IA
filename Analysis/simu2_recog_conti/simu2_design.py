"""
Original design from Schwartz. Use self-generated random semantic matrix.
"""

import numpy as np
import pandas as pd

# Load raw data
df = pd.read_table(
    "data/RN2_Pix.dat",
    sep=r"\s+",
    names=["subject", "session", "list", "recog_pos", "picture", "category", "study_pos", "old_lag", "study_lag", "confidence", "rt"],
)

# Map pic to itemno & old
pics = np.unique(df.picture)
pic2itemno = {pic: i + 1 for i, pic in enumerate(pics)}
df["itemno1"] = df.picture.map(pic2itemno)
df["itemno2"] = -1
df["old"] = df.category == "_OLD_"
df = df.sort_values(by=["subject", "list", "recog_pos"])

# Compute yes/no and organize
df["yes"] = (df["confidence"] >= 4).astype(int)
df["yes"] = df.apply(lambda x: np.nan if (x["confidence"] == 0 or x["rt"] < 50 or x["rt"] > 3000) else x["yes"], axis=1)
df = df.drop(columns=["session", "picture", "category", "confidence", "rt"])
df = df[["subject", "list", "recog_pos", "itemno1", "itemno2", "old", "old_lag", "study_pos", "study_lag", "yes"]]

# Assign session and split study / test
sess = np.unique(df.subject)
study_parts = []
test_parts = []
for i, subj in enumerate(sess):
    tmp = df.loc[df.subject == subj].copy()
    tmp["session"] = i
    test_parts.append(tmp)
    tmp_study = tmp.loc[tmp.old].copy()
    tmp_study = tmp_study.sort_values(by=["list", "study_pos"])
    study_parts.append(tmp_study)
df_test = pd.concat(test_parts).reset_index(drop=True)
df_study = pd.concat(study_parts).reset_index(drop=True)
df_study = df_study.drop(columns=["recog_pos", "old", "old_lag", "study_lag", "yes"])

# Save
df_study.to_parquet("data/simu2_study.parquet", index=False)
df_test.to_parquet("data/simu2_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
