"""
Original design from exp1. Use self-generated w2v semantic matrix.
"""

import numpy as np
import pandas as pd

# Load raw data
df = pd.read_csv("../exp1_recog_recsim/data/cr_preproc_data_mturk.csv")

# Drop redundant columns
df = df.drop(columns=[
    "rt", "time_elapsed", "correct", "correct_num", "block_type",
    "item_name", "prev_cat", "prev_cat_match", "prev_cat_label",
    "prev_cat_label_match", "curr_cat_length", "curr_cat_label_length",
    "category",
])

# Drop subjects with > 250 no responses; also discard subject 200
subjlist = np.unique(df.subject_ID)
discard = [s for s in subjlist if np.sum(np.isnan(df.loc[df.subject_ID == s, "yes"].to_numpy().astype("float"))) > 250]
discard.append(200)
df = df.loc[~df.subject_ID.isin(discard)].copy()
df = df.astype({"yes": "float"})

# Add itemno
items = np.unique(df.item)
item2no = {item: i + 1 for i, item in enumerate(items)}
df["itemno"] = df["item"].map(item2no)

# Organize columns
df = df.sort_values(by=["subject_ID", "position"]).reset_index(drop=True)
df = df[["subject_ID", "position", "item", "itemno", "category_label", "lag", "old", "yes", "confidence"]]

# Assign session and build itemno columns
sess = np.unique(df.subject_ID)
test_parts = []
for i, subj in enumerate(sess):
    tmp = df.loc[df.subject_ID == subj].copy()
    tmp["session"] = i
    tmp["study_itemno1"] = tmp["itemno"]
    tmp["study_itemno2"] = -1
    tmp["test_itemno1"] = tmp["itemno"]
    tmp["test_itemno2"] = -1
    test_parts.append(tmp)
df_test = pd.concat(test_parts).reset_index(drop=True)

# Save
df_test.to_parquet("data/simu1_test.parquet", index=False)
print(f"Saved: {len(df_test)} test rows")
