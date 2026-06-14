# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: cmr
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## Initial Inspection

# %%
# Load raw data
df = pd.read_csv("data/cr_preproc_data_mturk.csv")
df

# %%
# Drop redundant columns
df = df.drop(["time_elapsed", "correct", "correct_num", "block_type", "item_name", "prev_cat", "prev_cat_match", "prev_cat_label", "prev_cat_label_match"], axis=1)
df

# %% [markdown]
# ## Discard subjects with no-resp > 250

# %%
# Discard subjects with too many no-responses
subjlist = df.subject_ID.to_numpy()
subjlist = np.unique(subjlist)
discard = []
for subj in subjlist:
    df_subj = df.loc[df.subject_ID == subj]
    no_ans = np.isnan(df_subj.yes.to_numpy().astype("float"))
    num_no_ans = np.sum(no_ans)
    if num_no_ans > 250:
        discard.append(subj)

discard.append(200)  # additional, see David
len(discard)

# %%
# Filter discarded subjects
df_cl = df.loc[df.subject_ID.isin(discard) == False]
df_cl = df_cl.sort_values(by=["subject_ID", "position"])
df_cl = df_cl.astype({"yes": "float"})  # NaN will be kept and would not be counted in following analysis
df_cl

# %% [markdown]
# ## Add Additional Columns

# %%
# Add item number column
items = np.unique(df_cl.item)
item2no = {}
for i in range(len(items)):
    item2no[items[i]] = i + 1
df_cl["itemno"] = df_cl.apply(lambda x: item2no[x.loc["item"]], 1)
df_cl

# %%
# Count cleaned subjects
subjlist_cl = df_cl.subject_ID
subjlist_cl = np.unique(subjlist_cl)
len(subjlist_cl)

# %%
# Assign session index and reset index
df_cl["session"] = df_cl.apply(lambda x: np.flatnonzero(np.asarray(subjlist_cl == x.loc["subject_ID"])).item(), 1)
df_cl = df_cl.reset_index(drop=True)
df_cl

# %%
# Save
df_cl.to_parquet("data/exp1_behav.parquet")
