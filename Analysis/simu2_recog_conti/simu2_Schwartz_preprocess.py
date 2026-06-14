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
import numpy as np
import pandas as pd

# %%
# Load raw data
df = pd.read_table("data/RN2_Pix.dat", sep="\s+", names=["subject", "session", "list", "recog_pos", "picture", "category", "study_pos", "old_lag", "study_lag", "confidence", "rt"])
df

# %%
# Check the number of subjects
df["subject"].unique().shape

# %%
# Get rid of invalid trials where rt < 50 or rt > 3000 (also get rid of rt == 0 and confidence == 0)
df = df.query("50 <= rt <= 3000").copy()
df

# %%
# Binarize old/new
df["old"] = df.category == "_OLD_"
df

# %%
# Save
df.to_parquet("data/simu2_Schwartz_preproc.parquet")
