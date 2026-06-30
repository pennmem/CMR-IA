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

# %% metadata={}
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import CMR_IA as cmr
from CMR_IA.fitting import make_boundary, _simu6b_subj_stats as anal_perform

SAVERES = True

# %% [markdown]
# ## Load Stimuli and Semantic Matrix

# %% metadata={}
# Load study and test data
df_study = pd.read_parquet("data/simu6b_study.parquet")
df_test = pd.read_parquet("data/simu6b_test.parquet")

# %% metadata={}
# Inspect study data
df_study

# %% metadata={}
# Inspect test data
df_test

# %% metadata={}
# Load semantic matrix
sem_mat = np.load("../wordpools/ltp_FR_similarity_matrix.npy")

# %% [markdown]
# ## Run CMR-IA

# %% metadata={}
# Define parameters and load PSO results
params = cmr.load_params("6b", fixed_params={"learn_while_retrieving": True, "nitems_in_accumulator": 96})
params

# %% metadata={}
# Run model or load saved results
if SAVERES:
    df_simu, f_in, f_dif = cmr.run_success_multi_sess(params, df_study, df_test, sem_mat, mode="CR-CR")
    df_simu["test_pos"] = df_test["test_pos"]
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno1", "test_itemno2", "test_pos"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
    df_simu.to_parquet("data/simu6b_result.parquet")
else:
    df_simu = pd.read_parquet("data/simu6b_result.parquet")
df_simu

# %% [markdown]
# ## Analysis

# %% metadata={}
# Get condition and congruence
df_cond = df_simu.groupby(["pair_idx", "test"])["order"].mean().to_frame(name="corr_rate").reset_index()
df_cond = df_cond.pivot_table(index="pair_idx", columns="test", values="corr_rate").reset_index()
df_cond.columns = ["pair_idx", "test1", "test2"]


def cond(x):
    test1 = x["test1"]
    test2 = x["test2"]
    if test1 == 1 and test2 == 1:
        return "F-F"
    elif test1 == 1 and test2 == 2:
        return "F-B"
    elif test1 == 2 and test2 == 1:
        return "B-F"
    elif test1 == 2 and test2 == 2:
        return "B-B"


df_cond["cond"] = df_cond.apply(lambda x: cond(x), axis=1)
df_cond["cong"] = df_cond.apply(lambda x: "Identical" if x["cond"] == "F-F" or x["cond"] == "B-B" else "Reversed", axis=1)
df_cond

# %% metadata={}
# Merge condition and congruence into df_simu
pairidx2cond = df_cond.loc[:, ["pair_idx", "cond"]].set_index("pair_idx").to_dict()["cond"]
pairidx2cong = df_cond.loc[:, ["pair_idx", "cong"]].set_index("pair_idx").to_dict()["cong"]
df_simu["cond"] = df_simu.apply(lambda x: pairidx2cond[x["pair_idx"]], axis=1)
df_simu["cong"] = df_simu.apply(lambda x: pairidx2cong[x["pair_idx"]], axis=1)
df_simu.head(24)

# %%
# Compute anal_perform stats per subject
subjects = np.unique(df_simu.session)
inde_stats = []
reve_stats = []
for subj in subjects:
    df_subj_inde = df_simu.query(f"session == {subj} and cong == 'Identical'").copy()
    inde_stats.append(list(anal_perform(df_subj_inde)))
    df_subj_reve = df_simu.query(f"session == {subj} and cong == 'Reversed'").copy()
    reve_stats.append(list(anal_perform(df_subj_reve)))

# %%
# Inspect identical stats array
np.array(inde_stats)

# %%
# Inspect reversed stats array
np.array(reve_stats)

# %%
# Print mean identical stats
np.mean(inde_stats, axis=0).round(3)

# %%
# Print mean reversed stats
np.mean(reve_stats, axis=0).round(3)

# %% [markdown]
# ## Err Check

# %%
# Compute error against ground truth
inde_stats_mean = np.mean(inde_stats, axis=0)
reve_stats_mean = np.mean(reve_stats, axis=0)
with open("data/simu6b_gt.json") as f:
    gt = json.load(f)
inde_ground_truth = np.array(gt["inde"])
reve_ground_truth = np.array(gt["reve"])
err = np.sum(np.power(inde_stats_mean - inde_ground_truth, 2)) + np.sum(np.power(reve_stats_mean - reve_ground_truth, 2)) + np.power(inde_stats_mean[-1] - inde_ground_truth[-1], 2) + np.power(reve_stats_mean[-1] - reve_ground_truth[-1], 2)
err
