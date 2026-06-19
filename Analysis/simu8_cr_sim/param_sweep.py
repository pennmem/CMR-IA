"""Sweep recognition parameters to find a neighbourhood effect on HR/FAR for group-2 associative recognition."""

import numpy as np
import pandas as pd
import CMR_IA as cmr
import warnings

warnings.filterwarnings("ignore")

# Load data, semantic matrix, distances, and base params
df_study = pd.read_parquet("data/simu8_study.parquet")
df_test = pd.read_parquet("data/simu8_test.parquet")
sem_mat = np.load("data/simu8_smat.npy")
face_distance = np.load("data/simu8_distance.npy")

# Group 2, subsample sessions for speed
N_SESS = 4000
df_study_g2 = df_study.query("group == 2 and session < @N_SESS").copy()
df_test_g2 = df_test.query("group == 2 and session < @N_SESS").copy()

# Neighbour count per face (euclidean distance < 3)
neighbour_count = ((face_distance < 3.0) & (face_distance > 0)).sum(axis=1)

# Per-face semantic load from sem_mat (the model's actual channel)
sem_face = sem_mat[:16, :16].copy()
np.fill_diagonal(sem_face, 0)
sum_cos = sem_face.sum(axis=1)
print("=== sem_mat structure vs neighbour count ===")
print(f"corr(neighbour, sum_cos)  = {np.corrcoef(neighbour_count, sum_cos)[0,1]:+.3f}")
print(f"corr(neighbour, sum_cos2) = {np.corrcoef(neighbour_count, (sem_face**2).sum(axis=1))[0,1]:+.3f}")

# Base params
params_path = "/Users/bei/BeiWorld/Research/2022CMRIA/CMR_IA/Analysis/simuS1_recog_cr/data/simuS1_params.json"
base = cmr.load_params("S1", params_path=params_path, fixed_params={"learn_while_retrieving": True, "nitems_in_accumulator": 32, "ban_recall": np.arange(1, 17)})

# Ground truth (3 density bins: low/med/high)
hr_gt = [0.8296, 0.8087, 0.7862]
far_gt = [0.2801, 0.2810, 0.3031]


def evaluate(params, tag):
    """Run the model and report HR/FAR by density and csim-neighbour correlations."""
    df_simu, _, _ = cmr.run_success_multi_sess(params, df_study_g2, df_test_g2, sem_mat, mode="Recog-CR", design="S1G3", disable_tqdm=True)
    df_simu["test"] = df_test_g2["test"].to_numpy()
    df_simu = df_simu.merge(df_test_g2, on=["session", "list", "test", "test_itemno1", "test_itemno2"])
    rec = df_simu.query("test == 1").copy()
    rec["neighbour"] = rec["test_itemno1"].apply(lambda f: neighbour_count[f - 1])
    rec["density"] = pd.cut(rec["neighbour"], [4, 6, 8, 10], labels=["low", "medium", "high"])
    rec["yes"] = (rec["s_resp"] == 1).astype(int)
    hr = rec.query("correct_ans == 1").groupby("density", observed=True).yes.mean()
    far = rec.query("correct_ans == 0").groupby("density", observed=True).yes.mean()
    tgt = rec.query("correct_ans == 1")
    lure = rec.query("correct_ans == 0")
    rt = np.corrcoef(tgt.neighbour, tgt.csim)[0, 1]
    rl = np.corrcoef(lure.neighbour, lure.csim)[0, 1]
    # Per-neighbour HR/FAR for monotonicity check
    hr_n = tgt.groupby("neighbour").yes.mean()
    far_n = lure.groupby("neighbour").yes.mean()
    print(f"\n--- {tag} ---")
    print(f"  HR  low/med/high = {hr.values[0]:.4f} {hr.values[1]:.4f} {hr.values[2]:.4f}  (slope={hr.values[2]-hr.values[0]:+.4f}, GT slope={hr_gt[2]-hr_gt[0]:+.4f})")
    print(f"  FAR low/med/high = {far.values[0]:.4f} {far.values[1]:.4f} {far.values[2]:.4f}  (slope={far.values[2]-far.values[0]:+.4f}, GT slope={far_gt[2]-far_gt[0]:+.4f})")
    print(f"  HR  per-neigh  : " + " ".join(f"{n}:{v:.3f}" for n, v in hr_n.items()))
    print(f"  FAR per-neigh  : " + " ".join(f"{n}:{v:.3f}" for n, v in far_n.items()))
    print(f"  corr(neighbour, csim): target={rt:+.3f}  lure={rl:+.3f}")
    return hr.values, far.values


# Sweep configurations
configs = [
    ("s_fc=2.0, psi_s=-3.0, cta=0.96", dict(s_fc=2.0, psi_s=-3.0, c_thresh_assoc=0.96)),
    ("s_fc=2.0, psi_s=-3.0, cta=0.97", dict(s_fc=2.0, psi_s=-3.0, c_thresh_assoc=0.97)),
    ("s_fc=2.0, psi_s=-3.5, cta=0.97", dict(s_fc=2.0, psi_s=-3.5, c_thresh_assoc=0.97)),
    ("s_fc=2.5, psi_s=-3.5, cta=0.96", dict(s_fc=2.5, psi_s=-3.5, c_thresh_assoc=0.96)),
]

for tag, upd in configs:
    p = base.copy()
    p.update(upd)
    evaluate(p, tag)
