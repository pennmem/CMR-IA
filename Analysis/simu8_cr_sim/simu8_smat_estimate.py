"""
Cheap proxy estimator for the simu8 g2 (associative recognition) neighbourhood effects.
Given a face-similarity kernel S(d) = logistic(d; D0, K), predict -- without running CMR --
the direction and size of (i) the HR drop and (ii) the FAR rise across neighbour-density
groups, plus (iii) how well S(d) reproduces the ground-truth yes-by-probe-distance shape.
Uses the analytical intact/rearranged cosines from note_assoc_recog_similarity. Absolute
magnitudes are proxies (a fixed representative CMR regime); the comparison *across* (D0, K)
is what matters, not the absolute values.
"""

import numpy as np
import pandas as pd
import json

# Representative CMR regime for the proxy (sets absolute scale only; relative ranking is robust)
GAMMA_FC = 0.8
BETA_ENC = 0.6
S_FC = 1.5

# Neighbour radius used to bin faces into density groups (must match the experiment/analysis)
NEIGH_RADIUS = 3.0

# Load MDS distance matrix and ground-truth yes-by-distance (lure bins 1.5, 2.5, 3.5, 4.5)
dist = np.load("data/simu8_distance.npy")
gt_yesdist = np.array(json.load(open("data/simu8_gt.json"))["exp3_yesdist_mean"])[1:]
lure_bins = np.array([1.5, 2.5, 3.5, 4.5])

# Per-face neighbour count and density group
neigh = ((dist < NEIGH_RADIUS) & (dist > 0)).sum(axis=1)
density = pd.cut(neigh, [4, 6, 8, 10], labels=["low", "med", "high"])


def logistic_smat(D0, K):
    # Logistic face-similarity with zero diagonal (off-diagonal neighbours only)
    S = 1.0 / (1.0 + np.exp(K * (dist - D0)))
    np.fill_diagonal(S, 0.0)
    return S


def proxies(S, gamma=GAMMA_FC, beta_enc=BETA_ENC, s_fc=S_FC):
    # Analytical intact (HR) and rearranged (FAR) cosines per face, from the note
    delta = 1 - gamma
    b = gamma * np.sqrt(1 - beta_enc ** 2)
    S2 = S @ S
    p = np.sqrt(1 + s_fc ** 2 * np.diag(S2))
    kap = gamma * beta_enc / np.sqrt(1 + p ** 2)
    a = delta + kap
    normA = np.sqrt(a ** 2 * p ** 2 + b ** 2 + kap ** 2)
    cos_intact = (delta * gamma * beta_enc * np.sqrt(1 + p ** 2) + gamma ** 2) / normA ** 2
    q = 2 * s_fc * S + s_fc ** 2 * S2
    np.fill_diagonal(q, 0.0)
    normD = np.sqrt(kap ** 2 * p ** 2 + b ** 2 + (delta + kap) ** 2)
    cos_rearr = (a[:, None] * kap[None, :] * q) / (normA[:, None] * normD[None, :])
    far_proxy = cos_rearr.sum(axis=1) / (len(S) - 1)
    return cos_intact, far_proxy


def shape_sse(D0, K):
    # Floor-aware fit yes(d) ~ floor + scale * S(d) to the GT lure curve; lower = better shape
    sb = 1.0 / (1.0 + np.exp(K * (lure_bins - D0)))
    A = np.c_[np.ones(len(sb)), sb]
    coef, *_ = np.linalg.lstsq(A, gt_yesdist, rcond=None)
    return np.sum((A @ coef - gt_yesdist) ** 2)


def evaluate(D0, K):
    # One row of metrics for a given kernel
    S = logistic_smat(D0, K)
    ci, fp = proxies(S)
    g = pd.DataFrame({"d": density, "ci": ci, "fp": fp}).groupby("d", observed=True).mean()
    return {
        "D0": D0,
        "K": K,
        "corr_svec2": np.corrcoef(neigh, np.diag(S @ S))[0, 1],  # HR-channel direction (>0 wanted)
        "corr_msim": np.corrcoef(neigh, S.sum(1) / (len(S) - 1))[0, 1],  # FAR-channel direction (>0 wanted)
        "hr_drop": g.ci["low"] - g.ci["high"],  # HR drop proxy (>0 wanted)
        "far_rise": g.fp["high"] - g.fp["low"],  # FAR rise proxy (>0 wanted)
        "hr_mono": bool(g.ci["low"] >= g.ci["med"] >= g.ci["high"]),
        "far_mono": bool(g.fp["low"] <= g.fp["med"] <= g.fp["high"]),
        "shape_sse": shape_sse(D0, K),  # yes-distance shape mismatch (small wanted)
    }


# Scan a grid of (D0, K) and print a table
if __name__ == "__main__":
    rows = [evaluate(D0, K) for K in (1.0, 1.5, 2.0) for D0 in (2.0, 2.5, 3.0, 3.5)]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"GT lure yes-distance: {gt_yesdist.round(3)}   neigh_radius={NEIGH_RADIUS}   regime: gamma={GAMMA_FC} beta_enc={BETA_ENC} s_fc={S_FC}")
    print("HR mirror needs corr_svec2>0 AND hr_drop>0;  FAR needs far_rise>0;  shape_sse smaller is better.\n")
    print(df.round({"corr_svec2": 2, "corr_msim": 2, "hr_drop": 4, "far_rise": 4, "shape_sse": 5}).to_string(index=False))
