import os
import signal
import numpy as np
import pandas as pd
import json
from contextlib import contextmanager
from CMR_IA.utils import make_params, param_vec_to_dict, wmse, Yule_Q
from CMR_IA import _core as cmr


# --- Helpers for parameters --- #


def make_boundary(simu_name):
    """
    Make two vectors of boundary for parameters you want to fit.
    Also returns what parameters to fit in each simulation.
    """

    # Generate a base paramater dictionary
    lb_dict = make_params()
    lb_dict.update(
        beta_enc=0,
        beta_rec=0,
        beta_cue=0,
        beta_distract=0,
        beta_rec_post=0,
        phi_s=0,
        phi_d=0,
        s_cf=0,
        s_fc=0,
        kappa=0,
        eta=0,
        omega=1,
        alpha=0.5,
        c_thresh=0,
        c_thresh_itm=0,
        c_thresh_assoc=0,
        lamb=0,
        gamma_fc=0,
        gamma_cf=0,
        d_assoc=0,
        thresh_sigma=0,
        c_d=0,
    )

    ub_dict = make_params()
    ub_dict.update(
        beta_enc=1,
        beta_rec=1,
        beta_cue=1,
        beta_distract=1,
        beta_rec_post=1,
        phi_s=8,
        phi_d=5,
        s_cf=1,
        s_fc=1,
        kappa=0.5,
        eta=0.25,
        omega=10,
        alpha=1,
        c_thresh=1,
        c_thresh_itm=2,
        c_thresh_assoc=2,
        lamb=0.25,
        gamma_fc=1,
        gamma_cf=1,
        d_assoc=1,
        thresh_sigma=0.5,
        c_d=10,
    )

    # Determine which parameters to fit and their boundary for different simulations
    if simu_name == "1":

        what_to_fit = [
            "beta_enc",
            "beta_rec_post",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "c_d",
        ]
        ub_dict.update(
            beta_enc=0.4,
            beta_rec_post=0.4,
            s_fc=0.4,
            gamma_fc=0.4,
        )

    elif simu_name == "2":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "beta_rec_post",
            "s_fc",
            "gamma_fc",
            "c_d",
        ]
        lb_dict.update(
            beta_enc=0.2,
        )
        ub_dict.update(
            beta_enc=0.8,
            gamma_fc=0.5,
        )

    elif simu_name == "2b":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "beta_rec_post",
            "s_fc",
            "gamma_fc",
            "c_thresh_assoc",
            "c_d",
        ]

    elif simu_name == "3":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_rec_post",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "c_thresh_assoc",
            "c_d",
        ]

    elif simu_name == "4":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "c_s",
            "psi_s",
            "psi_c",
            "c_d",
        ]
        lb_dict.update(
            c_thresh_itm=-10,
            c_s=0,
            psi_s=0,
            psi_c=-10,
        )
        ub_dict.update(
            s_fc=0.3,
            c_thresh_itm=10,
            c_s=100,
            psi_s=100,
            psi_c=10,
        )

    elif simu_name == "4base":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "c_d",
        ]
        ub_dict.update(
            c_thresh_itm=10,
        )

    elif simu_name == "4shift":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "c_s",
            "c_d",
        ]
        lb_dict.update(
            c_thresh_itm=-10,
            c_s=0,
        )
        ub_dict.update(
            c_thresh_itm=10,
            c_s=100,
        )

    elif simu_name == "4attn":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_distract",
            "s_fc",
            "gamma_fc",
            "c_thresh_itm",
            "psi_s",
            "psi_c",
            "c_d",
        ]
        lb_dict.update(
            c_thresh_itm=-10,
            psi_s=0,
            psi_c=-10,
        )
        ub_dict.update(
            c_thresh_itm=10,
            psi_s=100,
            psi_c=10,
        )

    elif simu_name == "5":

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "s_cf",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
        ]

    elif simu_name == "6a":

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "s_cf",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
        ]

    elif simu_name == "6b":

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "s_cf",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
        ]

    elif simu_name == "7":

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "s_cf",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
        ]

    elif simu_name == "8":

        # g1 cued recall (separate fit)
        # what_to_fit = [
        #     "beta_enc",
        #     "beta_rec",
        #     "beta_cue",
        #     "beta_distract",
        #     "gamma_fc",
        #     "gamma_cf",
        #     "s_fc",
        #     "phi_s",
        #     "phi_d",
        #     "kappa",
        #     "lamb",
        #     "eta",
        #     "omega",
        #     "alpha",
        #     "c_thresh",
        # ]

        # g2 associative recognition (separate fit)
        # what_to_fit = [
        #     "beta_enc",
        #     "beta_cue",
        #     "beta_distract",
        #     "beta_rec_post",
        #     "s_fc",
        #     "gamma_fc",
        #     "c_thresh_assoc",
        #     "c_d",
        # ]

        # Full fit: g1 cued recall + g2 recognition & final cued recall
        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_distract",
            "beta_rec_post",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
            "c_thresh_assoc",
            "c_d",
        ]
        ub_dict.update(
            s_fc=3.0,
        )

    elif simu_name == "S1":

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "gamma_cf",
            "s_fc",
            "s_cf",
            "phi_s",
            "phi_d",
            "kappa",
            "lamb",
            "eta",
            "omega",
            "alpha",
            "c_thresh",
            "c_thresh_itm",
            "c_thresh_assoc",
            "c_d",
        ]
        lb_dict.update(
            beta_enc=0.4,
            beta_rec=0,
            beta_cue=0.4,
            beta_distract=0,
            beta_rec_post=0.2,
            gamma_fc=0,
            gamma_cf=0,
            s_cf=0,
            s_fc=0,
            phi_s=1,
            phi_d=2,
            kappa=0,
            lamb=0,
            eta=0,
            omega=2,
            alpha=0.5,
            c_thresh=0,
            c_thresh_itm=0,
            c_thresh_assoc=0,
        )
        ub_dict.update(
            beta_enc=1,
            beta_rec=0.8,
            beta_cue=1,
            beta_distract=1,
            beta_rec_post=1,
            gamma_fc=1,
            gamma_cf=1,
            s_cf=0.6,
            s_fc=0.4,
            phi_s=5,
            phi_d=5,
            kappa=0.5,
            lamb=0.2,
            eta=0.2,
            omega=10,
            alpha=1,
            c_thresh=1,
            c_thresh_itm=2,
            c_thresh_assoc=2,
        )

    elif simu_name == "S2":

        what_to_fit = [
            "beta_enc",
            "beta_cue",
            "beta_rec_post",
            "beta_distract",
            "gamma_fc",
            "s_fc",
            "c_thresh_itm",
            "c_thresh_assoc",
            "thresh_sigma",
            "c_d",
        ]
        lb_dict.update(
            thresh_sigma=0,
        )
        ub_dict.update(
            thresh_sigma=0.2,
        )

    # Create lb and ub as list
    lb = [lb_dict[key] for key in what_to_fit]
    ub = [ub_dict[key] for key in what_to_fit]

    return lb, ub, what_to_fit


# --- Helpers for objective function --- #


_EVAL_TIMEOUT = float(os.environ.get("CMR_EVAL_TIMEOUT", 360))
_TIMEOUT_PENALTY = 1e6


class _EvalTimeout(Exception):
    """Raised when a single model evaluation exceeds _EVAL_TIMEOUT seconds."""


@contextmanager
def _time_limit(seconds):
    """Abort the wrapped block with _EvalTimeout after `seconds` wall-clock seconds.

    Relies on SIGALRM, which is delivered between Python bytecodes. This is safe
    here because the multi-session model runners loop over sessions in Python, so
    the alarm fires promptly between Cython per-session calls.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _EvalTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# --- Numpy helpers  --- #

def _codes(arr):
    """Integer codes for ``arr`` ordered by sorted unique values (pandas groupby order)."""
    uniq, inv = np.unique(arr, return_inverse=True)
    return uniq, inv


def _sess_collapse_mean(values, session, cats):
    """Two-level grouped mean: per (session, *cats), then averaged across sessions.

    ``cats`` is an (N, k) integer-code array. Reproduces
    ``df.groupby([session]+cats).mean().groupby(cats).mean()``: the returned
    ``ucats`` rows are the unique category combinations in lexicographic order
    with their session-collapsed means.
    """
    values = np.asarray(values, dtype=float)
    cats = np.asarray(cats, dtype=np.int64)
    if cats.ndim == 1:
        cats = cats[:, None]
    full = np.column_stack([np.asarray(session, dtype=np.int64), cats])
    uf, inv = np.unique(full, axis=0, return_inverse=True)
    cell = np.bincount(inv, weights=values) / np.bincount(inv)
    ucats, inv2 = np.unique(uf[:, 1:], axis=0, return_inverse=True)
    collapsed = np.bincount(inv2, weights=cell) / np.bincount(inv2)
    return ucats, collapsed


def _Yule_Q_smoothed(n11, n10, n01, n00):
    """Yule's Q with 0.5 continuity correction, matching the original crosstab order."""
    return Yule_Q(n11 + 0.5, n01 + 0.5, n10 + 0.5, n00 + 0.5)



def _pair_aligned(pair, test, value, t1=1, t2=2):
    """Per-pair mean of ``value`` at ``test==t1`` and ``test==t2``, aligned over common pairs.

    Reproduces a ``pivot_table(index=pair, columns=test, values=value)`` of the two
    test columns (default ``aggfunc='mean'``), keeping only pairs present at both tests.
    """
    value = np.asarray(value, dtype=float)
    m1, m2 = test == t1, test == t2
    up1, inv1 = np.unique(pair[m1], return_inverse=True)
    c1 = np.bincount(inv1, weights=value[m1]) / np.bincount(inv1)
    up2, inv2 = np.unique(pair[m2], return_inverse=True)
    c2 = np.bincount(inv2, weights=value[m2]) / np.bincount(inv2)
    common = np.intersect1d(up1, up2)
    return c1[np.searchsorted(up1, common)], c2[np.searchsorted(up2, common)]


def _contingency(a, b, categorical=True):
    """2x2 counts (n11, n10, n01, n00) of paired binary outcomes ``a`` (test1), ``b`` (test2).

    ``categorical=True`` mimics ``pd.Categorical(.., categories=[0/1])`` by dropping pairs
    whose mean is not exactly 0 or 1; ``False`` truncates to int (``to_numpy(dtype=int)``).
    """
    if categorical:
        keep = np.isin(a, (0.0, 1.0)) & np.isin(b, (0.0, 1.0))
        a, b = a[keep], b[keep]
    a, b = a.astype(int), b.astype(int)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    return n11, n10, n01, n00


def _roc_interp(far, hr, xs):
    """Piecewise-linear interpolation of ``hr`` onto ``far`` at points ``xs`` (sorted far)."""
    out = []
    for x in xs:
        idx = np.searchsorted(far, x)
        if idx < len(far):
            v = hr[idx - 1] + (x - far[idx - 1]) * (hr[idx] - hr[idx - 1]) / (far[idx] - far[idx - 1])
        else:
            v = hr[idx - 1]
        out.append(v)
    return np.array(out)


# --- Per-simulation stats helpers --- #

def _simu1_stats(df_simu, gt):
    """simu1 continuous recognition: HR / FAR by log-lag and rolling category similarity."""
    df_simu = df_simu.copy()
    n = len(df_simu)

    # Rolling count of same-category items in the trailing window of 9 trials
    rolling_window = 9
    cat_dummies = df_simu["category_label"].str.get_dummies()
    cat_dummies.columns = ["cl_" + col for col in cat_dummies.columns]
    events = pd.concat([df_simu, cat_dummies], axis=1)
    cl_rolling_sum = events.groupby("session").rolling(rolling_window, min_periods=1, on="position")[cat_dummies.columns].sum().reset_index()
    df_rollcat = df_simu.merge(cl_rolling_sum, on=["session", "position"])
    clcols = list(cat_dummies.columns)
    col_index = {c: i for i, c in enumerate(clcols)}
    code = np.array([col_index["cl_" + c] for c in df_simu["category_label"].to_numpy()])
    df_simu["roll_cat_label_length"] = df_rollcat[clcols].to_numpy()[np.arange(n), code] - 1
    df_simu["roll_cat_len_level"] = pd.cut(x=df_simu.roll_cat_label_length, bins=[0, 2, np.inf], right=False, include_lowest=True, labels=["0-1", ">=2"]).astype("str")

    # Log-lag bin (collapse bin 1 into 0, cap at 5)
    log_lag = np.log(df_simu["lag"].to_numpy())
    bins = pd.cut(log_lag, np.arange(np.max(log_lag) + 1), labels=False, right=False).astype(float)
    bins = np.where(bins == 1, 0, bins)
    bins = np.where(bins > 5, 5, bins)

    # Local FAR: carry a neighbouring old item's lag bin onto a new item (vectorised)
    old_vec = df_simu["old"].to_numpy().astype(bool)
    position_vec = df_simu["position"].to_numpy()
    max_position = np.max(position_vec)
    prev_old = np.concatenate([[False], old_vec[:-1]])
    next_old = np.concatenate([old_vec[1:], [False]])
    prev_bin = np.concatenate([[None], bins[:-1].astype(object)])
    next_bin = np.concatenate([bins[1:].astype(object), [None]])
    newpre = np.where((position_vec > 0) & (~old_vec) & prev_old, prev_bin, "N")
    newpost = np.where((position_vec < max_position) & (~old_vec) & next_old, next_bin, "N")

    # Membership of each log-lag bin (own bin or a carried neighbour bin)
    log_lag_bins = [0, 2, 3, 4, 5]
    for b in log_lag_bins:
        df_simu["log_lag_bin_" + str(b)] = (bins == b) | (newpre == b) | (newpost == b)

    # Clean the first 20 trials
    df_simu = df_simu.query("position >= 20").copy()

    # Yes rate per (session, old, similarity level) for each log-lag bin
    df_lst = []
    for b in log_lag_bins:
        col_name = "log_lag_bin_" + str(b)
        df_tmp = df_simu.query(col_name + " == True").groupby(["session", "old", "roll_cat_len_level"])["s_resp"].agg(["mean", "sum", "count"]).reset_index()
        df_tmp["log_lag_bin"] = b
        df_lst.append(df_tmp)
    df_laggp = pd.concat(df_lst)
    df_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)

    # Pivot to HR / FAR and collapse across sessions
    df_laggp["log_lag_disp"] = np.ceil(np.e ** df_laggp.log_lag_bin)
    df_laggp["old"] = df_laggp["old"].astype("str")
    df_dprime = pd.pivot_table(df_laggp, values=["yes_rate"], index=["session", "roll_cat_len_level", "log_lag_disp"], columns="old").reset_index()
    df_dprime.columns = [" ".join(col).strip() for col in df_dprime.columns.values]
    df_dprime = df_dprime.rename(columns={"yes_rate False": "far", "yes_rate True": "hr"})
    df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].mean().reset_index()
    hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["hr"].to_numpy()
    hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["hr"].to_numpy()
    far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["far"].to_numpy()
    far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["far"].to_numpy()

    err = wmse(np.array(gt["hr_lowsim"]), hr_lowsim, np.array(gt["hr_lowsim_std"])) \
        + wmse(np.array(gt["hr_highsim"]), hr_highsim, np.array(gt["hr_highsim_std"])) \
        + wmse(np.array(gt["far_lowsim"]), far_lowsim, np.array(gt["far_lowsim_std"])) \
        + wmse(np.array(gt["far_highsim"]), far_highsim, np.array(gt["far_highsim_std"]))
    return hr_lowsim, hr_highsim, far_lowsim, far_highsim, err


def _simu2_stats(df_simu, gt):
    """simu2 continuous recognition ROC: interpolated HR for adjacent / remote lags."""
    old_lag = df_simu["old_lag"].to_numpy()
    abslag = np.abs(old_lag)
    code = np.zeros(len(old_lag), dtype=np.int64)  # 0 = nan, 1 = 'a', 2 = 'r'
    code[(old_lag != -999) & (abslag == 1)] = 1
    code[(old_lag != -999) & (abslag > 10)] = 2

    # Carry the previous trial's lag category onto a new item following an old item
    old = df_simu["old"].to_numpy()
    recog_pos = df_simu["recog_pos"].to_numpy()
    prev_code = np.empty_like(code)
    prev_code[1:] = code[:-1]
    prev_code[0] = 0
    prev_old = np.empty_like(old)
    prev_old[1:] = old[:-1]
    prev_old[0] = 0
    carry = (recog_pos > 1) & (old == 0) & (prev_old == 1)
    new_code = np.where(carry, prev_code, code)

    # Keep classified trials; level = old*2 + (lag == 'a'): 0 new_r, 1 new_a, 2 old_r, 3 old_a
    keep = new_code != 0
    level = (old[keep] * 2 + (new_code[keep] == 1)).astype(np.int64)
    csim = df_simu["csim"].to_numpy()[keep]
    base_thresh = df_simu["thresh"].to_numpy()[keep]
    session = df_simu["session"].to_numpy()[keep]

    # (session, level) cells, then collapse cells to per-level means
    cells, cell_inv = np.unique(np.column_stack([session, level]), axis=0, return_inverse=True)
    cell_level = cells[:, 1]
    cell_cnt = np.bincount(cell_inv)
    level_cnt = np.bincount(cell_level, minlength=4)

    thresh_arr = np.arange(0, 2, 0.001)
    roc = np.empty((len(thresh_arr), 4))
    for ti, t in enumerate(thresh_arr):
        above = (csim > t * base_thresh).astype(float)
        cell_mean = np.bincount(cell_inv, weights=above, minlength=len(cells)) / cell_cnt
        roc[ti] = np.bincount(cell_level, weights=cell_mean, minlength=4) / level_cnt

    far_a, hr_a = np.sort(roc[:, 1]), np.sort(roc[:, 3])
    far_r, hr_r = np.sort(roc[:, 0]), np.sort(roc[:, 2])

    far_a_gt, hr_a_gt = np.array(gt["far_a"]), np.array(gt["hr_a"])
    far_r_gt, hr_r_gt = np.array(gt["far_r"]), np.array(gt["hr_r"])
    hr_a_interp = _roc_interp(far_a, hr_a, far_a_gt)
    hr_r_interp = _roc_interp(far_r, hr_r, far_r_gt)
    err = np.power(hr_a_interp - hr_a_gt, 2).sum() + np.power(hr_r_interp - hr_r_gt, 2).sum()

    # Penalise ROCs where adjacent does not dominate remote
    above_range = np.arange(0.08, 0.61, 0.01)
    if not np.all(_roc_interp(far_a, hr_a, above_range) > _roc_interp(far_r, hr_r, above_range)):
        err += 0.5
    if not np.all((hr_a_interp > hr_r_interp)[1:]):
        err += 0.5
    return hr_a_interp, hr_r_interp, err


def _simu2b_stats(df_simu, gt):
    """simu2b associative continuous recognition: overall HR + FAR by lag."""
    session = df_simu["session"].to_numpy()
    typ = df_simu["type"].to_numpy()
    s_resp = df_simu["s_resp"].to_numpy()
    lag = df_simu["lag"].to_numpy()
    old = (typ == "intact").astype(int)
    correct = (s_resp == old).astype(float)

    # HR: session-collapsed correct rate for intact pairs
    utype, tcode = _codes(typ)
    ucats, type_mean = _sess_collapse_mean(correct, session, tcode)
    hr = type_mean[np.where(utype == "intact")[0][0]]

    # FAR by lag among rearranged pairs
    rmask = typ == "rearranged"
    _, lag_mean = _sess_collapse_mean(correct[rmask], session[rmask], lag[rmask])
    far = 1 - lag_mean

    err = 5 * wmse(np.array(gt["hr"]), hr, np.array(gt["hr_std"])) + wmse(np.array(gt["far"]), far, np.array(gt["far_std"]))
    return hr, far, err


def _simu3_stats(df_simu, gt):
    """simu3 recognition & forgetting: item / associative HR & FAR by lag."""
    session = df_simu["session"].to_numpy()
    lag = df_simu["lag"].to_numpy()
    s_resp = df_simu["s_resp"].to_numpy(dtype=float)
    utype, tcode = _codes(df_simu["type"].to_numpy())
    cats = np.column_stack([tcode, lag])
    ucats, collapsed = _sess_collapse_mean(s_resp, session, cats)

    def sel(name):
        return collapsed[ucats[:, 0] == np.where(utype == name)[0][0]]

    I_hr = sel("single_old")
    I_far = np.mean(sel("single_new"))
    A_hr = sel("pair_old")
    A_far = sel("pair_new")

    I_hr_gt = np.array(gt["I_hr"])
    I_far_gt = np.array(gt["I_far"])
    A_hr_gt = np.array(gt["A_hr"])
    A_far_gt = 1 - np.array(gt["A_cr"])
    err = np.sum((I_hr - I_hr_gt) ** 2) + np.sum((A_hr - A_hr_gt) ** 2) + (I_far - I_far_gt) ** 2 * 5 + np.sum((A_far - A_far_gt) ** 2)
    return I_hr, I_far, A_hr, A_far, err


def _simu4_stats(df_simu, gt):
    """simu4 word-frequency effect: HR / FAR by quantile."""
    session = df_simu["session"].to_numpy()
    quantile = df_simu["quantile"].to_numpy()
    old = df_simu["old"].to_numpy().astype(int)
    s_resp = df_simu["s_resp"].to_numpy(dtype=float)
    cats = np.column_stack([quantile, old])
    ucats, collapsed = _sess_collapse_mean(s_resp, session, cats)
    hr = collapsed[ucats[:, 1] == 1]
    far = collapsed[ucats[:, 1] == 0]
    err = wmse(np.array(gt["hr"]), hr, np.array(gt["hr_std"])) + wmse(np.array(gt["far"]), far, np.array(gt["far_std"]))
    return hr, far, err


def _simu5_stats(df_simu, gt):
    """simu5 cued recall: correct rate by study-test lag."""
    session = df_simu["session"].to_numpy()
    lag = df_simu["lag"].to_numpy()
    correct = df_simu["correct"].to_numpy(dtype=float)
    _, hr = _sess_collapse_mean(correct, session, lag)
    err = wmse(np.array(gt["hr"]), hr, np.array(gt["hr_std"]))
    return hr, err


def _simu6a_stats(df_simu, gt):
    """simu6a symmetric cued recall: forward / backward correct rate by lag."""
    mask = df_simu["list"].to_numpy() > 1
    session = df_simu["session"].to_numpy()[mask]
    lag = df_simu["lag"].to_numpy()[mask]
    order = df_simu["order"].to_numpy()[mask]
    correct = df_simu["correct"].to_numpy(dtype=float)[mask]
    cats = np.column_stack([lag, order])
    ucats, collapsed = _sess_collapse_mean(correct, session, cats)
    fw = collapsed[ucats[:, 1] == 1]
    bw = collapsed[ucats[:, 1] == 2]
    err = np.power(fw - np.array(gt["fw"]), 2).sum() + np.power(bw - np.array(gt["bw"]), 2).sum()
    return fw, bw, err


def _simu6b_subj_stats(df_simu):
    """simu6b per-subject joint test1/test2 proportions and Yule's Q. Requires a 'correct' column."""
    pair = df_simu["pair_idx"].to_numpy()
    test = df_simu["test"].to_numpy()
    correct = df_simu["correct"].to_numpy().astype(int)
    a, b = _pair_aligned(pair, test, correct)
    n11, n10, n01, n00 = _contingency(a, b, categorical=True)
    N = n11 + n10 + n01 + n00
    q = _Yule_Q_smoothed(n11, n10, n01, n00)
    return n11 / N, n10 / N, n01 / N, n00 / N, q


def _simu6b_stats(df_simu, gt):
    """simu6b symmetric cued recall: identical vs reversed test-order pair statistics."""
    df_simu = df_simu.copy()
    df_simu["correct"] = df_simu["s_resp"] == df_simu["correct_ans"]
    session = df_simu["session"].to_numpy()
    pair = df_simu["pair_idx"].to_numpy()
    test = df_simu["test"].to_numpy()
    order = df_simu["order"].to_numpy(dtype=float)

    # Congruence per pair from the test-direction at test1 vs test2
    o1, o2 = _pair_aligned(pair, test, order)
    common = np.intersect1d(pair[test == 1], pair[test == 2])
    identical = ((o1 == 1) & (o2 == 1)) | ((o1 == 2) & (o2 == 2))
    cong_map = dict(zip(common, identical))
    df_simu["cong"] = np.array([cong_map[p] for p in pair])

    inde_stats, reve_stats = [], []
    for subj in np.unique(session):
        ssel = df_simu["session"] == subj
        for flag, store in ((True, inde_stats), (False, reve_stats)):
            store.append(list(_simu6b_subj_stats(df_simu[ssel & (df_simu["cong"] == flag)])))
    inde_mean = np.mean(inde_stats, axis=0)
    reve_mean = np.mean(reve_stats, axis=0)
    inde_gt = np.array(gt["inde"])
    reve_gt = np.array(gt["reve"])
    err = np.sum(np.power(inde_mean - inde_gt, 2)) + np.sum(np.power(reve_mean - reve_gt, 2)) \
        + np.power(inde_mean[-1] - inde_gt[-1], 2) + np.power(reve_mean[-1] - reve_gt[-1], 2)
    return inde_mean, reve_mean, err


def _simu7_resp_origin(df_simu, df_study):
    """For each response, find its study (list, pair-position); NaN for non-responses (-1/-2)."""
    nlist = len(np.unique(df_simu["list"].to_numpy()))
    resp_list = np.full(len(df_simu), np.nan)
    resp_pos = np.full(len(df_simu), np.nan)
    study_sess = df_study["session"].to_numpy()
    item1 = df_study["study_itemno1"].to_numpy()
    item2 = df_study["study_itemno2"].to_numpy()
    sim_sess = df_simu["session"].to_numpy()
    s_resp = df_simu["s_resp"].to_numpy()
    for sess in np.unique(sim_sess):
        sm = study_sess == sess
        pres = np.stack([item1[sm], item2[sm]], axis=1).reshape(nlist, -1, 2)
        lookup = {}
        for li in range(pres.shape[0]):
            for pi in range(pres.shape[1]):
                for k in range(2):
                    lookup.setdefault(int(pres[li, pi, k]), (li, pi))
        rows = np.where(sim_sess == sess)[0]
        for ridx in rows:
            r = s_resp[ridx]
            if r == -1 or r == -2:
                continue
            li, pi = lookup[int(r)]
            resp_list[ridx] = li
            resp_pos[ridx] = pi
    return resp_list, resp_pos


def _simu7_stats(df_simu, df_study, gt):
    """simu7 cued recall: correct / PLI / ILI probabilities and their lag distributions."""
    resp_list, resp_pos = _simu7_resp_origin(df_simu, df_study)
    list_no = df_simu["list"].to_numpy()
    study_pos = df_simu["study_pos"].to_numpy()
    session = df_simu["session"].to_numpy()
    list_lag = resp_list - list_no
    pos_lag = resp_pos - study_pos

    # Intrusion code: 0 NoResp, 1 Correct, 2 PLI, 3 ILI, -1 other (forward intrusion)
    intr = np.full(len(df_simu), -1, dtype=int)
    noresp = np.isnan(resp_list)
    intr[noresp] = 0
    valid = ~noresp
    intr[valid & (list_lag == 0) & (pos_lag == 0)] = 1
    intr[valid & (list_lag < 0)] = 2
    intr[valid & (list_lag == 0) & (pos_lag != 0)] = 3

    # Keep list > 0
    keep = list_no > 0
    session, intr, list_no = session[keep], intr[keep], list_no[keep]
    list_lag, pos_lag, study_pos = list_lag[keep], pos_lag[keep], study_pos[keep]

    # Overall correct / PLI / ILI probability (per-session rate, averaged over sessions)
    usess, sidx = np.unique(session, return_inverse=True)
    total = np.bincount(sidx, minlength=len(usess))
    p_correct_mean = np.mean(np.bincount(sidx[intr == 1], minlength=len(usess)) / total)
    p_ILI_mean = np.mean(np.bincount(sidx[intr == 3], minlength=len(usess)) / total)
    p_PLI_mean = np.mean(np.bincount(sidx[intr == 2], minlength=len(usess)) / total)

    try:
        # PLI list-lag distribution (lists > 5, lag in -5..-1)
        fm = (intr == 2) & (list_no > 5) & (list_lag > -6)
        ps, pinv = np.unique(session[fm], return_inverse=True)
        cnt_sess = np.bincount(pinv, minlength=len(ps))
        al = np.abs(list_lag[fm]).astype(int)
        lag_PLI_mean = np.array([np.mean(np.bincount(pinv[al == lag], minlength=len(ps)) / cnt_sess) for lag in [1, 2, 3, 4, 5]])

        # ILI position-lag distribution (lag in -5..-1, 1..5), normalised by reachable positions
        im = intr == 3
        lag_vals = list(range(-5, 0)) + list(range(1, 6))
        rows = {lag: [] for lag in lag_vals}
        for sess in np.unique(session[im]):
            sm = (session == sess) & im
            sp_sess = study_pos[sm]
            pl_sess = pos_lag[sm].astype(int)
            possible = {}
            for pair_pos in sp_sess:
                for i in range(-int(pair_pos), 11 - int(pair_pos) + 1):
                    possible[i] = possible.get(i, 0) + 1
            for lag in lag_vals:
                cnt = np.sum(pl_sess == lag)
                rows[lag].append(cnt / possible[lag] if lag in possible else np.nan)
        lag_ILI_mean = np.array([np.nanmean(rows[lag]) for lag in lag_vals])
    except Exception:
        lag_PLI_mean = np.full(5, 0)
        lag_ILI_mean = np.full(10, 0)

    wls_p_correct = wmse(np.array(gt["p_correct_mean"]), p_correct_mean, np.array(gt["p_correct_se"]))
    wls_p_PLI = wmse(np.array(gt["p_PLI_mean"]), p_PLI_mean, np.array(gt["p_PLI_se"]))
    wls_p_ILI = wmse(np.array(gt["p_ILI_mean"]), p_ILI_mean, np.array(gt["p_ILI_se"]))
    wls_lag_PLI = wmse(np.array(gt["lag_PLI_mean"]), lag_PLI_mean, np.array(gt["lag_PLI_se"])) / len(gt["lag_PLI_mean"])
    wls_lag_ILI = wmse(np.array(gt["lag_ILI_mean"]), lag_ILI_mean, np.array(gt["lag_ILI_se"])) / len(gt["lag_ILI_mean"])
    err = wls_p_correct + wls_p_PLI + wls_p_ILI + wls_lag_PLI + wls_lag_ILI
    return p_correct_mean, p_PLI_mean, p_ILI_mean, lag_PLI_mean, lag_ILI_mean, err


def _simu8_name_face_lookup(df_study):
    """Build a vectorized per-session (study name -> studied face) lookup."""
    sess = df_study["session"].to_numpy(dtype=np.int64)
    name = df_study["study_itemno2"].to_numpy(dtype=np.int64)
    face = df_study["study_itemno1"].to_numpy(dtype=np.int64)
    base = int(name.max()) + 1
    keys = sess * base + name
    order = np.argsort(keys, kind="stable")
    keys_sorted, face_sorted = keys[order], face[order]

    def lookup(sess_q, name_q):
        q = sess_q.astype(np.int64) * base + name_q.astype(np.int64)
        return face_sorted[np.searchsorted(keys_sorted, q)]

    return lookup


def _simu8_ili_bin(d):
    """ILI distance -> bin index 0..6 (cats 1.0,1.5,2.0,2.5,3.0,3.5,>3.5); -1 excluded."""
    d = np.asarray(d, dtype=float)
    idx = np.floor(2.0 * d).astype(int) - 1  # d in [0.5,1.0) -> 0 ("1.0")
    idx = np.where(d >= 3.5, 6, idx)  # ">3.5"
    idx = np.where(idx < 0, -1, idx)  # d < 0.5 excluded
    return idx


def _simu8_cut_bin(x, edges):
    """pd.cut-equivalent (right-closed): 0-based label index, -1 if outside range."""
    x = np.asarray(x, dtype=float)
    d = np.digitize(x, edges, right=True)  # 0..len(edges)
    return np.where((d < 1) | (d >= len(edges)), -1, d - 1)


def _simu8_grouped_mean(values, group_idx, n_bins):
    """Mean of ``values`` per bin index in [0, n_bins); NaN where empty. Returns (mean, count)."""
    valid = group_idx >= 0
    cnt = np.bincount(group_idx[valid], minlength=n_bins)[:n_bins]
    summ = np.bincount(group_idx[valid], weights=values[valid], minlength=n_bins)[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = summ / cnt
    out[cnt == 0] = np.nan
    return out, cnt


def _simu8_g1_stats(df_simu, df_study_g1, face_distance, thresh, gt):
    """g1 cued recall: neighbourhood effect on correct rate + ILI by distance bin."""
    sess = df_simu["session"].to_numpy(dtype=np.int64)
    test_itemno = df_simu["test_itemno"].to_numpy(dtype=np.int64)
    s_resp = df_simu["s_resp"].to_numpy(dtype=np.int64)
    correct_ans = df_simu["correct_ans"].to_numpy(dtype=np.int64)
    correct = s_resp == correct_ans
    correct_rate = correct.mean()

    # Group rows by session (each session has exactly 8 unique test faces)
    order = np.argsort(sess, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    S = len(sess) // 8
    F = test_itemno[order].reshape(S, 8) - 1  # face indices, (S, 8)

    # Neighbour count: other session faces within thresh (exclude self)
    A = face_distance < thresh  # includes diagonal (dist 0)
    neighbour = (A[F[:, :, None], F[:, None, :]].sum(axis=2) - 1).ravel()[inv]
    ng = np.where(neighbour == 7, 6, neighbour)
    mask = ng > 0
    cnt = np.bincount(ng[mask])
    summ = np.bincount(ng[mask], weights=correct[mask].astype(float))
    present = np.where(cnt > 0)[0]
    present = present[present > 0]
    neighbor_mean = summ[present] / cnt[present]

    # ILI possible counts: off-diagonal session distances, binned
    Dsub = face_distance[F[:, :, None], F[:, None, :]]
    off = ~np.eye(8, dtype=bool)
    poss_idx = _simu8_ili_bin(Dsub[:, off].ravel())
    ILI_poss = np.bincount(poss_idx[poss_idx >= 0], minlength=7)[:7]

    try:
        # is_studied: s_resp among the session's correct answers
        CA = correct_ans[order].reshape(S, 8)
        SR = s_resp[order].reshape(S, 8)
        is_studied = (SR[:, :, None] == CA[:, None, :]).any(axis=2).ravel()[inv]
        is_ILI = is_studied & (~correct)

        lookup = _simu8_name_face_lookup(df_study_g1)
        resp_face = lookup(sess[is_ILI], s_resp[is_ILI])
        resp_corr_d = face_distance[test_itemno[is_ILI] - 1, resp_face - 1]
        ili_idx = _simu8_ili_bin(resp_corr_d)
        ILI_cnt = np.bincount(ili_idx[ili_idx >= 0], minlength=7)[:7]
        with np.errstate(invalid="ignore", divide="ignore"):
            ILI_mean = ILI_cnt / ILI_poss
    except Exception:
        ILI_mean = np.full(7, 0)

    neighbor_mean_gt = np.array(gt["exp1_neighbor_mean"])
    neighbor_se_gt = np.array(gt["exp1_neighbor_se"])
    ILI_mean_gt = np.array(gt["exp1_ILI_mean"])
    ILI_se_gt = np.array(gt["exp1_ILI_se"])
    wls_neighbor = wmse(neighbor_mean_gt, neighbor_mean, neighbor_se_gt) / len(neighbor_mean_gt)
    wls_ILI = wmse(ILI_mean_gt, ILI_mean, ILI_se_gt) / len(ILI_mean_gt)
    return neighbor_mean, ILI_mean, correct_rate, wls_neighbor, wls_ILI


def _simu8_g2_stats(df_recog, df_cr, df_study_g2, face_distance, thresh, gt):
    """g2: recognition density (HR/FAR) + probe-distance yes rate + final cued recall."""
    lookup = _simu8_name_face_lookup(df_study_g2)

    # Recognition: density effect on HR / FAR
    r_sess = df_recog["session"].to_numpy(dtype=np.int64)
    r_f1 = df_recog["test_itemno1"].to_numpy(dtype=np.int64)
    r_f2 = df_recog["test_itemno2"].to_numpy(dtype=np.int64)
    r_corr = df_recog["correct_ans"].to_numpy(dtype=np.int64)
    r_yes = (df_recog["s_resp"].to_numpy(dtype=np.int64) == 1).astype(float)

    neighbour_count = ((face_distance < thresh) & (face_distance > 0)).sum(axis=1)
    dens = _simu8_cut_bin(neighbour_count[r_f1 - 1], [4, 6, 8, 10])  # 0 low,1 med,2 high
    hr_all, hr_cnt = _simu8_grouped_mean(r_yes[r_corr == 1], dens[r_corr == 1], 3)
    far_all, far_cnt = _simu8_grouped_mean(r_yes[r_corr == 0], dens[r_corr == 0], 3)
    hr_mean = hr_all[hr_cnt > 0]  # observed=True drops empty densities
    far_mean = far_all[far_cnt > 0]

    # Recognition: probe distance effect on yes rate (Targets + 4 lure bins)
    lure = r_corr != 1
    probe_d = face_distance[r_f1[lure] - 1, lookup(r_sess[lure], r_f2[lure]) - 1]
    yesdist_mean = np.full(5, np.nan)
    yesdist_mean[0] = r_yes[r_corr == 1].mean() if (r_corr == 1).any() else np.nan
    lure_mean, _ = _simu8_grouped_mean(r_yes[lure], _simu8_cut_bin(probe_d, [0.5, 1.5, 2.5, 3.5, 4.5]), 4)
    yesdist_mean[1:] = lure_mean

    # Final cued recall by recalled-name distance
    c_sess = df_cr["session"].to_numpy(dtype=np.int64)
    c_f1 = df_cr["test_itemno1"].to_numpy(dtype=np.int64)
    c_resp = df_cr["s_resp"].to_numpy(dtype=np.int64)
    correct = c_resp == df_cr["correct_ans"].to_numpy(dtype=np.int64)
    recalled = c_resp > 0
    rec_f1 = c_f1[recalled]
    resp_face = rec_f1.copy()  # correct -> own face
    inc = ~correct[recalled]
    resp_face[inc] = lookup(c_sess[recalled][inc], c_resp[recalled][inc])
    fin_edges = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]  # labels 0, 1.5, 2.5, 3.5, 4.5
    cr_idx = _simu8_cut_bin(face_distance[rec_f1 - 1, resp_face - 1], fin_edges)
    recall_cnt = np.bincount(cr_idx[cr_idx >= 0], minlength=5)[:5]
    poss_idx = _simu8_cut_bin(face_distance[c_f1 - 1, :].ravel(), fin_edges)
    recall_poss = np.bincount(poss_idx[poss_idx >= 0], minlength=5)[:5]
    with np.errstate(invalid="ignore", divide="ignore"):
        crdist_mean = recall_cnt / recall_poss

    hr_mean_gt = np.array(gt["exp3_neighbor_hr_mean"])
    hr_se_gt = np.array(gt["exp3_neighbor_hr_se"])
    far_mean_gt = np.array(gt["exp3_neighbor_far_mean"])
    far_se_gt = np.array(gt["exp3_neighbor_far_se"])
    yesdist_mean_gt = np.array(gt["exp3_yesdist_mean"])
    yesdist_se_gt = np.array(gt["exp3_yesdist_se"])
    crdist_mean_gt = np.array(gt["exp3_crdist_mean"])
    crdist_se_gt = np.array(gt["exp3_crdist_se"])
    wls_hr = wmse(hr_mean_gt, hr_mean, hr_se_gt) / len(hr_mean_gt)
    wls_far = wmse(far_mean_gt, far_mean, far_se_gt) / len(far_mean_gt)
    wls_yesdist = wmse(yesdist_mean_gt, yesdist_mean, yesdist_se_gt) / len(yesdist_mean_gt)
    wls_crdist = wmse(crdist_mean_gt, crdist_mean, crdist_se_gt) / len(crdist_mean_gt)
    return (hr_mean, far_mean, yesdist_mean, crdist_mean, wls_hr, wls_far, wls_yesdist, wls_crdist)


def _simuS1_subj_stats(df_simu):
    """simuS1 per-subject cued-recall rate, recognition HR/FAR, and successive-test Q."""
    test = df_simu["test"].to_numpy()
    s_resp = df_simu["s_resp"].to_numpy()
    correct_ans = df_simu["correct_ans"].to_numpy()
    pair_idx = df_simu["pair_idx"].to_numpy()
    correct = (s_resp == correct_ans).astype(int)

    rmask = test == 1
    is_old = correct_ans[rmask]
    resp = s_resp[rmask]
    hr = np.sum(resp * is_old) / np.sum(is_old)
    far = np.sum(resp * (1 - is_old)) / np.sum(1 - is_old)

    cmask = test == 2
    p_rc = np.mean(s_resp[cmask] == correct_ans[cmask])

    smask = pair_idx >= 0
    a, b = _pair_aligned(pair_idx[smask], test[smask], correct[smask])
    n11, n10, n01, n00 = _contingency(a, b, categorical=False)
    q = _Yule_Q_smoothed(n11, n10, n01, n00)
    return p_rc, hr, far, q


def _simuS1_stats(dfs, gt):
    """simuS1: stack per-group means and score against ground truth (base err, no constraint)."""
    stats = []
    for df_gp in dfs:
        subjects = np.unique(df_gp["subject"].to_numpy())
        stats_gp = [list(_simuS1_subj_stats(df_gp[df_gp["subject"] == subj])) for subj in subjects]
        stats.append(list(np.mean(stats_gp, axis=0)))
    stats = np.array(stats)
    ground_truth = np.array([gt["g1_mean"], gt["g2_mean"], gt["g3_mean"]])
    err = np.sum(np.power(stats - ground_truth, 2))
    return stats, err


def _simuS2_condition(typ, correct_ans):
    """Map (type, correct_ans) to the seven simuS2 conditions (matching get_cond)."""
    cond = np.full(len(typ), None, dtype=object)
    for name in ["Different_Item", "Item_Pair", "Pair_Item", "Same_Item", "Intact_Pair"]:
        cond[(correct_ans == 1) & (typ == name)] = name
    cond[(correct_ans == 0) & (typ == "extra")] = "NR_Lure"
    cond[(correct_ans == 0) & ((typ == "Same_Item") | (typ == "Intact_Pair"))] = "Repeated_Lure"
    cond[(correct_ans == 0) & ~((typ == "extra") | (typ == "Same_Item") | (typ == "Intact_Pair"))] = "Discard"
    return cond


def _simuS2_subj_stats(df_simu):
    """simuS2 per-subject [Test1_p, Test2_p, Q] for each condition. Requires a 'condition' column."""
    _SIMUS2_CONDS = ["Different_Item", "Item_Pair", "Pair_Item", "Same_Item", "Intact_Pair", "Repeated_Lure", "NR_Lure"]
    condition = df_simu["condition"].to_numpy()
    test = df_simu["test"].to_numpy()
    s_resp = df_simu["s_resp"].to_numpy()
    pair_idx = df_simu["pair_idx"].to_numpy()
    correct = (s_resp == df_simu["correct_ans"].to_numpy()).astype(int)
    stats = []
    for cond in _SIMUS2_CONDS:
        cm = condition == cond
        m1, m2 = cm & (test == 1), cm & (test == 2)
        t1p = s_resp[m1].mean() if m1.any() else np.nan
        t2p = s_resp[m2].mean() if m2.any() else np.nan
        if cond == "NR_Lure":  # excluded from the pair table; original except-branch -> 0
            q = 0.0
        elif not cm.any():  # no pairs -> empty crosstab raises in original -> nan
            q = np.nan
        else:
            a, b = _pair_aligned(pair_idx[cm], test[cm], correct[cm])
            q = _Yule_Q_smoothed(*_contingency(a, b, categorical=True))
        stats.append([t1p, t2p, q])
    return stats


def _simuS2_stats(df_simu, gt):
    """simuS2 successive recognition: per-condition Test1_p, Test2_p, and Q vs ground truth."""
    df_simu = df_simu.copy()
    df_simu["condition"] = _simuS2_condition(df_simu["type"].to_numpy(), df_simu["correct_ans"].to_numpy())
    subject = df_simu["subject"].to_numpy()
    list_no = df_simu["list"].to_numpy()

    stats = []
    for subj in np.unique(subject):
        m = (subject == subj) & (list_no % 3 != 0)
        stats.append(_simuS2_subj_stats(df_simu[m]))
    stats_mean = np.nanmean(stats, axis=0)
    _conds = ["diff_item", "item_pair", "pair_item", "same_item", "intact_pair", "rep_lure", "nrep_lure"]
    ground_truth = np.array([gt[f"{c}_mean"] for c in _conds])
    err = np.sum(np.power(stats_mean - ground_truth, 2))
    return stats_mean, err


# --- Objective function --- #

def obj_func(param_vec, df_study, df_test, sem_mat, sources, simu_name, return_df=False):
    """
    Combined objective function for all simulations. Dispatches on simu_name.
    return_df is only used by simu_name == "S1".
    """

    # Reformat parameter vector to the dictionary format expected by CMR
    param_dict = param_vec_to_dict(param_vec, simu_name=simu_name)


    ## SIMU1 ##
    if simu_name == "1":

        assert df_study is None
        df = df_test

        # Run model
        df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, design="EXP1", disable_tqdm=True)
        df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

        # Calculate error
        with open("../../Analysis/simu1_recog_recsim/data/simu1_gt.json") as f:
            _gt = json.load(f)
        _, _, _, _, err = _simu1_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": []}


    ## SIMU2 ##
    elif simu_name == "2":

        # Run model
        df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])

        # Calculate err
        with open("../../Analysis/simu2_recog_conti/data/simu2_gt.json") as f:
            _gt = json.load(f)
        hr_a_interp, hr_r_interp, err = _simu2_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr_a_interp, hr_r_interp]}


    ## SIMU2b ##
    elif simu_name == "2b":

        # Run model
        df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, design="Osth", disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])

        # Calculate err
        with open("../../Analysis/simu2b_recog_assoc_conti/data/simu2b_gt.json") as f:
            _gt = json.load(f)
        hr, far, err = _simu2b_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr, far]}


    ## SIMU3 ##
    elif simu_name == "3":

        assert df_study is None
        df = df_test

        # Run model
        df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, design="Hockley", disable_tqdm=True)
        df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

        # Calculate err
        with open("../../Analysis/simu3_recog_forget/data/simu3_gt.json") as f:
            _gt = json.load(f)
        I_hr, I_far, A_hr, A_far, err = _simu3_stats(df_simu, _gt)

        # Apply some constraints
        if np.any(np.diff(I_hr) > 0):
            err += 1
        if np.any(np.diff(A_hr) > 0):
            err += 1
        if np.any(np.diff(A_far) > 0):
            err += 1
        if np.any(I_hr < A_hr):
            err += 1
        cmr_stats = {"err": err, "params": param_vec, "stats": [I_hr, I_far, A_hr, A_far]}


    ## SIMU4 ##
    elif simu_name in ("4", "4base", "4shift", "4attn"):

        # Run model
        df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
        if simu_name == "4":
            df_simu = df_simu.merge(df_test, on=["session", "list", "itemno"])
        else:
            df_simu = df_simu.merge(df_test, on=["session", "itemno"])

        # Get behavioral stats and compare with ground truth
        with open("../../Analysis/simu4_recog_wfe/data/simu4_gt.json") as f:
            _gt = json.load(f)
        hr, far, err = _simu4_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr, far]}


    ## SIMU5 ##
    elif simu_name == "5":

        # Run model
        param_dict.update(nitems_in_accumulator=48)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Get error
        with open("../../Analysis/simu5_cr_rec/data/simu5_gt.json") as f:
            _gt = json.load(f)
        hr, err = _simu5_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": hr}


    ## SIMU6a ##
    elif simu_name == "6a":

        # Run model
        param_dict.update(nitems_in_accumulator=48)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Get error (helper drops the first list internally)
        with open("../../Analysis/simu6a_cr_recsym/data/simu6a_gt.json") as f:
            _gt = json.load(f)
        fw, bw, err = _simu6a_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [fw, bw]}


    ## SIMU6b ##
    elif simu_name == "6b":

        # Run model
        param_dict.update(learn_while_retrieving=True, nitems_in_accumulator=96)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="CR-CR", disable_tqdm=True)
        df_simu["test_pos"] = df_test["test_pos"]
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno1", "test_itemno2", "test_pos"])

        # Score the model's behavioral stats as compared with the true data
        with open("../../Analysis/simu6b_cr_sym/data/simu6b_gt.json") as f:
            _gt = json.load(f)
        inde_stats_mean, reve_stats_mean, err = _simu6b_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [inde_stats_mean, reve_stats_mean]}


    ## SIMU7 ##
    elif simu_name == "7":

        # Run model
        param_dict.update(nitems_in_accumulator=96)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Get error
        with open("../../Analysis/simu7_cr_pliili/data/simu7_gt.json") as f:
            _gt = json.load(f)
        p_correct_mean, p_PLI_mean, p_ILI_mean, lag_PLI_mean, lag_ILI_mean, err = _simu7_stats(df_simu, df_study, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [p_correct_mean, p_PLI_mean, p_ILI_mean, lag_PLI_mean, lag_ILI_mean]}


    ## SIMU8 ##
    elif simu_name == "8":

        # Shared face distance matrix and ground truth
        face_distance = np.load("../../Analysis/simu8_cr_sim/data/simu8_distance.npy")
        thresh = 3.0
        with open("../../Analysis/simu8_cr_sim/data/simu8_gt.json") as f:
            _gt = json.load(f)

        # g1: run cued recall
        param_dict.update(nitems_in_accumulator=16, ban_recall=np.arange(1, 17))
        df_study_g1 = df_study.query("group == 1").copy()
        df_test_g1 = df_test.query("group == 1").copy()
        df_test_g1 = df_test_g1.rename(columns={"test_itemno1": "test_itemno", "test_item1": "test_item"})
        df_test_g1.drop(columns=["test_itemno2", "test_item2"], inplace=True)
        try:
            with _time_limit(_EVAL_TIMEOUT):
                df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study_g1, df_test_g1, sem_mat, disable_tqdm=True)
        except _EvalTimeout:
            return _TIMEOUT_PENALTY, {"err": _TIMEOUT_PENALTY, "params": param_vec, "timeout": "g1"}
        df_simu = df_simu.merge(df_test_g1, on=["session", "list", "test_itemno"])
        neighbor_mean, ILI_mean, correct_rate, wls_neighbor, wls_ILI = _simu8_g1_stats(df_simu, df_study_g1, face_distance, thresh, _gt)

        # g2: run recognition + final cued recall
        param_dict.update(nitems_in_accumulator=32, ban_recall=np.arange(1, 17), learn_while_retrieving=True)
        df_study_g2 = df_study.query("group == 2").copy()
        df_test_g2 = df_test.query("group == 2").copy()
        try:
            with _time_limit(_EVAL_TIMEOUT):
                df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study_g2, df_test_g2, sem_mat, mode="Recog-CR", design="S1G3", disable_tqdm=True)
        except _EvalTimeout:
            return _TIMEOUT_PENALTY, {"err": _TIMEOUT_PENALTY, "params": param_vec, "timeout": "g2"}
        df_simu["test"] = df_test_g2["test"].values
        df_simu = df_simu.merge(df_test_g2, on=["session", "list", "test", "test_itemno1", "test_itemno2"])
        df_recog = df_simu.query("test == 1").copy()
        df_cr = df_simu.query("test == 2").copy()
        hr_mean, far_mean, yesdist_mean, crdist_mean, wls_hr, wls_far, wls_yesdist, wls_crdist = _simu8_g2_stats(df_recog, df_cr, df_study_g2, face_distance, thresh, _gt)

        # Combine g1 + g2 into one error
        err = wls_neighbor + wls_ILI + wls_hr * 10 + wls_far * 10 + wls_yesdist + wls_crdist
        if correct_rate < 0.6:
            err += 5
        if np.any(np.diff(hr_mean) > 0):
            err += 10
        if np.any(np.diff(far_mean) < 0):
            err += 10
        cmr_stats = {"err": err, "params": param_vec, "stats": [neighbor_mean, ILI_mean, hr_mean, far_mean, yesdist_mean, crdist_mean]}


    ## SIMUS1 ##
    elif simu_name == "S1":

        # Separate 3 groups of simulation
        dfs = []
        for i in [1, 2, 3]:

            df_study_gp = df_study.query(f"group == {i}").copy()
            df_test_gp = df_test.query(f"group == {i}").copy()
            mode = "Recog-CR"
            design = "S1G3" if i == 3 else None
            nitems = 4 * 48

            # Run model
            param_dict.update(nitems_in_accumulator=nitems, learn_while_retrieving=True, rec_time_limit=10000)
            df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study_gp, df_test_gp, sem_mat, mode=mode, design=design, disable_tqdm=True)
            df_simu["test"] = df_test_gp["test"]
            df_simu = df_simu.merge(df_test_gp, on=["session", "test", "test_itemno1", "test_itemno2"])
            dfs.append(df_simu)

        # Score the model's behavioral stats as compared with the true data
        with open("../../Analysis/simuS1_recog_cr/data/simuS1_gt.json") as f:
            _gt = json.load(f)
        stats, err = _simuS1_stats(dfs, _gt)

        # Apply some constraints that pair FAR should not be 0
        if stats[1, 2] == 0:
            err += 1
        cmr_stats = {"err": err, "params": param_vec, "stats": stats}

        if return_df:
            return err, cmr_stats, df_simu


    ## SIMUS2 ##
    elif simu_name == "S2":

        # Run model
        param_dict.update(learn_while_retrieving=True)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="Recog-Recog", disable_tqdm=True)
        df_simu["test"] = df_test["test"]
        df_simu = df_simu.merge(df_test, on=["session", "list", "test", "test_itemno1", "test_itemno2"])

        # Score the model's behavioral stats as compared with the true data
        with open("../../Analysis/simuS2_recog_recog/data/simuS2_gt.json") as f:
            _gt = json.load(f)
        stats_mean, err = _simuS2_stats(df_simu, _gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": stats_mean}

    else:
        raise ValueError(f"Unknown simu_name: {simu_name!r}")

    return err, cmr_stats
