import numpy as np
import pandas as pd
import scipy.stats as ss


def setup_notebook(fallback_font_path=None):
    """Configure matplotlib, seaborn, and pandas for paper-quality output.

    On systems without Times New Roman (e.g. HPC clusters), a Times.ttc
    bundled alongside this file is used automatically. Override by passing
    an explicit path via fallback_font_path.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # plt font
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available_fonts:
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        if fallback_font_path is None:
            fallback_font_path = os.path.join(os.path.dirname(__file__), "Times.ttc")
        if os.path.exists(fallback_font_path):
            font_manager.fontManager.addfont(fallback_font_path)
            plt.rc("font", family="serif", serif=["Times"])
            plt.rcParams["mathtext.fontset"] = "custom"
            plt.rcParams["mathtext.rm"] = "Times"

    # plt font size
    plt.rcParams["font.size"] = 16
    
    # Other settings
    pd.set_option("display.max_columns", None)


def compute_roc_core(csim, base_thresh, session, level, thresh_arr, n_levels):
    """Sweep a threshold multiplier over csim/base_thresh to build an ROC curve.

    Canonical numpy implementation shared by the plotting wrapper below and by
    the fitting objective function. ``level`` is an integer-code array in
    [0, n_levels). For each threshold t, classifies trials as 'above threshold'
    (csim > t * base_thresh), averages per (session, level), then collapses
    across sessions. Returns a (len(thresh_arr), n_levels) array of fractions.
    """
    csim = np.asarray(csim, dtype=float)
    base_thresh = np.asarray(base_thresh, dtype=float)
    level = np.asarray(level, dtype=np.int64)

    # (session, level) cells; collapse cells to per-level means
    cells, cell_inv = np.unique(np.column_stack([np.asarray(session), level]), axis=0, return_inverse=True)
    cell_level = cells[:, 1]
    cell_cnt = np.bincount(cell_inv)
    level_cnt = np.bincount(cell_level, minlength=n_levels)

    roc = np.empty((len(thresh_arr), n_levels))
    for ti, t in enumerate(thresh_arr):
        above = (csim > t * base_thresh).astype(float)
        cell_mean = np.bincount(cell_inv, weights=above, minlength=len(cells)) / cell_cnt
        roc[ti] = np.bincount(cell_level, weights=cell_mean, minlength=n_levels) / level_cnt
    return roc


def compute_roc(df, thresh_arr=None, csim_col="csim", thresh_col="thresh", level_col="level", session_col="session"):
    """Sweep a threshold multiplier over csim/thresh to build an ROC curve.

    Thin DataFrame wrapper around compute_roc_core. Returns a DataFrame with one
    row per threshold value and one column per level (e.g. new_a, new_r, old_a,
    old_r, sorted), containing the fraction of trials classified as 'above
    threshold', averaged over sessions.
    """
    if thresh_arr is None:
        thresh_arr = np.arange(0, 2, 0.001)

    # Map level labels to integer codes in sorted order (matches pandas groupby)
    levels, level_code = np.unique(df[level_col].to_numpy(), return_inverse=True)
    roc = compute_roc_core(df[csim_col].to_numpy(), df[thresh_col].to_numpy(), df[session_col].to_numpy(), level_code, thresh_arr, len(levels))
    return pd.DataFrame(roc, columns=levels)


def compute_zroc(df_roc):
    """Apply the normal quantile transform to each column of an ROC DataFrame.

    Returns a new DataFrame with columns prefixed by 'z_'.
    """
    df_z = pd.DataFrame()
    for col in df_roc.columns:
        df_z[f"z_{col}"] = ss.norm.ppf(df_roc[col].to_numpy())
    return df_z
