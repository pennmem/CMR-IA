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


def compute_roc(df, thresh_arr=None, csim_col="csim", thresh_col="thresh", level_col="level", session_col="session"):
    """Sweep a threshold multiplier over csim/thresh to build an ROC curve.

    Returns a DataFrame with one row per threshold value and one column per
    level (e.g. new_a, new_r, old_a, old_r), containing the fraction of
    trials classified as 'above threshold', averaged over sessions.
    """
    if thresh_arr is None:
        thresh_arr = np.arange(0, 2, 0.001)

    df_thin = df[[csim_col, thresh_col, level_col, session_col]].copy()
    csim_vec = df_thin[csim_col].to_numpy()
    base_thresh_vec = df_thin[thresh_col].to_numpy()

    rows = []
    for t in thresh_arr:
        df_thin["above"] = csim_vec > t * base_thresh_vec
        df_lv = df_thin.groupby([session_col, level_col])["above"].mean().groupby(level_col).mean()
        rows.append(df_lv)

    return pd.concat(rows, axis=1, ignore_index=True).T.reset_index(drop=True)


def compute_zroc(df_roc):
    """Apply the normal quantile transform to each column of an ROC DataFrame.

    Returns a new DataFrame with columns prefixed by 'z_'.
    """
    df_z = pd.DataFrame()
    for col in df_roc.columns:
        df_z[f"z_{col}"] = ss.norm.ppf(df_roc[col].to_numpy())
    return df_z


def interpolate_roc(far_arr, hr_arr, far_gt):
    """Linearly interpolate HR values at specified FAR ground-truth points.

    Both far_arr and hr_arr are sorted ascending before interpolation,
    which is correct for ROC curves where both axes are monotone.
    """
    far_sorted = np.sort(far_arr)
    hr_sorted = np.sort(hr_arr)
    hr_interp = []
    for x in far_gt:
        idx = np.searchsorted(far_sorted, x)
        if idx == 0:
            hr_interp.append(hr_sorted[0])
        elif idx >= len(far_sorted):
            hr_interp.append(hr_sorted[-1])
            print(f"Warning: far_gt value {x:.4f} is out of range")
        else:
            t = (x - far_sorted[idx - 1]) / (far_sorted[idx] - far_sorted[idx - 1])
            hr_interp.append(hr_sorted[idx - 1] + t * (hr_sorted[idx] - hr_sorted[idx - 1]))
    return np.array(hr_interp)
