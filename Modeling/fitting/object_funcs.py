import numpy as np
import scipy.stats as ss
import CMR_IA as cmr
import time
import pandas as pd
import math
import pickle
import scipy as sp
from scipy.stats import norm
from optimization_utils import *
from sklearn.cluster import KMeans


def obj_func_1(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="1")
    param_dict.update(use_new_context=True, use_flexible_thresh=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Continuous", disable_tqdm=True)
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

    # calculate perceptron loss
    # csim = df_simu["csim"].to_numpy()
    # 1
    # score = csim - param_dict["c_thresh_itm"]
    # true_resp = df_simu["yes"].to_numpy() * 2 - 1
    # err = np.nansum(np.maximum(0, -true_resp * score))
    # 2
    # true_resp = df_simu["yes"].to_numpy()
    # err = np.nansum(np.abs(true_resp - csim))
    # 3
    # csim_threshs = np.linspace(0, 1, 8)
    # conf = df_simu["confidence"].to_numpy() - 1
    # target_csim = np.array([csim_threshs[int(i)] if not np.isnan(i) else i for i in conf])
    # err = np.nansum(np.abs(target_csim - csim))
    # 4
    # kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(csim.reshape(-1, 1))
    # labels = kmeans.labels_
    # cluster_means = {i: csim[labels == i].mean() for i in range(8)}
    # sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    # label_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    # sorted_labels = np.vectorize(label_mapping.get)(labels)
    # conf = df_simu["confidence"].to_numpy() - 1
    # err = np.nansum(np.abs(conf - sorted_labels))
    # 5
    # yes = df_simu["yes"].to_numpy()
    # nan_idx = np.isnan(conf)
    # err1 = np.nansum(conf[~nan_idx] != sorted_labels[~nan_idx])
    # err2 = np.nansum(yes[~nan_idx] != (sorted_labels[~nan_idx] > 3))
    # err = err1 + err2
    # 6
    # conf_freq_group = np.histogram(conf, bins=np.arange(9))[0] / 10000
    # conf_freq = conf_freq_group[conf[~nan_idx].astype(int)]
    # err = np.nansum((conf[~nan_idx] != sorted_labels[~nan_idx]) / conf_freq)
    # 7
    # s_resp = df_simu["s_resp"].to_numpy()
    # position = df_simu["position"].to_numpy()
    # is_position_later = position >= 20
    # err = np.nansum(np.abs(s_resp[is_position_later] - yes[is_position_later]))
    
    # 8 fit HR and FAR
    # calculate the rolling category length
    rolling_window = 9
    category_label_dummies = df_simu["category_label"].str.get_dummies()
    category_label_dummies.columns = ["cl_" + col for col in category_label_dummies.columns]
    category_label_dummies_events = pd.concat([df_simu, category_label_dummies], axis=1)  # record the occurrence of every cat label
    cl_rolling_sum = category_label_dummies_events.groupby("session").rolling(rolling_window, min_periods=1, on="position")[category_label_dummies.columns].sum().reset_index()
    df_rollcat = df_simu.merge(cl_rolling_sum, on=["session", "position"])
    df_simu["roll_cat_label_length"] = df_rollcat.apply(lambda x: x["cl_" + x["category_label"]], axis=1)  # how many cat within 10 window
    df_simu["roll_cat_label_length"] = df_simu["roll_cat_label_length"] - 1  # how many cat in previous 9 window. not include self
    df_simu["roll_cat_len_level"] = pd.cut(x=df_simu.roll_cat_label_length, bins=[0, 2, np.inf], right=False, include_lowest=True, labels=["0-1", ">=2"]).astype("str")

    # add log and log lag bin
    df_simu["log_lag"] = np.log(df_simu["lag"])
    df_simu["log_lag_bin"] = pd.cut(df_simu["log_lag"], np.arange(df_simu["log_lag"].max() + 1), labels=False, right=False)
    df_simu["log_lag_bin"] = df_simu.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
    df_simu["log_lag_bin"] = df_simu.apply(lambda x: 5 if x["log_lag_bin"] > 5 else x["log_lag_bin"], axis=1)
    
    # construct local FAR
    old_vec = df_simu.old.to_numpy()
    log_lag_bin_vec = df_simu.log_lag_bin.to_numpy()
    position_vec = df_simu.position.to_numpy()
    max_position = np.max(position_vec)
    log_lag_bin_newpre_lst = []
    log_lag_bin_newpost_lst = []
    for i in range(len(df_simu)):
        if position_vec[i] > 0:
            if old_vec[i] == False and old_vec[i - 1] == True:
                log_lag_bin_newpre_lst.append(log_lag_bin_vec[i - 1])
            else:
                log_lag_bin_newpre_lst.append("N")
        else:
            log_lag_bin_newpre_lst.append("N")

        if position_vec[i] < max_position:
            if old_vec[i] == False and old_vec[i + 1] == True:
                log_lag_bin_newpost_lst.append(log_lag_bin_vec[i + 1])
            else:
                log_lag_bin_newpost_lst.append("N")
        else:
            log_lag_bin_newpost_lst.append("N")
    df_simu["log_lag_bin_newpre"] = log_lag_bin_newpre_lst
    df_simu["log_lag_bin_newpost"] = log_lag_bin_newpost_lst
    
    # distribute items into bins
    log_lag_bins = [0, 2, 3, 4, 5]
    for bin in log_lag_bins:
        col_name = "log_lag_bin_" + str(bin)
        df_simu[col_name] = (df_simu.log_lag_bin == bin) | (df_simu.log_lag_bin_newpre == bin) | (df_simu.log_lag_bin_newpost == bin)
        
    # clean the first 20
    df_simu = df_simu.query("position >= 20").copy()
    
    # get yes rate
    df_lst = []
    for bin in log_lag_bins:
        col_name = "log_lag_bin_" + str(bin)
        df_tmp = df_simu.query(col_name + " == True").groupby(["session", "old", "roll_cat_len_level"])["s_resp"].agg(["mean", "sum", "count"]).reset_index()
        df_tmp["log_lag_bin"] = bin
        df_lst.append(df_tmp)
    df_rollcat_laggp = pd.concat(df_lst)
    df_rollcat_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
    
    # pivot for hr and far
    df_rollcat_laggp["log_lag_disp"] = np.ceil(np.e**df_rollcat_laggp.log_lag_bin)
    df_rollcat_laggp["old"] = df_rollcat_laggp["old"].astype("str")
    df_dprime = pd.pivot_table(df_rollcat_laggp, values=["yes_rate"], index=["session", "roll_cat_len_level", "log_lag_disp"], columns="old").reset_index()
    df_dprime.columns = [" ".join(col).strip() for col in df_dprime.columns.values]
    df_dprime = df_dprime.rename(columns={"yes_rate False": "far", "yes_rate True": "hr"})
    
    # calculate hr and far
    df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].mean().reset_index()
    hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["hr"].to_numpy()
    hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["hr"].to_numpy()
    far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["far"].to_numpy()
    far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["far"].to_numpy()
    
    # # calculate hr
    # df_rollcat_laggp = df_simu.query("old == True").groupby(["session", "roll_cat_len_level", "log_lag_bin"])["s_resp"].agg(["mean", "sum", "count"]).reset_index()
    # df_rollcat_laggp["yes_rate_adj"] = (df_rollcat_laggp["sum"] + 0.5) / (df_rollcat_laggp["count"] + 1)
    # df_hr = df_rollcat_laggp.groupby(["roll_cat_len_level", "log_lag_bin"]).yes_rate_adj.mean().to_frame(name="hr_adj").reset_index()
    # hr_lowsim = df_hr.query("roll_cat_len_level == '0-1'")["hr_adj"].to_numpy()
    # hr_highsim = df_hr.query("roll_cat_len_level == '>=2'")["hr_adj"].to_numpy()
    
    # # calculate far
    # df_far = df_simu.query("old == False").groupby(["session", "roll_cat_len_level"])["s_resp"].mean().to_frame(name="far").reset_index()
    # far_lowsim_overall = df_far.groupby("roll_cat_len_level")["far"].mean()["0-1"]
    # far_highsim_overall = df_far.groupby("roll_cat_len_level")["far"].mean()[">=2"]
    
    # calculate error
    with open("../../Analysis/simu1_recog_recsim/simu1_data/simu1_gt.pkl", "rb") as f:
        hr_lowsim_gt = pickle.load(f)
        hr_lowsim_std_gt = pickle.load(f)
        hr_highsim_gt = pickle.load(f)
        hr_highsim_std_gt = pickle.load(f)
        far_lowsim_gt = pickle.load(f)
        far_lowsim_std_gt = pickle.load(f)
        far_highsim_gt = pickle.load(f)
        far_highsim_std_gt = pickle.load(f)
        far_lowsim_overall_gt = pickle.load(f)
        far_lowsim_overall_std_gt = pickle.load(f)
        far_highsim_overall_gt = pickle.load(f)
        far_highsim_overall_std_gt = pickle.load(f)
    # err = get_wmse(hr_lowsim_gt, hr_lowsim, hr_lowsim_std_gt) / len(hr_lowsim_gt) + get_wmse(hr_highsim_gt, hr_highsim, hr_highsim_std_gt) / len(hr_highsim_gt) \
    #     + get_wmse(far_lowsim_overall_gt, far_lowsim_overall, far_lowsim_overall_std_gt) + get_wmse(far_highsim_overall_gt, far_highsim_overall, far_highsim_overall_std_gt)
    err = get_wmse(hr_lowsim_gt, hr_lowsim, hr_lowsim_std_gt) + get_wmse(hr_highsim_gt, hr_highsim, hr_highsim_std_gt) + get_wmse(far_lowsim_gt, far_lowsim, far_lowsim_std_gt) + get_wmse(far_highsim_gt, far_highsim, far_highsim_std_gt)
    
    # 9 calculate cross entropy by prob and yes
    # prob = df_simu["prob"].to_numpy()
    # yes = df_simu["yes"].to_numpy()
    # err = -np.nansum(yes * np.log(prob) + (1 - yes) * np.log(1 - prob))

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = []

    return err, cmr_stats


def obj_func_1x(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="1x")
    param_dict.update(use_new_context=True, use_flexible_thresh=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Continuous", disable_tqdm=True)
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

    # clean the first 20
    df_simu = df_simu.query("position >= 20").copy()
    
    # 9 calculate cross entropy by prob and yes
    prob = df_simu["prob"].to_numpy()
    yes = df_simu["yes"].to_numpy()
    err = -np.nansum(yes * np.log(prob) + (1 - yes) * np.log(1 - prob))

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = []

    return err, cmr_stats


def obj_func_2(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="2")
    param_dict.update(use_new_context=True, use_flexible_thresh=True, c_thresh_itm=1)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])

    # calculate loss
    # s_resp = df_simu["s_resp"].values
    # yes = df_simu["yes"].values
    # err = np.nansum(np.abs(s_resp - yes))

    # add lag condition
    def conditions(s):
        if s.old_lag == -999:
            return np.nan
        elif np.absolute(s.old_lag) == 1:
            return "a"
        elif np.absolute(s.old_lag) > 10:
            return "r"
        else:
            return np.nan

    df_simu["lag_cat"] = df_simu.apply(conditions, axis=1)
    
    # 1
    # # calculate HR and FAR
    # far = df_simu.query("old==False").s_resp.mean()
    # df_overall = df_simu.query("old==True").groupby(["subject", "lag_cat"]).s_resp.mean().reset_index()
    # hr_a = df_overall.groupby(["lag_cat"]).s_resp.mean()["a"]
    # hr_r = df_overall.groupby(["lag_cat"]).s_resp.mean()["r"]

    # # calcualte err
    # hr_a_gt = 0.724046
    # hr_r_gt = 0.693084
    # far_gt = 0.268861
    # err = (np.abs(hr_a - hr_a_gt) + np.abs(hr_r - hr_r_gt)) / 2 + np.abs(far - far_gt)
    
    # 2
    # construct local FAR
    recog_pos = df_simu.recog_pos.values
    old = df_simu.old.values
    lag_cat = df_simu.lag_cat.values
    lag_cat_with_new = []
    for i in range(len(df_simu)):
        if recog_pos[i] > 1:
            if not old[i] and old[i - 1]:
                lag_cat_with_new.append(lag_cat[i - 1])
            else:
                lag_cat_with_new.append(lag_cat[i])
        else:
            lag_cat_with_new.append(lag_cat[i])
    df_simu["lag_cat"] = lag_cat_with_new
    
    # get conditions
    df_t = df_simu.loc[pd.notna(df_simu.lag_cat)].copy()
    create_level = {0: "new_r", 1: "new_a", 2: "old_r", 3: "old_a"}
    df_t["level"] = df_t.apply(lambda x: create_level[x["old"] * 2 + (x["lag_cat"] == "a")], axis=1)
    
    # get roc
    thresh_arr = np.arange(0, 2, 0.001)
    df_thin = df_t.loc[:, ["csim", "thresh", "level", "session"]]
    csim_vec = df_thin.csim.to_numpy()
    base_thresh_vec = df_thin.thresh.to_numpy()
    df_roc_lst = []
    for t in thresh_arr:
        df_thin["above"] = csim_vec > t * base_thresh_vec
        df_sess_lv = df_thin.groupby(["session", "level"]).above.mean().to_frame(name="above")
        df_lv = df_sess_lv.groupby("level").above.mean()
        df_roc_lst.append(df_lv)
    df_roc = pd.concat(df_roc_lst, axis=1, ignore_index=True)
    df_roc = df_roc.transpose()
    
    # calculate err
    with open("../../Analysis/simu2_recog_conti/simu2_data/simu2_gt.pkl", "rb") as f:
        far_a_gt = pickle.load(f)
        hr_a_gt = pickle.load(f)
        far_r_gt = pickle.load(f)
        hr_r_gt = pickle.load(f)
    far_a = np.sort(df_roc["new_a"].values)
    hr_a = np.sort(df_roc["old_a"].values)
    far_r = np.sort(df_roc["new_r"].values)
    hr_r = np.sort(df_roc["old_r"].values)
    hr_a_interp = []
    for x in far_a_gt:
        idx = np.searchsorted(far_a, x)
        if idx < len(far_a):
            tmp_interp = hr_a[idx - 1] + (x - far_a[idx - 1]) * (hr_a[idx] - hr_a[idx - 1]) / (far_a[idx] - far_a[idx - 1])
        else:
            tmp_interp = hr_a[idx - 1]
            print("Warning: far_a_gt out of range")
        hr_a_interp.append(tmp_interp)
    hr_a_interp = np.array(hr_a_interp)
    hr_r_interp = []
    for x in far_r_gt:
        idx = np.searchsorted(far_r, x)
        if idx < len(far_r):
            tmp_interp = hr_r[idx - 1] + (x - far_r[idx - 1]) * (hr_r[idx] - hr_r[idx - 1]) / (far_r[idx] - far_r[idx - 1])
        else:
            tmp_interp = hr_r[idx - 1]
            print("Warning: far_r_gt out of range")
        hr_r_interp.append(tmp_interp)
    hr_r_interp = np.array(hr_r_interp)
    err = np.power(hr_a_interp - hr_a_gt, 2).sum() + np.power(hr_r_interp - hr_r_gt, 2).sum()
    
    # ensure a is above r in a range
    above_range = np.arange(0.08, 0.61, 0.01)
    hr_a_interp_range = []
    for x in above_range:
        idx = np.searchsorted(far_a, x)
        if idx < len(far_a):
            tmp_interp = hr_a[idx - 1] + (x - far_a[idx - 1]) * (hr_a[idx] - hr_a[idx - 1]) / (far_a[idx] - far_a[idx - 1])
        else:
            tmp_interp = hr_a[idx - 1]
            print("Warning: above_range out of range")
        hr_a_interp_range.append(tmp_interp)
    hr_a_interp_range = np.array(hr_a_interp_range)
    hr_r_interp_range = []
    for x in above_range:
        idx = np.searchsorted(far_r, x)
        if idx < len(far_r):
            tmp_interp = hr_r[idx - 1] + (x - far_r[idx - 1]) * (hr_r[idx] - hr_r[idx - 1]) / (far_r[idx] - far_r[idx - 1])
        else:
            tmp_interp = hr_r[idx - 1]
            print("Warning: above_range out of range")
        hr_r_interp_range.append(tmp_interp)
    hr_r_interp_range = np.array(hr_r_interp_range)
    if not np.all((hr_a_interp_range > hr_r_interp_range)):
        err += 0.5

    if not np.all((hr_a_interp > hr_r_interp)[1:]):
        err += 0.5

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr_a_interp, hr_r_interp]

    return err, cmr_stats


def obj_func_2b(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="2b")
    param_dict.update(use_new_context=True, use_flexible_thresh=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, mode="Osth", disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
    df_simu["old"] = df_simu.apply(lambda x: 1 if x["type"] == "intact" else 0, axis=1)
    df_simu["correct"] = df_simu["s_resp"] == df_simu["old"]

    # get yes rate
    df_hrfar = df_simu.groupby(["session", "type"]).correct.mean().to_frame(name="yes_rate").reset_index()
    df_hrfar = df_hrfar.pivot(index="session", columns="type", values="yes_rate").reset_index()
    df_hrfar["hr"] = df_hrfar["intact"]
    df_hrfar["far"] = 1 - df_hrfar["rearranged"]
    df_hrfar_plot = pd.melt(df_hrfar, id_vars=["session"], value_vars=["hr", "far"], var_name="type", value_name="yes_rate")
    
    # get far with lag
    df_lure = df_simu.query("type == 'rearranged'").copy()
    df_farlag = df_lure.groupby(["session", "lag"]).correct.mean().to_frame(name="yes_rate").reset_index()
    df_farlag["far"] = 1 - df_farlag["yes_rate"]

    # calculate err
    with open("../../Analysis/simu2b_recog_assoc_conti/simu2b_data/simu2b_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
        far_gt = pickle.load(f)
        far_std_gt = pickle.load(f)
    hr = df_hrfar_plot.query("type == 'hr'").yes_rate.mean()
    far = df_farlag.groupby("lag").far.mean().to_numpy()
    err = 5 * get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_3(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="3")
    param_dict.update(use_new_context=True, use_flexible_thresh=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Hockley", disable_tqdm=True)
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

    # session-wise, calculate the yes_rate for each condition
    df_sess_laggp = df_simu.groupby(["session", "type", "lag"]).s_resp.agg(["count", "sum", "mean"]).reset_index()
    df_sess_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
    df_sess_laggp["yes_rate_adj"] = (df_sess_laggp["sum"] + 0.5) / (df_sess_laggp["count"] + 1)
    df_sess_laggp["z_yes_rate"] = sp.stats.norm.ppf(df_sess_laggp["yes_rate_adj"])

    # collapse across session to get hit rate and false alarm rate
    df_laggp = df_sess_laggp.groupby(["type", "lag"]).yes_rate.mean().to_frame(name="yes_rate").reset_index()
    df_laggp["no_rate"] = 1 - df_laggp["yes_rate"]

    # get the vectors
    I_hr = df_laggp.loc[df_laggp.type == "single_old", "yes_rate"].to_numpy()
    I_far = np.mean(df_laggp.loc[df_laggp.type == "single_new", "yes_rate"].astype(float))
    A_hr = df_laggp.loc[df_laggp.type == "pair_old", "yes_rate"].to_numpy()
    A_far = df_laggp.loc[df_laggp.type == "pair_new", "yes_rate"].to_numpy()

    # calculate err
    with open("../../Analysis/simu3_recog_forget/simu3_data/simu3_gt.pkl", "rb") as f:
        I_hr_gt = pickle.load(f)
        I_far_gt = pickle.load(f)
        A_hr_gt = pickle.load(f)
        A_cr_gt = pickle.load(f)
    A_far_gt = 1 - A_cr_gt
    # err = np.mean(np.abs(I_hr - I_hr_gt)) + np.mean(np.abs(A_hr - A_hr_gt)) + np.abs(I_far - I_far_gt) * 5 + np.mean(np.abs(A_far - A_far_gt))
    err = np.sum((I_hr - I_hr_gt) ** 2) + np.sum((A_hr - A_hr_gt) ** 2) + (I_far - I_far_gt) ** 2 * 5 + np.sum((A_far - A_far_gt) ** 2)

    # apply some constraints
    if np.any(np.diff(I_hr) > 0):
        err += 1
    if np.any(np.diff(A_hr) > 0):
        err += 1
    if np.any(np.diff(A_far) > 0):
        err += 1
    if np.any(I_hr < A_hr):
        err += 1

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [I_hr, I_far, A_hr, A_far]

    return err, cmr_stats


def obj_func_4(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, use_flexible_thresh=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "list", "itemno"])

    # session-wise, get yes rate for each condition
    df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # collapse across session
    df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

    # Get behavioral stats and compare with ground truth
    with open("../../Analysis/simu4_recog_wfe/simu4_data/simu4_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
        far_gt = pickle.load(f)
        far_std_gt = pickle.load(f)
    hr = df_q.query("old == True")["yes_rate"].to_numpy()
    far = df_q.query("old == False")["yes_rate"].to_numpy()
    err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_4ctrl(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4ctrl")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, use_flexible_thresh=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "itemno"])

    # session-wise, get yes rate for each condition
    df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # collapse across session
    df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

    # Get behavioral stats and compare with ground truth
    with open("../../Analysis/simu4_recog_wfe/simu4_data/simu4_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
        far_gt = pickle.load(f)
        far_std_gt = pickle.load(f)
    hr = df_q.query("old == True")["yes_rate"].to_numpy()
    far = df_q.query("old == False")["yes_rate"].to_numpy()
    err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)
    
    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_4shift(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4shift")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, use_flexible_thresh=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "itemno"])

    # session-wise, get yes rate for each condition
    df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # collapse across session
    df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

    # Get behavioral stats and compare with ground truth
    with open("../../Analysis/simu4_recog_wfe/simu4_data/simu4_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
        far_gt = pickle.load(f)
        far_std_gt = pickle.load(f)
    hr = df_q.query("old == True")["yes_rate"].to_numpy()
    far = df_q.query("old == False")["yes_rate"].to_numpy()
    err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_4attn(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4attn")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, use_flexible_thresh=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "itemno"])

    # session-wise, get yes rate for each condition
    df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # collapse across session
    df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

    # Get behavioral stats and compare with ground truth
    with open("../../Analysis/simu4_recog_wfe/simu4_data/simu4_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
        far_gt = pickle.load(f)
        far_std_gt = pickle.load(f)
    hr = df_q.query("old == True")["yes_rate"].to_numpy()
    far = df_q.query("old == False")["yes_rate"].to_numpy()
    err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_5(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="5")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, nitems_in_accumulator=48)
    df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # session-wise, calculate correct rate for each lag
    df_sess_lag = df_simu.groupby(["session", "lag"]).correct.mean().to_frame(name="correct_rate").reset_index()
    
    # collapse across sessions
    hr = df_sess_lag.groupby("lag").correct_rate.mean().to_numpy()
    
    # Get error
    with open("../../Analysis/simu5_cr_rec/simu5_data/simu5_gt.pkl", "rb") as f:
        hr_gt = pickle.load(f)
        hr_std_gt = pickle.load(f)
    err = get_wmse(hr_gt, hr, hr_std_gt)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = hr

    return err, cmr_stats

def obj_func_6a(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="6a")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, nitems_in_accumulator=48)
    df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
    
    # clean first 2 list
    df_simu = df_simu.query("list > 1")

    # session-wise, calculate correct rate for each condition
    df_sess_lag = df_simu.groupby(["session", "lag", "order"]).correct.mean().to_frame(name="correct_rate").reset_index()
    
    # collapse across sessions
    df_lag = df_sess_lag.groupby(["lag", "order"]).correct_rate.mean().to_frame(name="correct_rate").reset_index()
    fw = df_lag.query("order == 1").correct_rate.values
    bw = df_lag.query("order == 2").correct_rate.values
    
    # Get error
    with open("../../Analysis/simu6a_cr_recsym/simu6a_data/simu6a_gt.pkl", "rb") as f:
        fw_gt = pickle.load(f)
        bw_gt = pickle.load(f)
    err = np.power(fw - fw_gt, 2).sum() + np.power(bw - bw_gt, 2).sum()

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [fw, bw]

    return err, cmr_stats


def obj_func_6b(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="6b")

    # Run model with the parameters given in param_vec
    param_dict.update(learn_while_retrieving=True, nitems_in_accumulator=96, use_new_context=True)
    # param_dict.update(learn_while_retrieving=False, nitems_in_accumulator=96, use_new_context=True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="CR-CR", disable_tqdm=True)
    df_simu["test_pos"] = df_test["test_pos"]
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno1", "test_itemno2", "test_pos"])

    # Get correctness
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # Get conditions
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
    pairidx2cond = df_cond.loc[:, ["pair_idx", "cond"]].set_index("pair_idx").to_dict()["cond"]
    pairidx2cong = df_cond.loc[:, ["pair_idx", "cong"]].set_index("pair_idx").to_dict()["cong"]
    df_simu["cond"] = df_simu.apply(lambda x: pairidx2cond[x["pair_idx"]], axis=1)
    df_simu["cong"] = df_simu.apply(lambda x: pairidx2cong[x["pair_idx"]], axis=1)

    # Get behavioral stats
    subjects = np.unique(df_simu.session)
    inde_stats = []
    reve_stats = []
    for subj in subjects:
        df_subj_inde = df_simu.query(f"session == {subj} and cong == 'Identical'").copy()
        inde_stats.append(list(anal_perform_6b(df_subj_inde)))

        df_subj_reve = df_simu.query(f"session == {subj} and cong == 'Reversed'").copy()
        reve_stats.append(list(anal_perform_6b(df_subj_reve)))

    # Score the model's behavioral stats as compared with the true data
    inde_stats_mean = np.mean(inde_stats, axis=0)
    reve_stats_mean = np.mean(reve_stats, axis=0)
    inde_ground_truth = np.array([0.319, 0.006, 0.012, 0.663, 0.94])
    reve_ground_truth = np.array([0.293, 0.049, 0.122, 0.537, 0.96])
    err = np.sum(np.power(inde_stats_mean - inde_ground_truth, 2)) + np.sum(np.power(reve_stats_mean - reve_ground_truth, 2)) \
        + np.power(inde_stats_mean[-1] - inde_ground_truth[-1], 2) + np.power(reve_stats_mean[-1] - reve_ground_truth[-1], 2)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [inde_stats_mean, reve_stats_mean]

    return err, cmr_stats


def anal_perform_6b(df_simu):

    # get pair
    df_pair = pd.pivot_table(df_simu, index="pair_idx", columns="test", values="correct")
    df_pair.columns = ["test1", "test2"]
    test2_rsp = pd.Categorical(df_pair.test2, categories=[1, 0])
    test1_rsp = pd.Categorical(df_pair.test1, categories=[1, 0])
    df_tab = pd.crosstab(index=test2_rsp, columns=test1_rsp, rownames=["test2"], colnames=["test1"], normalize=False, dropna=False)
    df_tab_norm = pd.crosstab(index=test2_rsp, columns=test1_rsp, rownames=["test2"], colnames=["test1"], normalize="all", dropna=False)
    t1_t2 = df_tab_norm[1][1]  # 1, 2
    t1_f2 = df_tab_norm[1][0]
    f1_t2 = df_tab_norm[0][1]
    f1_f2 = df_tab_norm[0][0]
    # print(df_tab)
    # print(df_tab_norm)
    # print(t1_t2, t1_f2, f1_t2, f1_f2)

    # compute" Q
    def Yule_Q(A, B, C, D):
        return (A * D - B * C) / (A * D + B * C)

    q = Yule_Q(df_tab[1][1] + 0.5, df_tab[0][1] + 0.5, df_tab[1][0] + 0.5, df_tab[0][0] + 0.5)  # add 0.5
    # print("Q: ", q)

    return t1_t2, t1_f2, f1_t2, f1_f2, q


def obj_func_7(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="7")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, nitems_in_accumulator=96)
    df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
    
    # get the study list and study pos of response
    sessions = np.unique(df_simu.session)
    nlist = len(np.unique(df_simu.list))
    resp_study_list, resp_study_pos = [], []
    for sess in sessions:
        pres_words = df_study.loc[df_study.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy()
        pres_words = np.reshape(pres_words, (nlist, -1, 2))
        responses = df_simu.loc[df_simu.session == sess, "s_resp"]
        for r in responses:
            if r == -1 or r == -2:
                r_list, r_pos = None, None
            else:
                r_list = np.where(pres_words == r)[0].item()
                r_pos = np.where(pres_words == r)[1].item()
            resp_study_list.append(r_list)
            resp_study_pos.append(r_pos)
    df_simu["resp_study_list"] = resp_study_list
    df_simu["resp_study_pos"] = resp_study_pos
    df_simu["list_lag"] = df_simu["resp_study_list"] - df_simu["list"]
    df_simu["pos_lag"] = df_simu["resp_study_pos"] - df_simu["study_pos"]
    
    # get intrution type
    def which_intrusion(x):
        x_list_lag = x["list_lag"]
        x_pos_lag = x["pos_lag"]
        if np.isnan(x_list_lag):
            return "NoResp"
        elif x_list_lag == 0 and x_pos_lag == 0:
            return "Correct"
        elif x_list_lag < 0:
            return "PLI"
        elif x_list_lag == 0 and x_pos_lag != 0:
            return "ILI"
        else:
            return np.nan


    df_simu["intrusion_type"] = df_simu.apply(lambda x: which_intrusion(x), axis=1)
    df_simu["intrusion_type"] = pd.Categorical(df_simu["intrusion_type"], categories=["NoResp", "Correct", "PLI", "ILI"])
    
    # clean list 1
    df_simu = df_simu.query("list > 0").copy()
    
    ## get overall prob
    df_cnt = df_simu.groupby(["session", "intrusion_type"]).s_resp.count().to_frame(name="count").reset_index()
    
    # check correct
    df_cnt_correct = df_cnt.query("intrusion_type == 'Correct'").copy()
    df_cnt_correct["total"] = df_simu.groupby("session").test_item.count().tolist()
    df_cnt_correct["p"] = df_cnt_correct["count"] / df_cnt_correct["total"]
    p_correct_mean = np.mean(df_cnt_correct["p"])
    
    # check ILI
    df_cnt_ILI = df_cnt.query("intrusion_type == 'ILI'").copy()
    df_cnt_ILI["total"] = df_simu.groupby("session").test_item.count().tolist()
    df_cnt_ILI["p"] = df_cnt_ILI["count"] / df_cnt_ILI["total"]
    p_ILI_mean = np.mean(df_cnt_ILI["p"])
    
    # check PLI
    df_cnt_PLI = df_cnt.query("intrusion_type == 'PLI'").copy()
    df_cnt_PLI["total"] = df_simu.groupby("session").test_item.count().tolist()
    df_cnt_PLI["p"] = df_cnt_PLI["count"] / df_cnt_PLI["total"]
    p_PLI_mean = np.mean(df_cnt_PLI["p"])

    try:
        ## PLI
        # pick list > 5 and list_lag -5 to -1
        df_PLI = df_simu.query("intrusion_type == 'PLI' and list > 5 and list_lag > -6").copy()
        df_PLI["abs_list_lag"] = df_PLI["list_lag"].abs().astype(int)
        df_PLI["abs_list_lag"] = pd.Categorical(df_PLI["abs_list_lag"], categories=[1, 2, 3, 4, 5], ordered=True)

        # session-wise, count PLI
        df_PLI_sess = df_PLI.groupby(["session"]).test_item.count().to_frame(name="PLI_cnt_sess").reset_index()
        
        # session-wise, count PLI by list_lag
        df_PLI_sess_lag = df_PLI.groupby(["session", "abs_list_lag"]).test_item.count().to_frame(name="PLI_cnt").reset_index()
        
        # calculate PLI probability
        df_PLI_sess_lag = pd.merge(df_PLI_sess_lag, df_PLI_sess, on="session")
        df_PLI_sess_lag["PLI_prob"] = df_PLI_sess_lag["PLI_cnt"] / df_PLI_sess_lag["PLI_cnt_sess"]
        lag_PLI_mean = df_PLI_sess_lag.groupby("abs_list_lag").PLI_prob.mean().values
        
        ## ILI
        df_ILI = df_simu.query("intrusion_type == 'ILI'").copy()
        df_ILI["pos_lag"] = df_ILI["pos_lag"].astype(int)
        df_ILI["pos_lag"] = pd.Categorical(df_ILI["pos_lag"], categories=np.concatenate([np.arange(-11, 0), np.arange(1, 12)]), ordered=True)

        # session-wise, calculate ILI probability for each lag
        def get_ILI_prob(df_tmp):
            # get possible ILI count
            possible_ILI_cnt = {}
            for pair_pos in df_tmp.study_pos:
                l_bound = -pair_pos
                r_bound = 11 - pair_pos
                for i in np.arange(l_bound, r_bound + 1):
                    if i in possible_ILI_cnt:
                        possible_ILI_cnt[i] += 1
                    else:
                        possible_ILI_cnt[i] = 1
            # get ILI count
            df_tmp_lag = df_tmp.groupby("pos_lag")["test_item"].count().to_frame(name="ILI_cnt")
            # merge possible ILI count
            df_tmp_lag["possible_ILI_cnt"] = df_tmp_lag.index.map(possible_ILI_cnt).astype(float)
            df_tmp_lag["ILI_prob"] = df_tmp_lag["ILI_cnt"] / df_tmp_lag["possible_ILI_cnt"]
            return df_tmp_lag


        df_ILI_sess_lag = df_ILI.groupby("session").apply(get_ILI_prob).reset_index()
        df_ILI_sess_lag = df_ILI_sess_lag.query("pos_lag > -6 and pos_lag < 6").copy()
        df_ILI_sess_lag["pos_lag_int"] = df_ILI_sess_lag["pos_lag"].astype(int)  # avoid nan from category vairables
        lag_ILI_mean = df_ILI_sess_lag.groupby("pos_lag_int").ILI_prob.mean().values

    except:  # sometimes there is no PLI or ILI
        lag_PLI_mean = np.full(5, 0)
        lag_ILI_mean = np.full(10, 0)
    
    # Get error
    with open("../../Analysis/simu7_cr_pliili/simu7_data/simu7_gt.pkl", "rb") as f:
        p_correct_mean_gt = pickle.load(f)
        p_correct_se_gt = pickle.load(f)
        p_PLI_mean_gt = pickle.load(f)
        p_PLI_se_gt = pickle.load(f)
        p_ILI_mean_gt = pickle.load(f)
        p_ILI_se_gt = pickle.load(f)
        lag_PLI_mean_gt = pickle.load(f)
        lag_PLI_se_gt = pickle.load(f)
        lag_ILI_mean_gt = pickle.load(f)
        lag_ILI_se_gt = pickle.load(f)
    wls_p_correct = get_wmse(p_correct_mean_gt, p_correct_mean, p_correct_se_gt)
    wls_p_PLI = get_wmse(p_PLI_mean_gt, p_PLI_mean, p_PLI_se_gt)
    wls_p_ILI = get_wmse(p_ILI_mean_gt, p_ILI_mean, p_ILI_se_gt)
    wls_lag_PLI = get_wmse(lag_PLI_mean_gt, lag_PLI_mean, lag_PLI_se_gt) / len(lag_PLI_mean_gt)
    wls_lag_ILI = get_wmse(lag_ILI_mean_gt, lag_ILI_mean, lag_ILI_se_gt) / len(lag_ILI_mean_gt)
    err = wls_p_correct + wls_p_PLI + wls_p_ILI + wls_lag_PLI + wls_lag_ILI
    
    # apply contraints
    # if np.cov(np.arange(5), lag_PLI_mean)[0, 1] >= 0:
    #     err += 5
    # if np.any(np.diff(lag_ILI_mean[:5]) <= 0) or np.any(np.diff(lag_ILI_mean[5:]) >= 0):
    #     err += 5

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [p_correct_mean, p_PLI_mean, p_ILI_mean, lag_PLI_mean, lag_ILI_mean]

    return err, cmr_stats


def obj_func_8(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="8")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True, nitems_in_accumulator=16, ban_recall=np.arange(0, 8))
    df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
    correct_rate = sum(df_simu.correct) / len(df_simu.correct)
    
    # load and get face distance
    face_distance = np.load("../../Analysis/simu8_cr_sim/simu8_data/simu8_distance.npy")
    thresh = 3.0

    ## Neighbor
    # get number of neighbours by distance
    def get_distance(df_tmp):
        faces = np.unique(df_tmp.test_itemno)
        face_dist = {}
        for face in faces:
            this_dist = []
            for other_face in faces:
                if face != other_face:
                    this_dist.append(face_distance[face - 1, other_face - 1])
            this_dist = np.array(this_dist)
            face_dist[face] = this_dist
        y = df_tmp.apply(lambda x: face_dist[x["test_itemno"]], axis=1)
        return y

   
    df_simu["distance"] = df_simu.groupby("session").apply(get_distance).to_frame(name="distance").reset_index()["distance"]
    df_simu["neighbour"] = df_simu.apply(lambda x: sum(x["distance"] < thresh), axis=1)
    distance_lsts = df_simu["distance"].to_list()
    df_simu.drop(columns=["distance"], inplace=True)
    df_simu["neighbour_group"] = df_simu.apply(lambda x: 6 if x["neighbour"] == 7 else x["neighbour"], axis=1)
    
    # get the correct rate by neighbour group
    df_neighbour_group = df_simu.query("neighbour_group > 0").groupby("neighbour_group").correct.mean().reset_index()
    neighbor_mean = df_neighbour_group["correct"].to_numpy()
    
    ## ILI
    try:
        # detect ILI
        def get_ILI(df_tmp):
            resp_names = df_tmp["s_resp"].values
            study_names = df_tmp["correct_ans"].values  # all correct answers are all studied names
            is_studied = np.isin(resp_names, study_names)
            is_incorrect = df_tmp["correct"] == False
            is_ILI = is_studied & is_incorrect
            return is_ILI


        df_simu["ILI"] = df_simu.groupby("session").apply(get_ILI).to_frame(name="ILI").reset_index()["ILI"].to_list()
        df_ILI = df_simu.query("ILI == True").copy()
        
        # get name face pair dict for each session
        sess_name_face = {}
        for sess in df_study.session.unique():
            sess_name_face[sess] = df_study.query(f"session == {sess}")[["study_itemno1", "study_itemno2"]].set_index("study_itemno2").to_dict()["study_itemno1"]
            
        # get distance between ILI and correct faces
        df_ILI["resp_face"] = df_ILI.apply(lambda x: sess_name_face[x["session"]][x["s_resp"]], axis=1)
        df_ILI["resp_corr_distance"] = df_ILI.apply(lambda x: face_distance[x["test_itemno"] - 1, x["resp_face"] - 1], axis=1)
        df_ILI["distance_bin"] = df_ILI.apply(lambda x: str(0.5 * (x["resp_corr_distance"] // 0.5 + 1)) if x["resp_corr_distance"] < 3.5 else ">3.5", axis=1)
        df_ILI["distance_bin"] = pd.Categorical(df_ILI["distance_bin"], categories=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", ">3.5"], ordered=True)

        # count possible ILI from all distance
        distance_cnt = {}
        for lst in distance_lsts:
            for d in lst:
                d_group = str(0.5 * (d // 0.5 + 1)) if d < 3.5 else ">3.5"
                if d_group in distance_cnt:
                    distance_cnt[d_group] += 1
                else:
                    distance_cnt[d_group] = 1
                    
        # get ILI probability
        df_ILI_distance = df_ILI.groupby("distance_bin")["test_itemno"].count().to_frame(name="ILI_cnt").reset_index()
        df_ILI_distance["ILI_poss"] = df_ILI_distance.apply(lambda x: distance_cnt[x["distance_bin"]], axis=1)
        df_ILI_distance["ILI_prob"] = df_ILI_distance["ILI_cnt"] / df_ILI_distance["ILI_poss"]
        ILI_mean = df_ILI_distance["ILI_prob"].to_numpy()
    
    except:  # sometimes there is no ILI
        ILI_mean = np.full(7, 0)
    
    # Get error
    with open("../../Analysis/simu8_cr_sim/simu8_data/simu8_gt.pkl", "rb") as f:
        neighbor_mean_gt = pickle.load(f)
        neighbor_se_gt = pickle.load(f)
        ILI_mean_gt = pickle.load(f)
        ILI_se_gt = pickle.load(f)
    wls_neighbor = get_wmse(neighbor_mean_gt, neighbor_mean, neighbor_se_gt) / len(neighbor_mean_gt)
    wls_ILI = get_wmse(ILI_mean_gt, ILI_mean, ILI_se_gt) / len(ILI_mean_gt)
    err = wls_neighbor + wls_ILI
    if correct_rate < 0.6:
        err += 5

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [neighbor_mean, ILI_mean]

    return err, cmr_stats


def obj_func_S1(param_vec, df_study, df_test, sem_mat, sources, return_df=False):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="S1")
    stats = []

    for i in [1, 2, 3]:
        # Separate 3 groups of simulation
        df_study_gp = df_study.query(f"group == {i}").copy()
        df_test_gp = df_test.query(f"group == {i}").copy()
        mode = "Recog-CR-Assoc" if i == 3 else "Recog-CR"

        # Run model with the parameters given in param_vec
        nitems = 4 * 48
        param_dict.update(nitems_in_accumulator=nitems, learn_while_retrieving=True, rec_time_limit=10000, use_new_context=True, use_flexible_thresh=True)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study_gp, df_test_gp, sem_mat, mode=mode, disable_tqdm=True)
        df_simu["test"] = df_test_gp["test"]
        df_simu = df_simu.merge(df_test_gp, on=["session", "test", "test_itemno1", "test_itemno2"])

        # Get behavioral stats
        subjects = np.unique(df_simu.subject)
        stats_gp = []
        for subj in subjects:
            df_subj = df_simu.query(f"subject == {subj}").copy()
            stats_gp.append(list(anal_perform_S1(df_subj)))
        stats_mean = np.mean(stats_gp, axis=0)
        stats.append(list(stats_mean))

    # Score the model's behavioral stats as compared with the true data
    stats = np.array(stats)
    ground_truth = np.array(
        [
            [0.19, 0.67, 0.15, 0.57],
            [0.30, 0.80, 0.12, 0.71],
            [0.42, 0.72, 0.22, 0.81],
        ]
    )  # p_rc, hr, far, q
    ground_truth_se = np.array(
        [
            [0.01, 0.02, 0.02, 0.05],
            [0.03, 0.02, 0.01, 0.04],
            [0.04, 0.03, 0.02, 0.02],
        ]
    )
    err = np.sum(np.power(stats - ground_truth, 2))
    # err = get_wmse(ground_truth, stats, ground_truth_se)

    # apply some constraints that pair FAR should not be 0
    if stats[1, 2] == 0:
        err += 1

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = stats

    if return_df:
        return err, cmr_stats, df_simu
    else:
        return err, cmr_stats


def anal_perform_S1(df_simu):

    # Get correctness
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # Recognition performance
    df_recog = df_simu.query("test == 1")
    recog_resp = df_recog["s_resp"].to_numpy()
    is_old = df_recog["correct_ans"].to_numpy()
    is_new = 1 - is_old
    old_num = np.sum(is_old)
    new_num = np.sum(is_new)
    hr = np.sum(recog_resp * is_old) / old_num
    far = np.sum(recog_resp * is_new) / new_num

    # Cued recall performance
    df_cr = df_simu.query("test == 2")
    cr_resp = df_cr["s_resp"].to_numpy()
    cr_truth = df_cr["correct_ans"].to_numpy()
    p_rc = np.mean(cr_resp == cr_truth)

    # successive test performance and calculate Q
    df_simu_study = df_simu.query("pair_idx >= 0")
    df_pair = pd.pivot_table(df_simu_study, index="pair_idx", columns="test", values="correct")
    test1_resp = df_pair[1].to_numpy(dtype=int)
    test2_resp = df_pair[2].to_numpy(dtype=int)
    A = np.sum((test1_resp == 1) & (test2_resp == 1)) + 0.5
    B = np.sum((test1_resp == 0) & (test2_resp == 1)) + 0.5
    C = np.sum((test1_resp == 1) & (test2_resp == 0)) + 0.5
    D = np.sum((test1_resp == 0) & (test2_resp == 0)) + 0.5
    q = (A * D - B * C) / (A * D + B * C)

    return p_rc, hr, far, q


def obj_func_S2(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="S2")

    # Run model with the parameters given in param_vec
    param_dict.update(learn_while_retrieving=True, use_new_context=True, use_flexible_thresh=True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="Recog-Recog", disable_tqdm=True)
    df_simu["test"] = df_test["test"]
    df_simu = df_simu.merge(df_test, on=["session", "list", "test", "test_itemno1", "test_itemno2"])

    # Get correctness
    df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

    # Get conditions
    def get_cond(x):
        this_type = x["type"]
        target = x["correct_ans"]
        if target == 1:
            if this_type == "Different_Item":
                return "Different_Item"
            elif this_type == "Item_Pair":
                return "Item_Pair"
            elif this_type == "Pair_Item":
                return "Pair_Item"
            elif this_type == "Same_Item":
                return "Same_Item"
            elif this_type == "Intact_Pair":
                return "Intact_Pair"
        elif target == 0:
            if this_type == "extra":
                return "NR_Lure"
            elif this_type == "Same_Item" or this_type == "Intact_Pair":
                return "Repeated_Lure"
            else:
                return "Discard"

    df_simu["condition"] = df_simu.apply(get_cond, axis=1)

    # Get behavioral stats
    subjects = np.unique(df_simu.subject)
    stats = []
    for subj in subjects:
        df_subj = df_simu.query(f"subject=={subj} and list % 3 != 0")  # discard first list
        stats_subj = anal_perform_S2(df_subj)
        stats.append(stats_subj)

    # Score the model's behavioral stats as compared with the true data
    stats_mean = np.nanmean(stats, axis=0)
    ground_truth = np.array(
        [
            [0.82, 0.68, 0.26],
            [0.82, 0.85, 0.64],
            [0.91, 0.85, 0.59],
            [0.81, 0.82, 0.86],
            [0.90, 0.92, 0.94],
            [0.07, 0.15, 0.54],
            [0.07, 0.06, 0],
        ]
    )
    ground_truth_se = np.array(
        [
            [0.020, 0.030, 0.10],  # diff item
            [0.016, 0.020, 0.12],  # item/pair
            [0.018, 0.021, 0.10],  # pair/item
            [0.017, 0.017, 0.03],  # same item
            [0.022, 0.019, 0.02],  # intact pair
            [0.014, 0.018, 0.12],  # repeated lure
            [0.009, 0.009, -1],  # non-repeated lure
        ]
    )
    err = np.sum(np.power(stats_mean - ground_truth, 2))
    # ground_truth_se[:, 2] /= 2
    # err = get_wmse(ground_truth, stats_mean, ground_truth_se)

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = stats_mean

    return err, cmr_stats


def anal_perform_S2(df_simu):

    # Get target items
    df_target = df_simu.query("condition != 'Discard'")

    # Get pairs data
    def get_pair(df_tmp):
        df_tmp_pair = pd.pivot_table(df_tmp, index=["pair_idx", "condition"], columns="test", values="correct")
        df_tmp_pair.columns = ["test1", "test2"]
        df_tmp_pair.reset_index(inplace=True)
        return df_tmp_pair

    df_p = df_target.query("condition != 'NR_Lure'")
    df_pair = get_pair(df_p).reset_index()

    # Get Q values
    def Yule_Q(A, B, C, D):
        return (A * D - B * C) / (A * D + B * C)

    qs = []
    conditions = ["Different_Item", "Item_Pair", "Pair_Item", "Same_Item", "Intact_Pair", "Repeated_Lure", "NR_Lure"]
    for cond in conditions:
        df_tmp = df_pair.query(f"condition == '{cond}'")
        test2_rsp = pd.Categorical(df_tmp.test2, categories=[0, 1])
        test1_rsp = pd.Categorical(df_tmp.test1, categories=[0, 1])
        df_tab = pd.crosstab(index=test2_rsp, columns=test1_rsp, rownames=["test2"], colnames=["test1"], normalize=False, dropna=False)

        try:
            q = Yule_Q(df_tab[1][1] + 0.5, df_tab[0][1] + 0.5, df_tab[1][0] + 0.5, df_tab[0][0] + 0.5)
        except:
            q = 0 if cond == "NR_Lure" else np.nan

        qs.append(q)

    # Get hit rates and aggregate
    df_res = pd.DataFrame({"Condition": conditions, "Q": qs})
    df_res.set_index("Condition", inplace=True)
    df_res["Test1_p"] = df_target.groupby(["test", "condition"])["s_resp"].mean()[1]
    df_res["Test2_p"] = df_target.groupby(["test", "condition"])["s_resp"].mean()[2]
    df_res = df_res[["Test1_p", "Test2_p", "Q"]]
    stats = df_res.values.tolist()

    return stats