import numpy as np
import pandas as pd
import json
import scipy as sp
from CMR_IA.utils import make_params, param_vec_to_dict
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

        what_to_fit = [
            "beta_enc",
            "beta_rec",
            "beta_cue",
            "beta_distract",
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
        ]

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


def get_wmse(y_true, y_pred, y_std):
    return np.sum((y_true - y_pred) ** 2 / y_std ** 2)


def Yule_Q(A, B, C, D):
    return (A * D - B * C) / (A * D + B * C)
    

def anal_perform_6b(df_simu):

    # Get pair
    df_pair = pd.pivot_table(df_simu, index="pair_idx", columns="test", values="correct")
    df_pair.columns = ["test1", "test2"]
    test2_rsp = pd.Categorical(df_pair.test2, categories=[1, 0])
    test1_rsp = pd.Categorical(df_pair.test1, categories=[1, 0])
    df_tab = pd.crosstab(index=test2_rsp, columns=test1_rsp, rownames=["test2"], colnames=["test1"], normalize=False, dropna=False)
    df_tab_norm = pd.crosstab(index=test2_rsp, columns=test1_rsp, rownames=["test2"], colnames=["test1"], normalize="all", dropna=False)
    t1_t2 = df_tab_norm[1][1]
    t1_f2 = df_tab_norm[1][0]
    f1_t2 = df_tab_norm[0][1]
    f1_f2 = df_tab_norm[0][0]

    # Compute Q
    q = Yule_Q(df_tab[1][1] + 0.5, df_tab[0][1] + 0.5, df_tab[1][0] + 0.5, df_tab[0][0] + 0.5)

    return t1_t2, t1_f2, f1_t2, f1_f2, q


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

    # Successive test performance and calculate Q
    df_simu_study = df_simu.query("pair_idx >= 0")
    df_pair = pd.pivot_table(df_simu_study, index="pair_idx", columns="test", values="correct")
    test1_resp = df_pair[1].to_numpy(dtype=int)
    test2_resp = df_pair[2].to_numpy(dtype=int)
    A = np.sum((test1_resp == 1) & (test2_resp == 1)) + 0.5
    B = np.sum((test1_resp == 0) & (test2_resp == 1)) + 0.5
    C = np.sum((test1_resp == 1) & (test2_resp == 0)) + 0.5
    D = np.sum((test1_resp == 0) & (test2_resp == 0)) + 0.5
    q = Yule_Q(A, B, C, D)

    return p_rc, hr, far, q


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

        assert df_study == None
        df = df_test
        
        # Run model
        df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, design="EXP1", disable_tqdm=True)
        df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

        # Calculate the rolling category length
        rolling_window = 9
        category_label_dummies = df_simu["category_label"].str.get_dummies()
        category_label_dummies.columns = ["cl_" + col for col in category_label_dummies.columns]
        category_label_dummies_events = pd.concat([df_simu, category_label_dummies], axis=1)
        cl_rolling_sum = category_label_dummies_events.groupby("session").rolling(rolling_window, min_periods=1, on="position")[category_label_dummies.columns].sum().reset_index()
        df_rollcat = df_simu.merge(cl_rolling_sum, on=["session", "position"])
        df_simu["roll_cat_label_length"] = df_rollcat.apply(lambda x: x["cl_" + x["category_label"]], axis=1)
        df_simu["roll_cat_label_length"] = df_simu["roll_cat_label_length"] - 1
        df_simu["roll_cat_len_level"] = pd.cut(x=df_simu.roll_cat_label_length, bins=[0, 2, np.inf], right=False, include_lowest=True, labels=["0-1", ">=2"]).astype("str")

        # Add log lag bin
        df_simu["log_lag"] = np.log(df_simu["lag"])
        df_simu["log_lag_bin"] = pd.cut(df_simu["log_lag"], np.arange(df_simu["log_lag"].max() + 1), labels=False, right=False)
        df_simu["log_lag_bin"] = df_simu.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
        df_simu["log_lag_bin"] = df_simu.apply(lambda x: 5 if x["log_lag_bin"] > 5 else x["log_lag_bin"], axis=1)

        # Construct local FAR
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

        # Distribute items into bins
        log_lag_bins = [0, 2, 3, 4, 5]
        for bin in log_lag_bins:
            col_name = "log_lag_bin_" + str(bin)
            df_simu[col_name] = (df_simu.log_lag_bin == bin) | (df_simu.log_lag_bin_newpre == bin) | (df_simu.log_lag_bin_newpost == bin)

        # Clean the first 20
        df_simu = df_simu.query("position >= 20").copy()

        # Get yes rate
        df_lst = []
        for bin in log_lag_bins:
            col_name = "log_lag_bin_" + str(bin)
            df_tmp = df_simu.query(col_name + " == True").groupby(["session", "old", "roll_cat_len_level"])["s_resp"].agg(["mean", "sum", "count"]).reset_index()
            df_tmp["log_lag_bin"] = bin
            df_lst.append(df_tmp)
        df_rollcat_laggp = pd.concat(df_lst)
        df_rollcat_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)

        # Pivot for hr and far
        df_rollcat_laggp["log_lag_disp"] = np.ceil(np.e**df_rollcat_laggp.log_lag_bin)
        df_rollcat_laggp["old"] = df_rollcat_laggp["old"].astype("str")
        df_dprime = pd.pivot_table(df_rollcat_laggp, values=["yes_rate"], index=["session", "roll_cat_len_level", "log_lag_disp"], columns="old").reset_index()
        df_dprime.columns = [" ".join(col).strip() for col in df_dprime.columns.values]
        df_dprime = df_dprime.rename(columns={"yes_rate False": "far", "yes_rate True": "hr"})

        # Calculate hr and far
        df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_disp"])[["hr", "far"]].mean().reset_index()
        hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["hr"].to_numpy()
        hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["hr"].to_numpy()
        far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"')["far"].to_numpy()
        far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"')["far"].to_numpy()

        # Calculate error
        with open("../../Analysis/simu1_recog_recsim/data/simu1_gt.json") as f:
            _gt = json.load(f)
        hr_lowsim_gt = np.array(_gt["hr_lowsim"])
        hr_lowsim_std_gt = np.array(_gt["hr_lowsim_std"])
        hr_highsim_gt = np.array(_gt["hr_highsim"])
        hr_highsim_std_gt = np.array(_gt["hr_highsim_std"])
        far_lowsim_gt = np.array(_gt["far_lowsim"])
        far_lowsim_std_gt = np.array(_gt["far_lowsim_std"])
        far_highsim_gt = np.array(_gt["far_highsim"])
        far_highsim_std_gt = np.array(_gt["far_highsim_std"])
        err = get_wmse(hr_lowsim_gt, hr_lowsim, hr_lowsim_std_gt) + get_wmse(hr_highsim_gt, hr_highsim, hr_highsim_std_gt) + get_wmse(far_lowsim_gt, far_lowsim, far_lowsim_std_gt) + get_wmse(far_highsim_gt, far_highsim, far_highsim_std_gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": []}
        

    ## SIMU2 ##
    elif simu_name == "2":

        # Run model
        df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])

        # Add lag condition
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

        # Construct local FAR
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

        # Get conditions
        df_t = df_simu.loc[pd.notna(df_simu.lag_cat)].copy()
        create_level = {0: "new_r", 1: "new_a", 2: "old_r", 3: "old_a"}
        df_t["level"] = df_t.apply(lambda x: create_level[x["old"] * 2 + (x["lag_cat"] == "a")], axis=1)

        # Get roc
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

        # Claculate err
        with open("../../Analysis/simu2_recog_conti/data/simu2_gt.json") as f:
            _gt = json.load(f)
        far_a_gt = np.array(_gt["far_a"])
        hr_a_gt = np.array(_gt["hr_a"])
        far_r_gt = np.array(_gt["far_r"])
        hr_r_gt = np.array(_gt["hr_r"])
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

        # Ensure a is above r in a range
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
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr_a_interp, hr_r_interp]}


    ## SIMU2b ##
    elif simu_name == "2b":

        # Run model
        df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat, design="Osth", disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "itemno1", "itemno2"])
        df_simu["old"] = df_simu.apply(lambda x: 1 if x["type"] == "intact" else 0, axis=1)
        df_simu["correct"] = df_simu["s_resp"] == df_simu["old"]

        # Get yes rate
        df_hrfar = df_simu.groupby(["session", "type"]).correct.mean().to_frame(name="yes_rate").reset_index()
        df_hrfar = df_hrfar.pivot(index="session", columns="type", values="yes_rate").reset_index()
        df_hrfar["hr"] = df_hrfar["intact"]
        df_hrfar["far"] = 1 - df_hrfar["rearranged"]
        df_hrfar_plot = pd.melt(df_hrfar, id_vars=["session"], value_vars=["hr", "far"], var_name="type", value_name="yes_rate")

        # Get far with lag
        df_lure = df_simu.query("type == 'rearranged'").copy()
        df_farlag = df_lure.groupby(["session", "lag"]).correct.mean().to_frame(name="yes_rate").reset_index()
        df_farlag["far"] = 1 - df_farlag["yes_rate"]

        # Calculate err
        with open("../../Analysis/simu2b_recog_assoc_conti/data/simu2b_gt.json") as f:
            _gt = json.load(f)
        hr_gt = np.array(_gt["hr"])
        hr_std_gt = np.array(_gt["hr_std"])
        far_gt = np.array(_gt["far"])
        far_std_gt = np.array(_gt["far_std"])
        hr = df_hrfar_plot.query("type == 'hr'").yes_rate.mean()
        far = df_farlag.groupby("lag").far.mean().to_numpy()
        err = 5 * get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr, far]}

    
    ## SIMU3 ##
    elif simu_name == "3":

        assert df_study == None
        df = df_test
        
        # Run model
        df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, design="Hockley", disable_tqdm=True)
        df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

        # Session-wise, calculate the yes_rate for each condition
        df_sess_laggp = df_simu.groupby(["session", "type", "lag"]).s_resp.agg(["count", "sum", "mean"]).reset_index()
        df_sess_laggp.rename(columns={"mean": "yes_rate"}, inplace=True)
        df_sess_laggp["yes_rate_adj"] = (df_sess_laggp["sum"] + 0.5) / (df_sess_laggp["count"] + 1)
        df_sess_laggp["z_yes_rate"] = sp.stats.norm.ppf(df_sess_laggp["yes_rate_adj"])

        # Collapse across session to get hit rate and false alarm rate
        df_laggp = df_sess_laggp.groupby(["type", "lag"]).yes_rate.mean().to_frame(name="yes_rate").reset_index()
        df_laggp["no_rate"] = 1 - df_laggp["yes_rate"]

        # Get the vectors
        I_hr = df_laggp.loc[df_laggp.type == "single_old", "yes_rate"].to_numpy()
        I_far = np.mean(df_laggp.loc[df_laggp.type == "single_new", "yes_rate"].astype(float))
        A_hr = df_laggp.loc[df_laggp.type == "pair_old", "yes_rate"].to_numpy()
        A_far = df_laggp.loc[df_laggp.type == "pair_new", "yes_rate"].to_numpy()

        # Calculate err
        with open("../../Analysis/simu3_recog_forget/data/simu3_gt.json") as f:
            _gt = json.load(f)
        I_hr_gt = np.array(_gt["I_hr"])
        I_far_gt = np.array(_gt["I_far"])
        A_hr_gt = np.array(_gt["A_hr"])
        A_cr_gt = np.array(_gt["A_cr"])
        A_far_gt = 1 - A_cr_gt
        err = np.sum((I_hr - I_hr_gt) ** 2) + np.sum((A_hr - A_hr_gt) ** 2) + (I_far - I_far_gt) ** 2 * 5 + np.sum((A_far - A_far_gt) ** 2)

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

        # Session-wise, get yes rate for each condition
        df_sess_q = df_simu.groupby(["session", "quantile", "old"]).s_resp.mean().to_frame(name="yes_rate").reset_index()
        
        # Collapse across session
        df_q = df_sess_q.groupby(["quantile", "old"]).yes_rate.mean().to_frame().reset_index()

        # Get behavioral stats and compare with ground truth
        with open("../../Analysis/simu4_recog_wfe/data/simu4_gt.json") as f:
            _gt = json.load(f)
        hr_gt = np.array(_gt["hr"])
        hr_std_gt = np.array(_gt["hr_std"])
        far_gt = np.array(_gt["far"])
        far_std_gt = np.array(_gt["far_std"])
        hr = df_q.query("old == True")["yes_rate"].to_numpy()
        far = df_q.query("old == False")["yes_rate"].to_numpy()
        err = get_wmse(hr_gt, hr, hr_std_gt) + get_wmse(far_gt, far, far_std_gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": [hr, far]}

    
    ## SIMU5 ##
    elif simu_name == "5":

        # Run model
        param_dict.update(nitems_in_accumulator=48)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Session-wise, calculate correct rate for each lag
        df_sess_lag = df_simu.groupby(["session", "lag"]).correct.mean().to_frame(name="correct_rate").reset_index()
        
        # Collapse across sessions
        hr = df_sess_lag.groupby("lag").correct_rate.mean().to_numpy()

        # Get error
        with open("../../Analysis/simu5_cr_rec/data/simu5_gt.json") as f:
            _gt = json.load(f)
        hr_gt = np.array(_gt["hr"])
        hr_std_gt = np.array(_gt["hr_std"])
        err = get_wmse(hr_gt, hr, hr_std_gt)
        cmr_stats = {"err": err, "params": param_vec, "stats": hr}


    ## SIMU6a ##
    elif simu_name == "6a":

        # Run model
        param_dict.update(nitems_in_accumulator=48)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Clean first 2 list
        df_simu = df_simu.query("list > 1")

        # Session-wise, calculate correct rate for each condition
        df_sess_lag = df_simu.groupby(["session", "lag", "order"]).correct.mean().to_frame(name="correct_rate").reset_index()
        
        # Collapse across sessions
        df_lag = df_sess_lag.groupby(["lag", "order"]).correct_rate.mean().to_frame(name="correct_rate").reset_index()
        fw = df_lag.query("order == 1").correct_rate.values
        bw = df_lag.query("order == 2").correct_rate.values

        # Get error
        with open("../../Analysis/simu6a_cr_recsym/data/simu6a_gt.json") as f:
            _gt = json.load(f)
        fw_gt = np.array(_gt["fw"])
        bw_gt = np.array(_gt["bw"])
        err = np.power(fw - fw_gt, 2).sum() + np.power(bw - bw_gt, 2).sum()
        cmr_stats = {"err": err, "params": param_vec, "stats": [fw, bw]}


    ## SIMU6b ##
    elif simu_name == "6b":

        # Run model
        param_dict.update(learn_while_retrieving=True, nitems_in_accumulator=96)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="CR-CR", disable_tqdm=True)
        df_simu["test_pos"] = df_test["test_pos"]
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno1", "test_itemno2", "test_pos"])

        # Get correctness
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Get conditions
        df_cond = df_simu.groupby(["pair_idx", "test"])["order"].mean().to_frame(name="corr_rate").reset_index()
        df_cond = df_cond.pivot_table(index="pair_idx", columns="test", values="corr_rate").reset_index()
        df_cond.columns = ["pair_idx", "test1", "test2"]

        # Get condition and congruence
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
        with open("../../Analysis/simu6b_cr_sym/data/simu6b_gt.json") as f:
            _gt = json.load(f)
        inde_ground_truth = np.array(_gt["inde"])
        reve_ground_truth = np.array(_gt["reve"])
        err = np.sum(np.power(inde_stats_mean - inde_ground_truth, 2)) + np.sum(np.power(reve_stats_mean - reve_ground_truth, 2)) \
            + np.power(inde_stats_mean[-1] - inde_ground_truth[-1], 2) + np.power(reve_stats_mean[-1] - reve_ground_truth[-1], 2)
        cmr_stats = {"err": err, "params": param_vec, "stats": [inde_stats_mean, reve_stats_mean]}

    
    ## SIMU7 ##
    elif simu_name == "7":

        # Run model
        param_dict.update(nitems_in_accumulator=96)
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat, disable_tqdm=True)
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans

        # Get the study list and study pos of response
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

        # Get intrusion type
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

        # Clean list 1
        df_simu = df_simu.query("list > 0").copy()

        # Get overall prob
        df_cnt = df_simu.groupby(["session", "intrusion_type"]).s_resp.count().to_frame(name="count").reset_index()

        df_cnt_correct = df_cnt.query("intrusion_type == 'Correct'").copy()
        df_cnt_correct["total"] = df_simu.groupby("session").test_item.count().tolist()
        df_cnt_correct["p"] = df_cnt_correct["count"] / df_cnt_correct["total"]
        p_correct_mean = np.mean(df_cnt_correct["p"])

        df_cnt_ILI = df_cnt.query("intrusion_type == 'ILI'").copy()
        df_cnt_ILI["total"] = df_simu.groupby("session").test_item.count().tolist()
        df_cnt_ILI["p"] = df_cnt_ILI["count"] / df_cnt_ILI["total"]
        p_ILI_mean = np.mean(df_cnt_ILI["p"])

        df_cnt_PLI = df_cnt.query("intrusion_type == 'PLI'").copy()
        df_cnt_PLI["total"] = df_simu.groupby("session").test_item.count().tolist()
        df_cnt_PLI["p"] = df_cnt_PLI["count"] / df_cnt_PLI["total"]
        p_PLI_mean = np.mean(df_cnt_PLI["p"])

        try:
            # PLI
            # Pick list > 5 and list_lag -5 to -1
            df_PLI = df_simu.query("intrusion_type == 'PLI' and list > 5 and list_lag > -6").copy()
            df_PLI["abs_list_lag"] = df_PLI["list_lag"].abs().astype(int)
            df_PLI["abs_list_lag"] = pd.Categorical(df_PLI["abs_list_lag"], categories=[1, 2, 3, 4, 5], ordered=True)

            # Session-wise, count PLI
            df_PLI_sess = df_PLI.groupby(["session"]).test_item.count().to_frame(name="PLI_cnt_sess").reset_index()
            
            # Session-wise, count PLI by list_lag
            df_PLI_sess_lag = df_PLI.groupby(["session", "abs_list_lag"]).test_item.count().to_frame(name="PLI_cnt").reset_index()
            
            # Calculate PLI probability
            df_PLI_sess_lag = pd.merge(df_PLI_sess_lag, df_PLI_sess, on="session")
            df_PLI_sess_lag["PLI_prob"] = df_PLI_sess_lag["PLI_cnt"] / df_PLI_sess_lag["PLI_cnt_sess"]
            lag_PLI_mean = df_PLI_sess_lag.groupby("abs_list_lag").PLI_prob.mean().values

            # ILI
            # Get pos lag
            df_ILI = df_simu.query("intrusion_type == 'ILI'").copy()
            df_ILI["pos_lag"] = df_ILI["pos_lag"].astype(int)
            df_ILI["pos_lag"] = pd.Categorical(df_ILI["pos_lag"], categories=np.concatenate([np.arange(-11, 0), np.arange(1, 12)]), ordered=True)

            # Session-wise, calculate ILI probability for each lag
            def get_ILI_prob(df_tmp):
                possible_ILI_cnt = {}
                for pair_pos in df_tmp.study_pos:
                    l_bound = -pair_pos
                    r_bound = 11 - pair_pos
                    for i in np.arange(l_bound, r_bound + 1):
                        if i in possible_ILI_cnt:
                            possible_ILI_cnt[i] += 1
                        else:
                            possible_ILI_cnt[i] = 1
                df_tmp_lag = df_tmp.groupby("pos_lag")["test_item"].count().to_frame(name="ILI_cnt")
                df_tmp_lag["possible_ILI_cnt"] = df_tmp_lag.index.map(possible_ILI_cnt).astype(float)
                df_tmp_lag["ILI_prob"] = df_tmp_lag["ILI_cnt"] / df_tmp_lag["possible_ILI_cnt"]
                return df_tmp_lag
            df_ILI_sess_lag = df_ILI.groupby("session").apply(get_ILI_prob).reset_index()
            df_ILI_sess_lag = df_ILI_sess_lag.query("pos_lag > -6 and pos_lag < 6").copy()
            df_ILI_sess_lag["pos_lag_int"] = df_ILI_sess_lag["pos_lag"].astype(int)
            lag_ILI_mean = df_ILI_sess_lag.groupby("pos_lag_int").ILI_prob.mean().values

        except:  # sometimes there is no PLI or ILI
            lag_PLI_mean = np.full(5, 0)
            lag_ILI_mean = np.full(10, 0)

        # Get error
        with open("../../Analysis/simu7_cr_pliili/data/simu7_gt.json") as f:
            _gt = json.load(f)
        p_correct_mean_gt = np.array(_gt["p_correct_mean"])
        p_correct_se_gt = np.array(_gt["p_correct_se"])
        p_PLI_mean_gt = np.array(_gt["p_PLI_mean"])
        p_PLI_se_gt = np.array(_gt["p_PLI_se"])
        p_ILI_mean_gt = np.array(_gt["p_ILI_mean"])
        p_ILI_se_gt = np.array(_gt["p_ILI_se"])
        lag_PLI_mean_gt = np.array(_gt["lag_PLI_mean"])
        lag_PLI_se_gt = np.array(_gt["lag_PLI_se"])
        lag_ILI_mean_gt = np.array(_gt["lag_ILI_mean"])
        lag_ILI_se_gt = np.array(_gt["lag_ILI_se"])
        wls_p_correct = get_wmse(p_correct_mean_gt, p_correct_mean, p_correct_se_gt)
        wls_p_PLI = get_wmse(p_PLI_mean_gt, p_PLI_mean, p_PLI_se_gt)
        wls_p_ILI = get_wmse(p_ILI_mean_gt, p_ILI_mean, p_ILI_se_gt)
        wls_lag_PLI = get_wmse(lag_PLI_mean_gt, lag_PLI_mean, lag_PLI_se_gt) / len(lag_PLI_mean_gt)
        wls_lag_ILI = get_wmse(lag_ILI_mean_gt, lag_ILI_mean, lag_ILI_se_gt) / len(lag_ILI_mean_gt)
        err = wls_p_correct + wls_p_PLI + wls_p_ILI + wls_lag_PLI + wls_lag_ILI
        cmr_stats = {"err": err, "params": param_vec, "stats": [p_correct_mean, p_PLI_mean, p_ILI_mean, lag_PLI_mean, lag_ILI_mean]}


    ## SIMU8 ##
    elif simu_name == "8":

        # Run model
        param_dict.update(nitems_in_accumulator=16, ban_recall=np.arange(1, 17))
        df_simu, _, _ = cmr.run_norm_cr_multi_sess(param_dict, df_study, df_test, sem_mat)
        df_simu = df_simu.merge(df_test, on=["session", "list", "test_itemno"])
        df_simu["correct"] = df_simu.s_resp == df_simu.correct_ans
        correct_rate = sum(df_simu.correct) / len(df_simu.correct)

        # Load and get face distance
        face_distance = np.load("../../Analysis/simu8_cr_sim/data/simu8_distance.npy")
        thresh = 3.0

        # Neighbor
        # Get number of neighbours by distance
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

        # Get the correct rate by neighbour group
        df_neighbour_group = df_simu.query("neighbour_group > 0").groupby("neighbour_group").correct.mean().reset_index()
        neighbor_mean = df_neighbour_group["correct"].to_numpy()

        # ILI
        try:
            # Detect ILI
            def get_ILI(df_tmp):
                resp_names = df_tmp["s_resp"].values
                study_names = df_tmp["correct_ans"].values
                is_studied = np.isin(resp_names, study_names)
                is_incorrect = df_tmp["correct"] == False
                is_ILI = is_studied & is_incorrect
                return is_ILI
            df_simu["ILI"] = df_simu.groupby("session").apply(get_ILI).to_frame(name="ILI").reset_index()["ILI"].to_list()
            df_ILI = df_simu.query("ILI == True").copy()

            # Get name face pair dict for each session
            sess_name_face = {}
            for sess in df_study.session.unique():
                sess_name_face[sess] = df_study.query(f"session == {sess}")[["study_itemno1", "study_itemno2"]].set_index("study_itemno2").to_dict()["study_itemno1"]

            # Get distance between ILI and correct faces
            df_ILI["resp_face"] = df_ILI.apply(lambda x: sess_name_face[x["session"]][x["s_resp"]], axis=1)
            df_ILI["resp_corr_distance"] = df_ILI.apply(lambda x: face_distance[x["test_itemno"] - 1, x["resp_face"] - 1], axis=1)
            df_ILI["distance_bin"] = df_ILI.apply(lambda x: str(0.5 * (x["resp_corr_distance"] // 0.5 + 1)) if x["resp_corr_distance"] < 3.5 else ">3.5", axis=1)
            df_ILI["distance_bin"] = pd.Categorical(df_ILI["distance_bin"], categories=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", ">3.5"], ordered=True)

            # Count possible ILI from all distance
            distance_cnt = {}
            for lst in distance_lsts:
                for d in lst:
                    d_group = str(0.5 * (d // 0.5 + 1)) if d < 3.5 else ">3.5"
                    if d_group in distance_cnt:
                        distance_cnt[d_group] += 1
                    else:
                        distance_cnt[d_group] = 1

            # Get ILI probability
            df_ILI_distance = df_ILI.groupby("distance_bin")["test_itemno"].count().to_frame(name="ILI_cnt").reset_index()
            df_ILI_distance["ILI_poss"] = df_ILI_distance.apply(lambda x: distance_cnt[x["distance_bin"]], axis=1)
            df_ILI_distance["ILI_prob"] = df_ILI_distance["ILI_cnt"] / df_ILI_distance["ILI_poss"]
            ILI_mean = df_ILI_distance["ILI_prob"].to_numpy()

        except:
            # Sometimes there is no ILI
            ILI_mean = np.full(7, 0)

        # Get error
        with open("../../Analysis/simu8_cr_sim/data/simu8_gt.json") as f:
            _gt = json.load(f)
        neighbor_mean_gt = np.array(_gt["neighbor_mean"])
        neighbor_se_gt = np.array(_gt["neighbor_se"])
        ILI_mean_gt = np.array(_gt["ILI_mean"])
        ILI_se_gt = np.array(_gt["ILI_se"])
        wls_neighbor = get_wmse(neighbor_mean_gt, neighbor_mean, neighbor_se_gt) / len(neighbor_mean_gt)
        wls_ILI = get_wmse(ILI_mean_gt, ILI_mean, ILI_se_gt) / len(ILI_mean_gt)
        err = wls_neighbor + wls_ILI
        if correct_rate < 0.6:
            err += 5
        cmr_stats = {"err": err, "params": param_vec, "stats": [neighbor_mean, ILI_mean]}

    
    ## SIMUS1 ##
    elif simu_name == "S1":

        # Separate 3 groups of simulation
        stats = []
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
        with open("../../Analysis/simuS1_recog_cr/data/simuS1_gt.json") as f:
            _gt = json.load(f)
        ground_truth = np.array([_gt["g1_mean"], _gt["g2_mean"], _gt["g3_mean"]])
        err = np.sum(np.power(stats - ground_truth, 2))
        
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
            df_subj = df_simu.query(f"subject=={subj} and list % 3 != 0")
            stats_subj = anal_perform_S2(df_subj)
            stats.append(stats_subj)

        # Score the model's behavioral stats as compared with the true data
        stats_mean = np.nanmean(stats, axis=0)
        with open("../../Analysis/simuS2_recog_recog/data/simuS2_gt.json") as f:
            _gt = json.load(f)
        _conds = ["diff_item", "item_pair", "pair_item", "same_item", "intact_pair", "rep_lure", "nrep_lure"]
        ground_truth = np.array([_gt[f"{c}_mean"] for c in _conds])
        err = np.sum(np.power(stats_mean - ground_truth, 2))
        cmr_stats = {"err": err, "params": param_vec, "stats": stats_mean}

    else:
        raise ValueError(f"Unknown simu_name: {simu_name!r}")

    return err, cmr_stats
