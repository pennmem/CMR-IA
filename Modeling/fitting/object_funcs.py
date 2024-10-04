import numpy as np
import scipy.stats as ss
import CMR_IA as cmr
import time
import pandas as pd
import math
import pickle
import scipy as sp
from scipy.stats import norm
from optimization_utils import param_vec_to_dict


def obj_func_S1(param_vec, df_study, df_test, sem_mat, sources, return_df=False):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="S1")
    stats = []

    for i in [1, 2, 3]:
        # Separate 3 groups of simulation
        df_study_gp = df_study.query(f"group == {i}").copy()
        df_test_gp = df_test.query(f"group == {i}").copy()

        # Run model with the parameters given in param_vec
        nitems = 4 * 48
        param_dict.update(nitems_in_accumulator=nitems, learn_while_retrieving=True, rec_time_limit=10000, use_new_context=True)
        df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study_gp, df_test_gp, sem_mat)
        # print(df_simu)
        # print(df_test_gp)
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
    err = np.mean(np.power(stats - ground_truth, 2))

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
    df_recog = df_simu.query("test==1")
    recog_resp = df_recog["s_resp"].to_numpy()
    is_old = df_recog["correct_ans"].to_numpy()
    is_new = 1 - is_old
    old_num = np.sum(is_old)
    new_num = np.sum(is_new)
    hr = np.sum(recog_resp * is_old) / old_num
    far = np.sum(recog_resp * is_new) / new_num

    # Cued recall performance
    df_cr = df_simu.query("test==2")
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
    param_dict.update(learn_while_retrieving=True, use_new_context=True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="Recog-Recog")
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
        # df_subj = df_simu.query(f"subject=={subj}")
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
    err = np.mean(np.power(stats_mean - ground_truth, 2))

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


def obj_func_6b(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="6b")

    # Run model with the parameters given in param_vec
    param_dict.update(learn_while_retrieving=True, nitems_in_accumulator=96, use_new_context=True)
    df_simu, _, _ = cmr.run_success_multi_sess(param_dict, df_study, df_test, sem_mat, mode="CR-CR")
    df_simu["test_pos"] = np.tile(np.arange(1, 25), 600)  # 100 * 6
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
    err = (np.mean(np.power(inde_stats_mean - inde_ground_truth, 2)) + np.mean(np.power(reve_stats_mean - reve_ground_truth, 2))) / 2

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


def obj_func_3(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="3")
    param_dict.update(use_new_context=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Hockley")
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

    # group by type and lag
    df_laggp = df_simu.groupby(["type", "lag"]).s_resp.mean().to_frame(name="yes_rate").reset_index()

    # get d prime
    # df_dprime = pd.DataFrame()
    # df_dprime['lag'] = [2,4,6,8,16]
    # df_dprime['I_z_hr'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 1, 'yes_rate'].astype(float))
    # df_dprime['I_z_far'] = np.mean(sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 0, 'yes_rate'].astype(float)))
    # df_dprime['I_dprime'] = df_dprime['I_z_hr'] - df_dprime['I_z_far']
    # df_dprime['A_z_hr'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 2, 'yes_rate'].astype(float))
    # df_dprime['A_z_far'] = sp.stats.norm.ppf(df_laggp.loc[df_laggp.type == 3, 'yes_rate'].astype(float))
    # df_dprime['A_dprime'] = df_dprime['A_z_hr'] - df_dprime['A_z_far']

    # get the vectors
    I_hr = df_laggp.loc[df_laggp.type == "single_old", "yes_rate"].to_numpy()
    I_far = np.mean(df_laggp.loc[df_laggp.type == "single_new", "yes_rate"].astype(float))
    A_hr = df_laggp.loc[df_laggp.type == "pair_old", "yes_rate"].to_numpy()
    A_far = df_laggp.loc[df_laggp.type == "pair_new", "yes_rate"].to_numpy()

    # ground truth
    I_hr_gt = np.array([0.865, 0.811, 0.752, 0.746, 0.708])
    I_far_gt = 0.15  # 0.12
    A_hr_gt = np.array([0.843, 0.787, 0.720, 0.735, 0.646])
    A_far_gt = np.array([0.406, 0.371, 0.285, 0.259, 0.202])

    # calculate the error
    pow_err = np.mean(np.power(I_hr - I_hr_gt, 2)) + np.mean(np.power(A_hr - A_hr_gt, 2)) + np.power(I_far - I_far_gt, 2) * 5 + np.mean(np.power(A_far - A_far_gt, 2))
    abs_err = np.mean(np.abs(I_hr - I_hr_gt)) + np.mean(np.abs(A_hr - A_hr_gt)) + np.abs(I_far - I_far_gt) * 5 + np.mean(np.abs(A_far - A_far_gt))
    err = pow_err + abs_err / 10

    # apply some constraints
    if not (I_hr[0] > I_hr[1] and I_hr[1] > I_hr[2] and I_hr[2] > I_hr[3] and I_hr[3] > I_hr[4]):
        err += 1
    if not (A_hr[0] > A_hr[1] and A_hr[1] > A_hr[2] and A_hr[2] > A_hr[3] and A_hr[3] > A_hr[4]):
        err += 1
    if not (I_hr > A_hr).all():
        err += 1
    if not (A_far[0] > A_far[1] and A_far[1] > A_far[2] and A_far[2] > A_far[3] and A_far[3] > A_far[4]):
        err += 1

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [I_hr, I_far, A_hr, A_far]

    return err, cmr_stats


def obj_func_1(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="1")
    param_dict.update(use_new_context=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Continuous")
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

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
    df_simu["log_lag_bin"] = df_simu.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
    df_simu["log_lag_bin_newpre"] = df_simu.apply(lambda x: 0 if x["log_lag_bin_newpre"] == 1 else x["log_lag_bin_newpre"], axis=1)
    df_simu["log_lag_bin_newpost"] = df_simu.apply(lambda x: 0 if x["log_lag_bin_newpost"] == 1 else x["log_lag_bin_newpost"], axis=1)

    # distribute items into bins
    log_lag_bins = [0, 2, 3, 4, 5]
    for bin in log_lag_bins:
        col_name = "log_lag_bin_" + str(bin)
        df_simu[col_name] = (df_simu.log_lag_bin == bin) | (df_simu.log_lag_bin_newpre == bin) | (df_simu.log_lag_bin_newpost == bin)

    # group by rollcat and lagbin
    df_lst = []
    for bin in log_lag_bins:
        col_name = "log_lag_bin_" + str(bin)
        df_tmp = df_simu.query(col_name + " == True").groupby(["session", "old", "roll_cat_len_level"])["s_resp"].mean().to_frame(name="yes_rate").reset_index()
        df_tmp["log_lag_bin"] = bin
        df_lst.append(df_tmp)
    df_rollcat_laggp = pd.concat(df_lst)

    # pivot for hr and far
    df_rollcat_laggp["old"] = df_rollcat_laggp["old"].astype("str")
    df_dprime = pd.pivot_table(df_rollcat_laggp, values="yes_rate", index=["session", "roll_cat_len_level", "log_lag_bin"], columns="old").reset_index()
    df_dprime = df_dprime.rename(columns={"False": "far", "True": "hr"})

    # get hr and far
    df_hrfar = df_dprime.groupby(["roll_cat_len_level", "log_lag_bin"])[["hr", "far"]].mean().reset_index()
    hr_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"').hr.to_numpy()
    hr_highsim = df_hrfar.query('roll_cat_len_level == ">=2"').hr.to_numpy()
    far_lowsim = df_hrfar.query('roll_cat_len_level == "0-1"').far.to_numpy()
    far_highsim = df_hrfar.query('roll_cat_len_level == ">=2"').far.to_numpy()

    # ground truth
    hr_lowsim_gt = np.array([0.885, 0.853, 0.787, 0.682, 0.630])
    hr_highsim_gt = np.array([0.893, 0.858, 0.795, 0.720, 0.671])
    far_lowsim_gt = np.array([0.190, 0.190, 0.190, 0.195, 0.212])
    far_highsim_gt = np.array([0.202, 0.210, 0.216, 0.229, 0.237])  # a bit diff from real gt

    # calculate the error
    pow_err = np.mean(np.power(hr_lowsim - hr_lowsim_gt, 2)) + np.mean(np.power(hr_highsim - hr_highsim_gt, 2)) + np.mean(np.power(far_lowsim - far_lowsim_gt, 2)) + np.mean(np.power(far_highsim - far_highsim_gt, 2))
    abs_err = np.mean(np.abs(hr_lowsim - hr_lowsim_gt)) + np.mean(np.abs(hr_highsim - hr_highsim_gt)) + np.mean(np.abs(far_lowsim - far_lowsim_gt)) + np.mean(np.abs(far_highsim - far_highsim_gt))
    err = pow_err + abs_err / 10
    if np.isnan(err):
        err = 10

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr_lowsim, hr_highsim, far_lowsim, far_highsim]

    return err, cmr_stats


def obj_func_1_Az(param_vec, df_study, df_test, sem_mat, sources):

    assert df_study == None
    df = df_test

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="1")
    param_dict.update(use_new_context=True)

    # Run model with the parameters given in param_vec
    df_simu = cmr.run_conti_recog_multi_sess(param_dict, df, sem_mat, mode="Continuous")
    df_simu = df_simu.merge(df, on=["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"])

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
    df_simu["log_lag_bin"] = df_simu.apply(lambda x: 0 if x["log_lag_bin"] == 1 else x["log_lag_bin"], axis=1)
    df_simu["log_lag_bin_newpre"] = df_simu.apply(lambda x: 0 if x["log_lag_bin_newpre"] == 1 else x["log_lag_bin_newpre"], axis=1)
    df_simu["log_lag_bin_newpost"] = df_simu.apply(lambda x: 0 if x["log_lag_bin_newpost"] == 1 else x["log_lag_bin_newpost"], axis=1)

    # distribute items into bins
    log_lag_bins = [0, 2, 3, 4, 5]
    for bin in log_lag_bins:
        col_name = "log_lag_bin_" + str(bin)
        df_simu[col_name] = (df_simu.log_lag_bin == bin) | (df_simu.log_lag_bin_newpre == bin) | (df_simu.log_lag_bin_newpost == bin)

    # get Az
    def calculate_Az(df_tmp1):
        log_lag_bins = [0, 2, 3, 4, 5]
        Azs = []
        for bin in log_lag_bins:
            # get the df of this log_lag_bin
            col_name = "log_lag_bin_" + str(bin)
            df_tmp = df_tmp1.query(col_name + " == True").copy()
            # get variables
            conf = df_tmp.csim.to_numpy()
            truth = df_tmp.old.to_numpy()
            old_num = np.sum(truth)
            new_num = np.sum(~truth)
            is_old = truth
            is_new = ~truth
            if np.sum(truth) == 0 or np.sum(~truth) == 0:
                Azs.append(np.nan)
                continue
            min_conf = np.round(np.min(conf), 2)
            max_conf = np.round(np.max(conf), 2)
            if max_conf - min_conf < 0.1:
                Azs.append(np.nan)
                continue
            # calculate HR and FAR for different thresholds
            step = 0.02
            thresholds = np.arange(min_conf + step, max_conf, step)
            hrs = []
            fars = []
            old_conf = conf * is_old
            new_conf = conf * is_new
            for thresh in thresholds:
                hr = (np.sum(old_conf > thresh) + 0.5) / (old_num + 1)
                far = (np.sum(new_conf > thresh) + 0.5) / (new_num + 1)
                hrs.append(hr)
                fars.append(far)
            # calculate z_hr and z_far
            z_hr = norm.ppf(hrs)
            z_far = norm.ppf(fars)
            # linear regression on z_hr and z_far manually
            try:
                n = len(z_far)
                X = np.column_stack((np.ones(n), z_far))
                beta = np.linalg.inv(X.T @ X) @ X.T @ z_hr
                intercept, slope = beta
                # get A_z
                Az = norm.cdf(intercept / np.sqrt(1 + slope**2))
                Azs.append(Az)
            except:
                Azs.append(np.nan)
        # df to return
        df_return = pd.DataFrame({"log_lag_bin": log_lag_bins, "Az": Azs})
        return df_return

    df_Az = df_simu.groupby(["session", "roll_cat_len_level"]).apply(calculate_Az).reset_index()
    df_plot = df_Az.groupby(["roll_cat_len_level", "log_lag_bin"]).Az.mean().to_frame(name="Az").reset_index()
    Az_lowsim = df_plot.query("roll_cat_len_level == '0-1'").Az.to_numpy()
    Az_highsim = df_plot.query("roll_cat_len_level == '>=2'").Az.to_numpy()

    # ground truth
    Az_lowsim_gt = np.array([0.82, 0.82, 0.80, 0.73, 0.63])
    Az_highsim_gt = np.array([0.81, 0.78, 0.76, 0.69, 0.61])

    # calculate the error
    err = np.mean(np.power(Az_lowsim - Az_lowsim_gt, 2)) + np.mean(np.power(Az_highsim - Az_highsim_gt, 2))
    if np.isnan(err):
        err = 10

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [Az_lowsim, Az_highsim]

    return err, cmr_stats


def obj_func_4ctrl(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4ctrl")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "itemno"])

    # get the stats of each item
    df_itemgp = pd.pivot_table(df_simu, values="s_resp", index=["itemno"], columns=["old"], aggfunc="mean")
    df_itemgp.columns = ["far", "hr"]
    df_itemgp = df_itemgp.reset_index()
    df_itemfq = df_simu.groupby(["itemno"])[["freq", "quantile"]].mean().reset_index()
    df_itemgp = df_itemgp.merge(df_itemfq, on=["itemno"])

    # get the stats of each group
    df_quantgp = df_itemgp.groupby(["quantile"]).agg({"hr": "mean", "far": "mean", "freq": "mean"}).reset_index()
    df_quantgp["freq_mean"] = df_quantgp["freq"].round(0)

    # Get behavioral stats and compare with ground truth
    hr = df_quantgp.hr.to_numpy()
    far = df_quantgp.far.to_numpy()
    hr_gt = np.array([0.903, 0.885, 0.888, 0.880, 0.879, 0.880, 0.870, 0.862, 0.842, 0.837])
    far_gt = np.array([0.114, 0.132, 0.143, 0.164, 0.171, 0.183, 0.187, 0.193, 0.192, 0.193])
    err = np.mean(np.power(hr - hr_gt, 2)) + np.mean(np.power(far - far_gt, 2))

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats


def obj_func_4shift(param_vec, df_study, df_test, sem_mat, sources):

    # Reformat parameter vector to the dictionary format expected by CMR2
    param_dict = param_vec_to_dict(param_vec, sim_name="4shift")

    # Run model with the parameters given in param_vec
    param_dict.update(use_new_context=True)
    df_simu = cmr.run_norm_recog_multi_sess(param_dict, df_study, df_test, sem_mat)
    df_simu = df_simu.merge(df_test, on=["session", "itemno"])

    # get the stats of each item
    df_itemgp = pd.pivot_table(df_simu, values="s_resp", index=["itemno"], columns=["old"], aggfunc="mean")
    df_itemgp.columns = ["far", "hr"]
    df_itemgp = df_itemgp.reset_index()
    df_itemfq = df_simu.groupby(["itemno"])[["freq", "quantile"]].mean().reset_index()
    df_itemgp = df_itemgp.merge(df_itemfq, on=["itemno"])

    # get the stats of each group
    df_quantgp = df_itemgp.groupby(["quantile"]).agg({"hr": "mean", "far": "mean", "freq": "mean"}).reset_index()
    df_quantgp["freq_mean"] = df_quantgp["freq"].round(0)

    # Get behavioral stats and compare with ground truth
    hr = df_quantgp.hr.to_numpy()
    far = df_quantgp.far.to_numpy()
    hr_gt = np.array([0.903, 0.885, 0.888, 0.880, 0.879, 0.880, 0.870, 0.862, 0.842, 0.837])
    far_gt = np.array([0.114, 0.132, 0.143, 0.164, 0.171, 0.183, 0.187, 0.193, 0.192, 0.193])
    err = np.mean(np.power(hr - hr_gt, 2)) + np.mean(np.power(far - far_gt, 2))

    cmr_stats = {}
    cmr_stats["err"] = err
    cmr_stats["params"] = param_vec
    cmr_stats["stats"] = [hr, far]

    return err, cmr_stats
