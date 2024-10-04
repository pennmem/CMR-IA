import json
import numpy as np
import scipy.stats as ss
import CMR_IA as cmr
import time
import pandas as pd
import math
import pickle
import scipy.optimize as opt
import scipy.stats as st


def param_vec_to_dict(param_vec, sim_name):
    """
    Convert parameter vector to dictionary format expected by CMR2.
    """

    # Generate a base paramater dictionary
    param_dict = cmr.make_default_params()
    # param_dict.update(c_thresh = 0.4)

    # Put vector values into dict
    _, _, what_to_fit = make_boundary(sim_name)
    for name, value in zip(what_to_fit, param_vec):
        param_dict[name] = value

    return param_dict


def make_boundary(sim_name):
    """
    Make two vectors of boundary for parameters you want to fit.
    """

    # Generate a base paramater dictionary
    lb_dict = cmr.make_params()
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
    )

    ub_dict = cmr.make_params()
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
        c_thresh_itm=1,
        c_thresh_assoc=1,
        lamb=0.25,
        gamma_fc=1,
        gamma_cf=1,
        d_assoc=1,
        thresh_sigma=0.5,
    )

    # Which Parameters to fit
    if sim_name == "1":
        # what_to_fit = ['beta_enc','beta_rec_post','s_fc','gamma_fc']
        what_to_fit = ["beta_enc", "beta_rec_post", "s_fc", "gamma_fc", "c_thresh_itm"]
        # simulation specific boundary
        lb_dict.update(
            c_thresh_itm=0.2,
        )
        ub_dict.update(
            beta_enc=0.4,
            beta_rec_post=0.4,
            s_fc=0.4,
            gamma_fc=0.4,
            c_thresh_itm=0.8,
        )

    elif sim_name == "3":
        what_to_fit = ["beta_enc", "beta_cue", "beta_rec_post", "s_fc", "gamma_fc", "c_thresh_itm", "c_thresh_assoc"]
        # simulation specific boundary
        ub_dict.update(
            beta_enc=0.5,
            beta_cue=0.5,
            beta_rec_post=0.5,
        )

    elif sim_name == "S1":
        # what_to_fit = ['beta_enc', 'beta_rec', 'beta_cue', 'beta_rec_post', 'beta_distract', 'gamma_fc', 'gamma_cf', 's_fc', 's_cf', 'phi_s', 'phi_d', 'kappa', 'lamb', 'eta', 'omega', 'alpha', 'c_thresh', 'c_thresh_itm', 'c_thresh_ass', 'd_ass']
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
        ]
        # simulation specific boundary
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
            c_thresh_assoc=0.2,
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
            c_thresh=0.8,
            c_thresh_itm=0.4,
            c_thresh_assoc=1,
        )

    elif sim_name == "S2":
        # what_to_fit = ['beta_enc', 'beta_rec', 'beta_cue', 'beta_rec_post', 'beta_distract', 'gamma_fc', 's_fc', 'c_thresh_itm', 'c_thresh_ass', 'd_ass', 'thresh_sigma']
        what_to_fit = ["beta_enc", "beta_cue", "beta_rec_post", "beta_distract", "gamma_fc", "s_fc", "c_thresh_itm", "c_thresh_assoc", "thresh_sigma"]
        # simulation specific boundary
        lb_dict.update(
            beta_enc=0,
            beta_cue=0,
            beta_rec_post=0.4,
            beta_distract=0,
            gamma_fc=0,
            s_fc=0.2,
            c_thresh_itm=0.6,
            c_thresh_assoc=0.6,
            thresh_sigma=0,
        )
        ub_dict.update(
            beta_enc=0.8,
            beta_cue=0.5,
            beta_rec_post=1,
            beta_distract=0.6,
            gamma_fc=0.4,
            s_fc=0.8,
            c_thresh_itm=1,
            c_thresh_assoc=1,
            thresh_sigma=0.2,
        )

    elif sim_name == "6b":
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
        # what_to_fit = ['beta_enc', 'beta_rec', 'beta_cue', 'beta_rec_post', 'beta_distract', 'gamma_fc', 'gamma_cf', 's_fc', 's_cf', 'phi_s', 'phi_d', 'kappa', 'lamb', 'eta', 'omega', 'alpha', 'c_thresh', 'd_ass']

    elif sim_name == "4ctrl":
        what_to_fit = ["beta_enc", "beta_cue", "beta_distract", "s_fc", "gamma_fc", "c_thresh_itm"]

    elif sim_name == "4shift":
        what_to_fit = ["beta_enc", "beta_cue", "beta_distract", "s_fc", "gamma_fc", "c_thresh_itm", "c_s"]
        lb_dict.update(
            c_thresh_itm=-1,
            c_s=0,
        )
        ub_dict.update(
            c_thresh_itm=1,
            c_s=10,
        )

    # create lb and ub as list
    lb = [lb_dict[key] for key in what_to_fit]
    ub = [ub_dict[key] for key in what_to_fit]

    return lb, ub, what_to_fit
