import os
import json
import numpy as np
import scipy.io


def make_params(source_coding=False):
    """
    Returns a dictionary containing all parameters that need to be defined in order for CMR to run. Can be used as a template for the "params" input. [Modified]

    :param source_coding: If True, parameter dictionary will contain the parameters required for the source coding version of the model. If False, the dictionary will only condain parameters required for the base version of the model.
    """
    param_dict = {
        # Beta parameters
        "beta_enc": None,  # Beta for encoding (Must set explicitly)
        "beta_rec": None,  # Beta for recall
        "beta_cue": None,  # Beta for cue [CMR-IA]
        "beta_rec_post": 0.,  # Beta for post-recall (Defaults to 0)
        "beta_distract": 0.,  # Beta for distractor task (Defaults to 0)
        "beta_enc_inpair": None,  # Beta for drift within pairs, not used [CMR-IA]

        # Semantic parameters
        "s_fc": 0.,  # Semantic scaling in feature-to-context associations (Defaults to 0)
        "s_cf": 0.,  # Semantic scaling in context-to-feature associations (Defaults to 0)
    
        # Primacy parameters
        "phi_s": 0.,  # Scaling (Defaults to 0)
        "phi_d": 0.,  # Exponential decay rate (Defaults to 0)
        
        # Elevated-attention parameters [CMR-IA]
        "psi_s": 0.,  # Slope with sem_mean (Defaults to 0)
        "psi_c": 1.,  # Intercept (Defaults to 1)
        
        # Other encoding parameters [CMR-IA]
        "d_assoc": None,  # Direct association, not used
        "use_new_context": True,  # Whether to use updated context for learning (Defaults to True, False to be consistent with CMR2)

        # Recognition parameters [CMR-IA]
        "use_flexible_thresh": True,  # Whether to use flexible recognition threshold (Defaults to True)
        "c_d": 0.,  # Exponential rate of weights for flexible threshold (Defaults to 0)
        "thresh_kernel_len": 10,  # How many previous items for flexible threshold (Defaults to 10)
        "c_thresh_itm": 1.,  # Threshold constant for item recognition (Defaults to 1)
        "c_thresh_assoc": 1.,  # Threshold constant for associative recognition (Defaults to 1)
        "c_s": 0.,  # Criteria-shift slope with sem_mean (Defaults to 0)
        "thresh_sigma": 0.,  # Retrieval variability (Defaults to 0)
        "recog_slope": 1.,  # Sigmoid slope for recognition probability, not used (Defaults to 1)
        
        # Recall parameters
        "kappa": None,
        "eta": None,
        "omega": None,
        "alpha": None,
        "lamb": None,
        "c_thresh": None,
        "ban_recall": None,  # List of items that should not be recalled [CMR-IA]
        
        # Timing & recall settings
        "rec_time_limit": 60000.,  # Duration of recall period (in ms) (Defaults to 60000)
        "dt": 10,  # Number of milliseconds to simulate in each loop of the accumulator (Defaults to 10)
        "nitems_in_accumulator": 50,  # Number of items in accumulator (Defaults to 50)
        "max_recalls": 50,  # Maximum recalls allowed per trial (Defaults to 50)
        "learn_while_retrieving": False,  # Whether associations should be learned during recall (Defaults to False)
    }

    # If not using source coding, set up 2 associative scaling parameters (gamma)
    if not source_coding:
        param_dict["gamma_fc"] = None  # Gamma FC
        param_dict["gamma_cf"] = None  # Gamma CF

    # If using source coding, add an extra beta parameter and set up 8 associative scaling parameters
    else:
        param_dict["beta_source"] = None  # Beta source

        param_dict["L_FC_tftc"] = None  # Scale of items reinstating past temporal contexts (Recommend setting to gamma FC)
        param_dict["L_FC_sftc"] = 0  # Scale of sources reinstating past temporal contexts (Defaults to 0)
        param_dict["L_FC_tfsc"] = None  # Scale of items reinstating past source contexts (Recommend setting to gamma FC)
        param_dict["L_FC_sfsc"] = 0  # Scale of sources reinstating past source contexts (Defaults to 0)

        param_dict["L_CF_tctf"] = None  # Scale of temporal context cueing past items (Recommend setting to gamma CF)
        param_dict["L_CF_sctf"] = None  # Scale of source context cueing past items (Recommend setting to gamma CF or fitting as gamma source)
        param_dict["L_CF_tcsf"] = 0  # Scale of temporal context cueing past sources (Defaults to 0, since model does not recall sources)
        param_dict["L_CF_scsf"] = 0  # Scale of source context cueing past sources (Defaults to 0, since model does not recall sources)

    return param_dict


def load_params(simu_name=None, params_path=None, fixed_params=None):
    """Load fitted parameters for a simulation. fixed_params is an optional dict of overrides applied after loading.

    At least one of simu_name or params_path must be provided.
    If simu_name is given, the JSON keys are validated against what_to_fit for that simulation.
    If params_path is None, defaults to data/simu{simu_name}_params.json relative to the caller's cwd.
    """
    if simu_name is None and params_path is None:
        raise ValueError("At least one of simu_name or params_path must be provided.")

    # Read fitted params
    if params_path is None:
        params_path = f"data/simu{simu_name}_params.json"
    with open(params_path, "r") as f:
        fitted_params = json.load(f)

    # Check what to fit
    if simu_name is not None:
        from CMR_IA.fitting import make_boundary
        _, _, what_to_fit = make_boundary(simu_name)
        json_keys = set(fitted_params.keys())
        expected_keys = set(what_to_fit)
        if json_keys != expected_keys:
            extra = json_keys - expected_keys
            missing = expected_keys - json_keys
            msg = f"JSON keys do not match what_to_fit for simu '{simu_name}'."
            if extra:
                msg += f" Extra keys: {extra}."
            if missing:
                msg += f" Missing keys: {missing}."
            raise ValueError(msg)

    # Load into full params
    params = make_params()
    params.update(fitted_params)

    # Apply fixed params
    if fixed_params is not None:
        params.update(fixed_params)

    return params


def load_pres(path):
    """
    Loads matrix of presented items from a .txt file, a .json behavioral data, file, or a .mat behavioral data file. Uses numpy's loadtxt function, json's load function, or scipy's loadmat function, respectively. [Unchanged from CMR2]

    :param path: The path to a .txt, .json, or .mat file containing a matrix where item (i, j) is the jth word presented on trial i.

    :returns: A 2D array of presented items.
    """
    if os.path.splitext(path) == ".txt":
        data = np.loadtxt(path)
    elif os.path.splitext(path) == ".json":
        with open(path, "r") as f:
            data = json.load(f)
            data = data["pres_nos"] if "pres_nos" in data else data["pres_itemnos"]
    elif os.path.splitext(path) == ".mat":
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["data"].pres_itemnos
    else:
        raise ValueError("Can only load presented items from .txt, .json, and .mat formats.")
    return np.atleast_2d(data)


def split_data(pres_mat, identifiers, source_mat=None):
    """
    If data from multiple subjects or sessions are in one matrix, separate out the data into separate presentation and source matrices for each unique identifier. [Unchanged from CMR2]

    :param pres_mat: A 2D array of presented items from multiple consolidated subjects or sessions.
    :param identifiers: A 1D array with length equal to the number of rows in pres_mat, where entry i identifies the subject/session/etc. to which row i of the presentation matrix belongs.
    :param source_mat: (Optional) A trials x serial positions x nsources array of source information for each presented item in pres_mat.

    :returns: A list of presented item matrices (one matrix per unique identifier), an array of the unique identifiers, and a list of source information matrices (one matrix per subject, None if no source_mat provided).
    """
    pres_mat = np.array(pres_mat)
    if source_mat is not None:
        source_mat = np.atleast_3d(source_mat)

    unique_ids = np.unique(identifiers)

    data = []
    sources = None if source_mat is None else []
    for i in unique_ids:
        mask = identifiers == i
        data.append(pres_mat[mask, :])
        if source_mat is not None:
            sources.append(source_mat[mask, :, :])

    return data, unique_ids, sources


def param_vec_to_dict(param_vec, simu_name):
    from CMR_IA.fitting import make_boundary
    param_dict = make_params()
    _, _, what_to_fit = make_boundary(simu_name)
    for name, value in zip(what_to_fit, param_vec):
        param_dict[name] = value
    return param_dict
