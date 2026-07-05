from __future__ import print_function
import math
import time
import numpy as np
from tqdm import tqdm
from libc.math cimport log, sqrt
from collections import deque
cimport numpy as np
cimport cython
from CMR_IA.utils import split_data


# ---------- Cython Functions ---------- #

# Portable xorshift64 RNG — fixed algorithm, identical output on every platform/OS [CMR-IA]
cdef unsigned long long _xorshift_state = 1

cdef inline void seed_xorshift(unsigned long long s):
    global _xorshift_state
    _xorshift_state = s if s != 0 else 1  # state must never be 0

cdef inline unsigned long long xorshift64():
    global _xorshift_state
    _xorshift_state ^= _xorshift_state << 13
    _xorshift_state ^= _xorshift_state >> 7
    _xorshift_state ^= _xorshift_state << 17
    return _xorshift_state

cdef double random_uniform():
    return (xorshift64() >> 11) * (1.0 / 9007199254740992.0)


# Credit to "senderle" for the cython random number generation functions used below. Original code can be found at:
# https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers
@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void assign_random_gaussian_pair(double[:] out, int assign_ix):
    cdef double x1, x2, w
    w = 2.0
    while (w >= 1.0):
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = sqrt((-2.0 * log(w)) / w)
    out[assign_ix] = x1 * w
    out[assign_ix + 1] = x2 * w  # fixed! [CMR-IA]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cython_randn(int n):
    cdef int i
    np_result = np.zeros(n, dtype="f8", order="C")
    cdef double[:] result = np_result
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()

    return result


# ---------- Main Model ---------- #

class CMR(object):

    def __init__(self, params, pres_mat, sem_mat,
                 source_mat=None, rec_mat=None, ffr_mat=None, cue_mat=None,
                 mode="IFR", design=None, seed=12345):
        """
        Initializes a CMR object and prepares it to simulate the session defined by pres_mat.

        :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
        :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored. If presenting word pairs, use a 3D array with the length of the third dimension equals to 2.
        :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
        :param source_mat: 3D array of source features for each presented word. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i. If None, no source features are used.
        :param rec_mat: 2D array of ID numbers of words recalled by real subjects in a free recall phase on each trial. Rows correspond to pres_mat.
        :param ffr_mat: 1D array of ID numbers of words recalled by real subjects in a final free recall phase.
        :param cue_mat: 1D array of ID numbers of words presented to the model in recognition and cued recall. If presenting word pairs, use a 2D array with the length of the second dimension equals to 2.
        :param mode: String indicating the task and mode to simulate.
            - Free recall: "IFR" (immediate), "DFR" (delayed).
            - Cued recall: "CRNormal" (cued recall in final stage).
            - Recognition: "RecogNormal" (recognition in final stage), "RecogContinuous" (continuous recognition).
            - Successive tests: "Recog-Recog", "Recog-CR", "CR-Recog", "CR-CR".
        :param design: String indicating the experimental design variant. "Hockley" for Hockley's variation of continuous recognition (mode="RecogContinuous"), "Osth" for Osth's associative recognition paradigm (mode="RecogNormal"). None uses the standard paradigm.
        [CMR2->CMR-IA]
        """
        ##########
        #
        # Set up model parameters and presentation data
        #
        ##########

        # Dictionary of model parameters
        self.params = params

        # Presented item ID numbers (trial x serial position)
        self.pres_nos = np.array(pres_mat, dtype=np.int16)

        # Semantic similarity matrix (e.g. Word2vec, LSA, WAS)
        self.sem_mat = np.array(sem_mat, dtype=np.float32)

        # Input cue mat [CMR-IA]
        if cue_mat is None:
            self.have_cue = False
        else:
            self.have_cue = True
            self.cues_nos = np.array(cue_mat, dtype=np.int16)
        
        # Input recall mat [CMR-IA]
        if rec_mat is None:
            self.have_rec = False
        else:
            self.have_rec = True
            self.rec_nos = np.array(rec_mat, dtype=np.int16)
        
        # Input final free recall mat [CMR-IA]
        if ffr_mat is None:
            self.have_ffr = False
        else:
            self.have_ffr = True
            self.ffr_nos = np.array(ffr_mat, dtype=np.int16)
        
        # Input source
        if source_mat is None:
            self.nsources = 0
        else:
            self.sources = np.atleast_3d(source_mat).astype(np.float32)
            self.nsources = self.sources.shape[2]
            if self.sources.shape[0:2] != self.pres_nos.shape[0:2]:
                raise ValueError("Source matrix must have the same number of rows and columns as the presented item matrix.")
        
        # Input mode [CMR-IA]
        if mode not in ("IFR", "DFR", "CRNormal", "RecogNormal", "RecogContinuous", "Recog-Recog", "Recog-CR", "CR-Recog", "CR-CR"):
            raise ValueError("Mode %s is invalid." % mode)
        if mode not in ("IFR", "DFR") and cue_mat is None:
            raise ValueError("Mode %s requires a cue matrix." % mode)
        self.mode = mode

        # Input design [CMR-IA]
        if design is not None and design not in ("EXP1", "Hockley", "Osth", "S1G3"):
            raise ValueError("Design %s is invalid." % design)
        _design_mode_map = {"EXP1": "RecogContinuous", "Hockley": "RecogContinuous", "S1G3": "Recog-CR", "Osth": "RecogNormal"}
        if design is not None and mode != _design_mode_map[design]:
            raise ValueError("Design %s requires mode='%s', got '%s'." % (design, _design_mode_map[design], mode))
        self.design = design

        # Validate parameters
        self._validate_parameters()

        # Determine the number of lists and the maximum list length (how many words or word-pairs)
        self.nlists = self.pres_nos.shape[0]
        self.max_list_length = self.pres_nos.shape[1]

        # Initialize the number of extra distractor [CMR-IA]
        self.extra_distract = 0

        # Create arrays of sorted and unique (nonzero) items [CMR-IA]
        self.pres_nonzero_mask = self.pres_nos > 0
        self.pres_nos_nonzero = self.pres_nos[self.pres_nonzero_mask]
        self.all_nos = self.pres_nos_nonzero
        if self.have_rec:
            self.rec_nonzero_mask = self.rec_nos > 0
            self.rec_nos_nonzero = self.rec_nos[self.rec_nonzero_mask]
            self.all_nos = np.concatenate((self.all_nos, self.rec_nos_nonzero), axis=None)
        if self.have_ffr:
            self.ffr_nonzero_mask = self.ffr_nos > 0
            self.ffr_nos_nonzero = self.ffr_nos[self.ffr_nonzero_mask]
            self.all_nos = np.concatenate((self.all_nos, self.ffr_nos_nonzero), axis=None)
            self.extra_distract += 1
        if self.have_cue:
            self.cues_nonzero_mask = self.cues_nos > 0
            self.cues_nos_nonzero = self.cues_nos[self.cues_nonzero_mask]
            self.all_nos = np.concatenate((self.all_nos,self.cues_nos_nonzero), axis=None)
        self.all_nos_sorted = np.sort(self.all_nos)
        self.all_nos_unique = np.unique(self.all_nos_sorted)  # 1D, order in feature vector

        # Convert presented item and cue item ID numbers to indexes within the feature vector [CMR-IA]
        indexer = lambda x: np.searchsorted(self.all_nos_unique, x) if x > 0 else x
        indexer_func = np.vectorize(indexer)
        self.pres_indexes = indexer_func(self.pres_nos)
        if self.have_rec:
            self.rec_indexes = indexer_func(self.rec_nos)
        if self.have_ffr:
            self.ffr_indexes = indexer_func(self.ffr_nos)
        if self.have_cue:
            self.cues_indexes = indexer_func(self.cues_nos)
        
        # Make sure items" associations with themselves are set to 0
        np.fill_diagonal(self.sem_mat, 0)

        # Average semantic association with other items (for attention and criteria shift) [CMR-IA]
        self.sem_mean = np.sum(self.sem_mat, axis=1) / (np.shape(self.sem_mat)[1] - 1)
        
        # Cut down semantic matrix to contain only the items in the session
        self.sem_mat = self.sem_mat[self.all_nos_unique - 1, :][:, self.all_nos_unique - 1]

        # Initialize phase
        self.phase = None

        # Learn while retrieving
        self.learn_while_retrieving = self.params["learn_while_retrieving"] if "learn_while_retrieving" in self.params else False

        ##########
        #
        # Set up context and feature vectors
        #
        ##########

        # Determine number of cells in each region of the feature/context vectors [CMR-IA]
        self.nitems_unique = len(self.all_nos_unique) 
        if self.mode in ("RecogNormal", "CRNormal"):  # for norm recog and norm cr
            self.extra_distract += self.nlists
        if self.mode in ("Recog-Recog", "Recog-CR", "CR-Recog", "CR-CR"):  # for successive tests
            self.extra_distract += 2*self.nlists
        if self.mode == "DFR":  # one extra distractor before each recall period if running DFR
            self.extra_distract += self.nlists
        self.ndistractors = self.nlists + self.extra_distract
        self.ntemporal = self.nitems_unique + self.ndistractors
        self.nelements = self.ntemporal + self.nsources

        # Create context and feature vectors
        self.f = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c = np.zeros_like(self.f)
        self.c_old = np.zeros_like(self.f)
        self.c_in = np.zeros_like(self.f)

        ##########
        #
        # Set up weight matrices
        #
        ##########

        # Set up primacy scaling vector
        self.prim_vec = self.params["phi_s"] * np.exp(-1 * self.params["phi_d"] * np.arange(self.max_list_length)) + 1

        # Set up learning rate matrix for M_FC (dimensions are context x features)
        self.L_FC = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            # If no source, uniformly gamma_fc
            self.L_FC.fill(self.params["gamma_fc"])
        else:
            # Temporal Context x Item Features (items reinstating their previous temporal contexts)
            self.L_FC[:self.ntemporal, :self.ntemporal] = self.params["L_FC_tftc"]
            # Temporal Context x Source Features (sources reinstating previous temporal contexts)
            self.L_FC[:self.ntemporal, self.ntemporal:] = self.params["L_FC_sftc"]
            # Source Context x Item Features (items reinstating previous source contexts)
            self.L_FC[self.ntemporal:, :self.ntemporal] = self.params["L_FC_tfsc"]
            # Source Context x Source Features (sources reinstating previous source contexts)
            self.L_FC[self.ntemporal:, self.ntemporal:] = self.params["L_FC_sfsc"]

        # Set up learning rate matrix for M_CF (dimensions are features x context)
        self.L_CF = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            # If no source, uniformly gamma_cf
            self.L_CF.fill(self.params["gamma_cf"])
        else:
            # Item Features x Temporal Context (temporal context cueing retrieval of items)
            self.L_CF[:self.ntemporal, :self.ntemporal] = self.params["L_CF_tctf"]
            # Item Features x Source Context (source context cueing retrieval of items)
            self.L_CF[:self.ntemporal, self.ntemporal:] = self.params["L_CF_sctf"]
            # Source Features x Temporal Context (temporal context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, :self.ntemporal] = self.params["L_CF_tcsf"]
            # Source Features x Source Context (source context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, self.ntemporal:] = self.params["L_CF_scsf"]

        # Initialize weight matrices as identity matrices
        self.M_FC = np.identity(self.nelements, dtype=np.float32)
        self.M_CF = np.identity(self.nelements, dtype=np.float32)

        # Scale the semantic similarity matrix by s_fc (Healey et al., 2016) and s_cf (Lohnas et al., 2015)
        fc_sem_mat = self.params["s_fc"] * self.sem_mat
        cf_sem_mat = self.params["s_cf"] * self.sem_mat

        # Complete the pre-experimental associative matrices by layering on the scaled semantic matrices
        self.M_FC[:self.nitems_unique, :self.nitems_unique] += fc_sem_mat
        self.M_CF[:self.nitems_unique, :self.nitems_unique] += cf_sem_mat

        # Scale pre-experimental associative matrices by 1 - gamma
        self.M_FC *= 1 - self.L_FC
        self.M_CF *= 1 - self.L_CF

        #####
        #
        # Initialize leaky accumulator and recall variables
        #
        #####

        # Retrieval thresholds
        self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)

        # Items that should not be recalled (necessary for simu8) [CMR-IA]
        if self.params["ban_recall"] is not None:
            ban_itemnos = np.atleast_1d(self.params["ban_recall"])
            self.ban_recall_idx = np.nonzero(np.isin(self.all_nos_unique, ban_itemnos))[0]
            self.ret_thresh[self.ban_recall_idx] = np.inf
        else:
            self.ban_recall_idx = None

        # Number of items in accumulator
        self.nitems_in_race = self.params["nitems_in_accumulator"]

        # Recalled items from each trial
        self.rec_items = []

        # Rectimes of recalled items from each trial
        self.rec_times = []

        # Calculate dt_tau and its square root based on dt
        self.params["dt_tau"] = self.params["dt"] / 1000.
        self.params["sq_dt_tau"] = np.sqrt(self.params["dt_tau"])

        ##########
        #
        # Initialize variables for tracking simulation progress
        #
        ##########

        # Current trial number (0-indexed)
        self.trial_idx = 0

        # Current serial position (0-indexed)
        self.serial_position = 0

        # Current distractor index
        self.distractor_idx = self.nitems_unique

        # Index of the first source feature
        self.first_source_idx = self.ntemporal

        # Set up random seed for reproducibility [CMR-IA]
        seed_xorshift(seed)
        self.rng = np.random.default_rng(seed)

        # Intermediate measures for debugging [CMR-IA]
        self.recog_csims = []
        self.recog_threshs = []
        self.recog_probs = []
        self.f_in_acc = []
        self.f_in_dif = []

        ##########
        #
        # Set up elevated-attention, criteria-shift, retrieval variability [CMR-IA]
        #
        ##########
        
        # Set up elevated-attention scaling vector for all itemno
        self.att_vec = self.params["psi_s"] * self.sem_mean + self.params["psi_c"]
        if self.nsources == 0:
            att_ceil = 1 / self.params["gamma_fc"]
        else:
            att_ceil = 1 / self.params["L_FC_tftc"]
        self.att_vec[self.att_vec > att_ceil] = att_ceil
        self.att_vec[self.att_vec < 0] = 0

        # Set up c_thresh vector for all itemno, allowing criterion shifting for different items
        self.c_vec = self.params["c_s"] * self.sem_mean + self.params["c_thresh_itm"]

        # Set up random mechanism for threshold
        self.thresh_rng = np.random.default_rng(87)
        self.thresh_sigma = self.params["thresh_sigma"]

        # Flexible threshold kernel
        self.thresh_kernel = np.exp(self.params["c_d"] * np.arange(self.params["thresh_kernel_len"]))
        self.thresh_kernel /= np.sum(self.thresh_kernel)
        self.init_csims_flag = False


    def _validate_parameters(self):
        """
        Validate required parameters.
        [CMR-IA]
        """
        # Always required
        for key in ["beta_enc", "beta_rec_post", "beta_distract", "s_fc", "s_cf", "phi_s", "phi_d", "psi_s", "psi_c", "c_s", "thresh_sigma", "c_d", "thresh_kernel_len"]:
            assert self.params[key] is not None, "params['%s'] must be set." % key

        # Source specific / gamma
        if self.nsources == 0:
            for key in ["gamma_fc", "gamma_cf"]:
                assert self.params[key] is not None, "params['%s'] must be set when not using source coding." % key
        else:
            for key in ["beta_source", "L_FC_tftc", "L_FC_tfsc", "L_CF_tctf", "L_CF_sctf"]:
                assert self.params[key] is not None, "params['%s'] must be set when using source coding." % key
        
        # Recall specific
        if self.mode in ("IFR", "DFR") or "CR" in self.mode:
            for key in ["beta_rec", "kappa", "eta", "omega", "alpha", "lamb", "c_thresh"]:
                assert self.params[key] is not None, "params['%s'] must be set for recall tasks." % key
            if "CR" in self.mode:
                for key in ["beta_cue"]:
                    assert self.params[key] is not None, "params['%s'] must be set for cued recall." % key

        # Recognition specific
        if "Recog" in self.mode:
            for key in ["beta_cue", "c_thresh_itm", "c_thresh_assoc"]:
                assert self.params[key] is not None, "params['%s'] must be set for recognition tasks." % key


    def present_item(self, item_idx, source=None, update_context=True, update_weights=True, use_new_context=False):
        """
        Presents a single item (or distractor) to the model by updating the feature vector. Options are provided to update context and the model's associative matrices after presentation.

        :param item_idx: Index of the cell within the feature vector to be activated by the presented item.
        :param source: If None, no source features are activated. If a 1D array, the source features in the feature vector are set to match the numbers in the source array.
        :param update_context: If True, the context vector updates after the feature vector is updated.
        :param update_weights: If True, the model's weight matrices update to strengthen the association between the presented item and the context state.
        :param use_new_context: If True, use the updated context vector to update the weight matrices (used in the paper). If False, use the old context vector (used in CMR2).
        [CMR2->CMR-IA]
        """
        ##########
        #
        # Activate item's features
        #
        ##########

        assert np.all(item_idx >= 0), "Item index must be greater than or equal to 0."
        paired_pres = np.logical_not(np.isscalar(item_idx))

        # Activate the presented item itself
        self.f.fill(0)
        self.f[item_idx] = 1

        # Activate the source feature(s) of the presented item
        if self.nsources > 0 and source is not None:
            self.f[self.first_source_idx:, 0] = np.atleast_1d(source)

        # Copy c_old [CMR-IA]
        self.c_old = self.c.copy()

        # Compute c_in
        self.c_in = np.dot(self.M_FC, self.f)

        # Normalize the temporal and source subregions of c_in separately
        norm_t = np.sqrt(np.sum(self.c_in[:self.ntemporal] ** 2))
        if norm_t != 0:
            self.c_in[:self.ntemporal] /= norm_t
        if self.nsources > 0:
            norm_s = np.sqrt(np.sum(self.c_in[self.ntemporal:] ** 2))
            if norm_s != 0:
                self.c_in[self.ntemporal:] /= norm_s

        ##########
        #
        # Update context
        #
        ##########

        if update_context:

            # Set beta separately for temporal and source subregions
            beta_vec = np.empty_like(self.c)
            beta_vec[:self.ntemporal] = self.beta
            beta_vec[self.ntemporal:] = self.beta_source

            # Calculate rho for the temporal and source subregions
            rho_vec = np.empty_like(self.c)
            c_dot_t = np.dot(self.c[:self.ntemporal].T, self.c_in[:self.ntemporal]).item()
            rho_vec[:self.ntemporal] = math.sqrt(1 + self.beta ** 2 * (c_dot_t ** 2 - 1)) - self.beta * c_dot_t
            c_dot_s = np.dot(self.c[self.ntemporal:].T, self.c_in[self.ntemporal:]).item()
            rho_vec[self.ntemporal:] = math.sqrt(1 + self.beta_source ** 2 * (c_dot_s ** 2 - 1)) - self.beta_source * c_dot_s

            # Update context
            self.c = (rho_vec * self.c_old) + (beta_vec * self.c_in)

        ##########
        #
        # Update weight matrices [CMR-IA]
        #
        ##########

        if update_weights:

            # Use updated c as in CMR-IA
            if use_new_context:
                if self.phase == "encoding":  # only apply elevated-attention and primacy during encoding
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c[:self.nitems_unique], self.f[:self.nitems_unique].T) \
                        * np.mean(self.att_vec[self.all_nos_unique[item_idx] - 1])  # mean for pair presentation (not used)
                    self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.f[:self.nitems_unique], self.c[:self.nitems_unique].T) \
                        * self.prim_vec[self.serial_position]
                else:
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c[:self.nitems_unique], self.f[:self.nitems_unique].T)
                    self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.f[:self.nitems_unique], self.c[:self.nitems_unique].T)
            
            # Use c_old as in CMR2
            else:
                if self.phase == "encoding":
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c_old[:self.nitems_unique], self.f[:self.nitems_unique].T) \
                        * np.mean(self.att_vec[self.all_nos_unique[item_idx] - 1])
                    self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.f[:self.nitems_unique], self.c_old[:self.nitems_unique].T) \
                        * self.prim_vec[self.serial_position]
                else:
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c_old[:self.nitems_unique], self.f[:self.nitems_unique].T)
                    self.M_CF[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_CF[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.f[:self.nitems_unique], self.c_old[:self.nitems_unique].T)
            
            # Direct association between pairs (not used)
            if paired_pres and self.params["d_assoc"] is not None:
                pair_ass = self.params["d_assoc"] * np.dot(self.f, self.f.T)
                np.fill_diagonal(pair_ass, 0)
                self.M_FC += self.L_FC * pair_ass
                self.M_CF += self.L_CF * self.prim_vec[self.serial_position] * pair_ass


    def _record_presented_items(self, item_idx, as_list=True):
        """
        Track presented items for flexible recognition threshold.
        [CMR-IA]
        """
        if not hasattr(self, 'presented_items'):
            self.presented_items = []
        if not hasattr(self, 'list_items'):
            self.list_items = []
        is_paired_idx = np.logical_not(np.isscalar(item_idx))

        # Record a pair
        if is_paired_idx:
            if not hasattr(self, 'presented_pairs'):
                self.presented_pairs = []
            if not hasattr(self, 'list_pairs'):
                self.list_pairs = []
            self.presented_items.append(item_idx[0]) if item_idx[0] not in self.presented_items else None
            self.presented_items.append(item_idx[1]) if item_idx[1] not in self.presented_items else None
            self.presented_pairs.append(tuple(item_idx)) if tuple(item_idx) not in self.presented_pairs else None
            if as_list:
                self.list_items.append(item_idx[0]) if item_idx[0] not in self.list_items else None
                self.list_items.append(item_idx[1]) if item_idx[1] not in self.list_items else None
                self.list_pairs.append(tuple(item_idx)) if tuple(item_idx) not in self.list_pairs else None
        
        # Record an item
        else:
            self.presented_items.append(item_idx) if item_idx not in self.presented_items else None
            if as_list:
                self.list_items.append(item_idx) if item_idx not in self.list_items else None


    def _init_recent_csims(self, num=5, do_item=False, do_pair=False):
        """
        Initialize some recent csims for flexible recognition threshold.
        [CMR-IA]
        """
        if do_item:
            try:
                # Get the csim for some random old items
                old_item_csims = []
                old_items = self.rng.choice(np.array(self.list_items), num)
                for pres_idx in old_items:
                    self.present_item(pres_idx, source=None, update_context=False, update_weights=False)  # just to get self.c_in
                    csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                    old_item_csims.append(csim.item())

                # Get the csim for some random new items
                new_item_csims = []
                all_new_items = [idx for idx in np.arange(self.nitems_unique) if idx not in np.array(self.presented_items)]
                new_items = self.rng.choice(all_new_items, num)
                for pres_idx in new_items:
                    self.present_item(pres_idx, source=None, update_context=False, update_weights=False)  # just to get self.c_in
                    csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                    new_item_csims.append(csim.item())
                
                # Fill the recent csims with the average csim
                csim_avg = np.mean(old_item_csims + new_item_csims)
                self.recent_item_csims = deque([csim_avg] * self.params["thresh_kernel_len"], maxlen=self.params["thresh_kernel_len"])
            except:
                pass  # usually because of no new items when no item recognition

        if do_pair:
            try:
                self.beta = self.params["beta_cue"]
                c_old_tmp = self.c_old.copy()
                c_tmp = self.c.copy()

                # Get the csim for old pairs
                old_pair_csims = []
                old_pairs = self.rng.choice(np.array(self.list_pairs), num)
                for pres_idx in old_pairs:
                    self.present_item(pres_idx[0], source=None, update_context=True, update_weights=False)
                    self.present_item(pres_idx[1], source=None, update_context=False, update_weights=False)
                    csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                    self.c_old = c_old_tmp
                    self.c = c_tmp
                    old_pair_csims.append(csim.item())

                # Rearranged pairs are a bit tricky, the expected csim may differ with experimental settings
                def get_rearranged_pairs(presented_pairs, num):
                    rearranged_pairs = []

                    if self.design == "Hockley":  # for Hockley's continuous experiment
                        for i in range(0, len(presented_pairs) - 1):
                            rearranged_pairs.append([presented_pairs[i][0], presented_pairs[i + 1][1]])
                            rearranged_pairs.append([presented_pairs[i][1], presented_pairs[i + 1][0]])
                        rearranged_pairs = self.rng.choice(rearranged_pairs, num)

                    elif self.design == "Osth":  # for Osth's associative recognition experiment
                        for _ in range(num):
                            lag = self.rng.choice(np.arange(1, 6))
                            fw = self.rng.choice([0, 1])
                            idx = self.rng.choice(np.arange(len(presented_pairs) - lag))
                            rearranged_pairs.append([presented_pairs[idx][fw], presented_pairs[idx + lag][1 - fw]])

                    elif self.design == "S1G3":  # for S1 where rearranged pairs are totally random
                        for _ in range(num):
                            idx1, idx2 = self.rng.choice(np.arange(len(presented_pairs)), 2, replace=False)
                            fw = self.rng.choice([0, 1])
                            rearranged_pairs.append([presented_pairs[idx1][fw], presented_pairs[idx2][1 - fw]])

                    else:  # for other experiments where new pairs are totally random
                        all_new_items = [idx for idx in np.arange(self.nitems_unique) if idx not in np.array(self.presented_items)]
                        for _ in range(num):
                            new_pair = self.rng.choice(all_new_items, 2, replace=False)
                            rearranged_pairs.append(new_pair)

                    return rearranged_pairs
                    
                # Get the csim for rearranged new pairs
                new_pair_csims = []
                new_pairs = get_rearranged_pairs(self.list_pairs, num)
                for pres_idx in new_pairs:
                    self.present_item(pres_idx[0], source=None, update_context=True, update_weights=False)
                    self.present_item(pres_idx[1], source=None, update_context=False, update_weights=False)
                    csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                    self.c_old = c_old_tmp
                    self.c = c_tmp
                    new_pair_csims.append(csim.item())

                # Fill the recent csims with the average csim
                csim_avg = np.mean(old_pair_csims + new_pair_csims)
                self.recent_pair_csims = deque([csim_avg] * self.params["thresh_kernel_len"], maxlen=self.params["thresh_kernel_len"])
            except:
                pass  # usually because of no new pairs when no pair recognition

    
    def simulate_recall(self, time_limit=60000, max_recalls=np.inf):
        """
        Simulates a recall period starting from the current state of context.

        :param time_limit: Simulated duration of the recall period in milliseconds. Determines the number of cycles of the leaky accumulator before the recall period ends.
        :param max_recalls: Maximum number of retrievals (not overt recalls) the model is allowed to make. If this limit is reached, the recall period ends early. This setting prevents the model from consuming excessive runtime if its parameters cause it to make numerous recalls per trial.
        [CMR2]
        """
        cycles_elapsed = 0
        nrecalls = 0
        max_cycles = time_limit // self.params["dt"]

        while cycles_elapsed < max_cycles and nrecalls < max_recalls:
            # Use context to cue items
            f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()

            # Identify set of items with the highest activation
            top_items = np.argsort(f_in)[self.nitems_unique-self.nitems_in_race:]
            top_activation = f_in[top_items]
            top_activation[top_activation < 0] = 0

            # Run accumulator until an item is retrieved
            winner_idx, ncycles = self.leaky_accumulator(top_activation, self.ret_thresh[top_items], int(max_cycles - cycles_elapsed))
            # Update elapsed time
            cycles_elapsed += ncycles
            nrecalls += 1

            # Perform the following steps only if an item was retrieved
            if winner_idx != -1:

                # Identify the feature index of the retrieved item
                item = top_items[winner_idx]

                # Decay retrieval thresholds, then set the retrieved item's threshold to maximum
                self.ret_thresh = 1 + self.params["alpha"] * (self.ret_thresh - 1)
                self.ret_thresh[item] = 1 + self.params["omega"]

                # Present retrieved item to the model, with no source information
                if self.learn_while_retrieving:
                    self.present_item(item, source=None, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])  # [CMR-IA]
                else:
                    self.present_item(item, source=None, update_context=True, update_weights=False)

                # Filter intrusions using temporal context comparison, and log item if overtly recalled
                csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                if csim >= self.params["c_thresh"]:
                    rec_itemno = self.all_nos_unique[item] # [CMR-IA]
                    self.rec_items[-1].append(rec_itemno)
                    self.rec_times[-1].append(cycles_elapsed * self.params["dt"])


    def simulate_recog(self, cue_idx):
        """
        Simulate a recognition period.

        :param cue_idx: The index of the provided cue in the feature vector.
        [CMR-IA]
        """
        # Check if the cue is negative, if so, just skip and return
        if np.any(cue_idx < 0):
            self.rec_items.append(-1)
            self.rec_times.append(-1)
            self.recog_csims.append(-1)
            self.recog_threshs.append(-1)
            self.recog_probs.append(-1)
            return

        # Present cue and update the context
        is_paired_cue = np.logical_not(np.isscalar(cue_idx))
        if self.mode == "RecogContinuous" and self.design == "EXP1":  # special for EXP1, as we just encode the same item
            if is_paired_cue:
                raise NotImplementedError
            else:
                self.present_item(cue_idx, source=None, update_context=False, update_weights=False)
        else:
            if is_paired_cue:
                self.present_item(cue_idx[0], source=None, update_context=True, update_weights=False)
                self.present_item(cue_idx[1], source=None, update_context=True, update_weights=False)
            else:
                self.present_item(cue_idx, source=None, update_context=True, update_weights=False)

        # Calculate context similarity
        csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal]).item()
        self.recog_csims.append(csim)

        # Get recognition threshold
        thresh_epsilon = self.thresh_rng.uniform(-self.thresh_sigma, self.thresh_sigma)
        if self.params["use_flexible_thresh"]:
            if is_paired_cue:
                thresh = np.dot(self.recent_pair_csims, self.thresh_kernel) * self.params["c_thresh_assoc"] + thresh_epsilon if self.init_csims_flag else 1
            else:
                thresh = np.dot(self.recent_item_csims, self.thresh_kernel) * self.c_vec[self.all_nos_unique[cue_idx] - 1] + thresh_epsilon if self.init_csims_flag else 1
        else:
            if is_paired_cue:
                thresh = self.params["c_thresh_assoc"] + thresh_epsilon
            else:
                thresh = self.c_vec[self.all_nos_unique[cue_idx] - 1] + thresh_epsilon
        self.recog_threshs.append(thresh)
        
        # Update recent context similarities
        if self.init_csims_flag:
            if is_paired_cue:
                self.recent_pair_csims.append(csim)
            else:
                self.recent_item_csims.append(csim)

        # Get response
        if csim > thresh:  # OLD response
            self.rec_items.append(1)
            # Output encoding for judged-as-old items or pairs
            if self.learn_while_retrieving:
                self.present_item(cue_idx, source=None, update_context=False, update_weights=True, use_new_context=self.params["use_new_context"])
        else:  # NEW response
            self.rec_items.append(0)  
        self.rec_times.append(0)  # RT not developed yet
        
        # Calculate recognition probability (not used)
        self.recog_probs.append(1 / (1 + np.exp(-self.params["recog_slope"] * (csim - thresh))))


    def simulate_cr(self, cue_idx, time_limit=5000):
        """
        Simulate a cued recall period.

        :param cue_idx: The index of the provided cue in the feature vector.
        :param time_limit: The simulated duration of the recall period (in ms). Determines how many cycles of the leaky accumulator will run before the recall period ends.
        [CMR-IA]
        """
        cycles_elapsed = 0
        max_cycles = time_limit // self.params["dt"]

        # Present cue and update the context
        self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
        if not np.isinf(self.ret_thresh[cue_idx]):
            self.ret_thresh[cue_idx] = 1 + self.params["omega"]  # can't recall the cue!

        # Use context to cue items
        f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()
        self.f_in_acc.append(f_in)  # for testing
        self.f_in_dif.append(f_in - self.ret_thresh)  # for testing, distance to threshold

        # Identify set of items with the highest activation
        top_items = np.argsort(f_in)[self.nitems_unique - self.nitems_in_race:]  # returns the original index of the sorted order
        if self.ban_recall_idx is not None:
            top_items = [x for x in top_items if x not in self.ban_recall_idx]
        top_activation = f_in[top_items]
        top_activation[top_activation < 0] = 0

        # Run accumulator until an item is retrieved, winnder_idx is the index with in top_activation
        winner_idx, ncycles = self.leaky_accumulator(top_activation, self.ret_thresh[top_items], int(max_cycles))
        cycles_elapsed += ncycles

        # Perform the following steps only if an item was retrieved
        if winner_idx != -1:

            # Identify the feature index of the retrieved item
            item = top_items[winner_idx]

            # Decay retrieval thresholds            
            self.ret_thresh = 1 + self.params["alpha"] * (self.ret_thresh - 1)

            # Present retrieved item to the model, with no source information
            self.beta = self.params["beta_rec"]
            self.present_item(item, source=None, update_context=True, update_weights=False)

            # Filter intrusions using temporal context comparison, and log item if overtly recalled
            csim = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
            self.recog_csims.append(csim.item())
            self.recog_threshs.append(self.params["c_thresh"])
            if csim >= self.params["c_thresh"]:

                # Set the retrieved item's threshold to maximum
                if not np.isinf(self.ret_thresh[item]):
                    self.ret_thresh[item] = 1 + self.params["omega"]

                # Output encoding for the pair of cue and recalled item
                if self.learn_while_retrieving:
                    self.present_item(np.array([item, cue_idx]), source=None, update_context=False, update_weights=True, use_new_context=self.params["use_new_context"])

                rec_itemno = self.all_nos_unique[item]
                self.rec_items.append(rec_itemno)
                self.rec_times.append(cycles_elapsed * self.params["dt"])
            else:
                self.rec_items.append(-2) # reject
                self.rec_times.append(-2)

        else:
            self.rec_items.append(-1) # fail
            self.rec_times.append(-1)
            self.recog_csims.append(-1)
            self.recog_threshs.append(-1)


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Skip checks for division by zero
    def leaky_accumulator(self, float [:] in_act, float [:] x_thresholds, Py_ssize_t max_cycles):
        """
        Simulates the item retrieval process using a leaky accumulator. The process loops until an item is retrieved or the recall period ends.

        :param in_act: 1D array of incoming activation values for all items in the competition.
        :param x_thresholds: 1D array of activation thresholds required to retrieve each item in the competition.
        :param max_cycles: Maximum number of cycles the accumulator can run before the recall period ends.

        :returns: Tuple containing the index of the retrieved item (or -1 if no item was retrieved) and the number of cycles that elapsed before retrieval.
        [CMR2]
        """
        # Set up indexes
        cdef Py_ssize_t i, j, cycle = 0
        cdef Py_ssize_t nitems_in_race = in_act.shape[0]

        # Set up time constants
        cdef float dt_tau = self.params["dt_tau"]
        cdef float sq_dt_tau = self.params["sq_dt_tau"]

        # Pre-scale decay rate (kappa) based on dt
        cdef float kappa = self.params["kappa"]
        kappa *= dt_tau
        # Pre-scale inhibition (lambda) based on dt
        cdef float lamb = self.params["lamb"]
        lamb *= dt_tau
        # Take sqrt(eta) and pre-scale it based on sqrt(dt_tau)
        # Note that we do this because (for cythonization purposes) we multiply the noise
        # vector by sqrt(eta), rather than directly setting the SD to eta
        cdef float eta = self.params["eta"] ** .5
        eta *= sq_dt_tau
        # Pre-scale incoming activation based on dt
        np_in_act_scaled = np.empty(nitems_in_race, dtype=np.float32)
        cdef float [:] in_act_scaled = np_in_act_scaled
        for i in range(nitems_in_race):
            in_act_scaled[i] = in_act[i] * dt_tau

        # Set up activation variables
        np_x = np.zeros(nitems_in_race, dtype=np.float32)
        cdef float [:] x = np_x
        cdef float act
        cdef float sum_x
        cdef float delta_x
        cdef double [:] noise_vec

        # Set up winner variables
        cdef int has_retrieved_item = 0
        cdef int nwinners = 0
        np_retrieved = np.zeros(nitems_in_race, dtype=np.int32)
        cdef int [:] retrieved = np_retrieved
        cdef int [:] winner_vec
        cdef int winner
        cdef (int, int) winner_and_cycle

        # Loop accumulator until retrieving an item or running out of time
        while cycle < max_cycles and not has_retrieved_item:

            # Compute sum of activations for lateral inhibition
            sum_x = 0
            i = 0
            while i < nitems_in_race:
                sum_x += x[i]
                i += 1

            # Update activation and check whether any items were retrieved
            noise_vec = cython_randn(nitems_in_race)
            i = 0
            while i < nitems_in_race:
                # Note that kappa, lambda, eta, and in_act have all been pre-scaled above based on dt
                x[i] += in_act_scaled[i] + (eta * noise_vec[i]) - (kappa * x[i]) - (lamb * (sum_x - x[i]))
                x[i] = max(x[i], 0)
                if x[i] >= x_thresholds[i]:
                    has_retrieved_item = 1
                    nwinners += 1
                    retrieved[i] = 1
                    winner = i
                i += 1

            cycle += 1

        # If no items were retrieved, set winner to -1
        if nwinners == 0:
            winner = -1
        # If multiple items crossed the retrieval threshold on the same cycle, choose one randomly
        elif nwinners > 1:
            winner_vec = np.zeros(nwinners, dtype=np.int32)
            i = 0
            j = 0
            while i < nitems_in_race:
                if retrieved[i] == 1:
                    winner_vec[j] = i
                    j += 1
                i += 1
            rand_idx = xorshift64() % nwinners
            winner = winner_vec[rand_idx]
        # If only one item crossed the retrieval threshold, we already set it as the winner above

        # Return winning item's index within in_act, as well as the number of cycles elapsed
        winner_and_cycle = (winner, cycle)
        return winner_and_cycle


    def run_fr_trial(self):
        """
        Simulates an entire standard trial, consisting of the following steps:
        1) A pre-trial context shift.
        2) A sequence of item presentations.
        3) A pre-recall distractor (only if the mode was set to "DFR").
        4) A recall period.
        [CMR2]
        """
        ##########
        #
        # Shift context before start of new list
        #
        ##########

        # On first trial, present orthogonal item that starts the system;
        # On subsequent trials, present an interlist distractor item
        # Assume source context changes at same rate as temporal between trials
        self.phase = "pretrial"
        self.serial_position = 0
        self.beta = 1 if self.trial_idx == 0 else self.params["beta_rec_post"]
        self.beta_source = 1 if self.trial_idx == 0 else self.params["beta_rec_post"]
        # Treat initial source and intertrial source as an even mixture of all sources
        source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1

        ##########
        #
        # Present items
        #
        ##########

        self.phase = "encoding"
        for self.serial_position in range(self.pres_indexes.shape[1]):
            # Skip over any zero-padding in the presentation matrix in order to allow variable list length
            if not self.pres_nonzero_mask[self.trial_idx, self.serial_position].all():
                continue
            pres_idx = self.pres_indexes[self.trial_idx, self.serial_position]
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params["beta_enc"]
            self.beta_source = self.params["beta_source"] if self.nsources > 0 else 0
            self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])

        ##########
        #
        # Pre-recall distractor (if delayed free recall)
        #
        ##########

        if self.mode == "DFR":
            self.phase = "distractor"
            self.beta = self.params["beta_distract"]
            # Assume source context changes at the same rate as temporal during distractors
            self.beta_source = self.params["beta_distract"]
            # By default, treat distractor source as an even mixture of all sources
            # [If your distractors and sources are related, you should modify this so that you can specify distractor source.]
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1

        ##########
        #
        # Recall period
        #
        ##########

        self.phase = "recall"
        self.beta = self.params["beta_rec"]
        # Follow Polyn et al. (2009) assumption that beta_source is the same at encoding and retrieval
        self.beta_source = self.params["beta_source"] if self.nsources > 0 else 0
        self.rec_items.append([])
        self.rec_times.append([])
        if "max_recalls" in self.params:  # Limit number of recalls per trial if user has specified a maximum
            self.simulate_recall(time_limit=self.params["rec_time_limit"], max_recalls=self.params["max_recalls"])
        else:
            self.simulate_recall(time_limit=self.params["rec_time_limit"])

        self.trial_idx += 1


    def run_norm_recog_single_sess(self):
        """
        Simulates a session of normal recognition, consisting of the following steps:
        1) Pre-trial context initialization / between-trial distractor.
        2) Item presentation as encoding.
        3) Pre-recog distractor.
        4) Recognition simulation.
        [CMR-IA]
        """
        phases = ["pretrial", "encoding", "prerecall", "recognition"]
        for trial_idx in range(self.nlists):
            for self.phase in phases:

                #####
                # Shift context before start of new list
                #####
                if self.phase == "pretrial":
                    self.beta = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.beta_source = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.present_item(self.distractor_idx, source=None, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.init_csims_flag = False
                    self.list_items = []
                    self.list_pairs = []

                #####
                # Present items
                #####
                if self.phase == "encoding":
                    self.beta = self.params["beta_enc"]
                    self.beta_source = 0
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]  # if word-pair, give a pair
                        if np.logical_not(np.isscalar(pres_idx)) and pres_idx[1] == -1:
                            pres_idx = pres_idx[0].astype(int)
                        self.present_item(pres_idx, source=None, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])
                        self._record_presented_items(pres_idx)

                #####
                # Shift context before recall phase (e.g., distractor)
                #####
                if self.phase == "prerecall":
                    self.beta = self.params["beta_distract"]
                    self.beta_source = self.params["beta_distract"]
                    self.present_item(self.distractor_idx, source=None, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    do_pair = self.design == "Osth"  # do pair only for Osth's experiment
                    self._init_recent_csims(do_item=True, do_pair=do_pair)
                    self.init_csims_flag = True

                #####
                # Simulate recognition
                #####
                if self.phase == "recognition":
                    self.beta = self.params["beta_cue"]
                    self.beta_source = 0
                    for test_position in range(self.cues_indexes.shape[1]):
                        cue_idx = self.cues_indexes[trial_idx, test_position]
                        if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                        self.simulate_recog(cue_idx)
                        self._record_presented_items(cue_idx, as_list=False)


    def run_conti_recog_single_sess(self):
        """
        Simulates a session of continuous recognition, consisting of the following steps:
        1) Pre-session context initialization / between-trial distractor.
        2) Recognition.
        3) Item presentation as encoding.
        4) Loop step 1-3.
        For Hockley's variant, we changes the order of encoding and recognition.
        [CMR-IA]
        """
        # Function to initialize the flexible threshold
        def do_init_csims(self):
            if not self.params["use_flexible_thresh"]:  # do nothing if not using flexible threshold
                return
            if self.init_csims_flag:  # do nothing if already initialized
                return
            if self.design == "EXP1":  # for our Exp1, start at trial 21
                if trial_idx == 20:
                    self._init_recent_csims(do_item=True, do_pair=False)
                    self.init_csims_flag = True
            elif self.design == "Hockley":  # for Hockley's experiment, start at the first valid test probe
                if np.all(cue_idx > 0):
                    self._init_recent_csims(do_item=True, do_pair=True)
                    self.init_csims_flag = True
            else:
                raise NotImplementedError

        if self.design == "Hockley":
            phases = ["pretrial", "encoding", "recognition"]
        else:
            phases = ["pretrial", "recognition", "encoding"]
        for trial_idx in range(self.nlists):
            for self.phase in phases:

                #####
                # Shift context before each trial
                #####
                if self.phase == "pretrial":
                    self.beta = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.beta_source = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.present_item(self.distractor_idx, source=None, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                #####
                # Present items
                #####
                if self.phase == "encoding":
                    self.beta = self.params["beta_enc"]
                    self.beta_source = 0
                    self.serial_position = 0
                    pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                    if np.logical_not(np.isscalar(pres_idx)) and pres_idx[1] == -1:
                        pres_idx = pres_idx[0].astype(int)
                    if np.all(pres_idx >= 0):  # skip encoding on test-only presentations (for Hockley)
                        self.present_item(pres_idx, source=None, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])
                        self._record_presented_items(pres_idx)

                #####
                # Simulate recognition
                #####
                if self.phase == "recognition":
                    self.beta = self.params["beta_cue"]
                    self.beta_source = 0
                    cue_idx = self.cues_indexes[trial_idx]
                    if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                        cue_idx = cue_idx[0].astype(int)
                    do_init_csims(self)  # initialize the flexible threshold
                    self.simulate_recog(cue_idx)
                    self._record_presented_items(cue_idx, as_list=False)


    def run_norm_cr_single_sess(self):
        """
        Simulates a standard session of cued recall, for each list (trial), consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A pre-recall context shift (distractor, beta_distractor = 0 to cancel)
        4) A cued recall period
        [CMR-IA]
        """
        phases = ["pretrial", "encoding", "prerecall", "recall"]
        for trial_idx in range(self.nlists):
            for self.phase in phases:

                #####
                # Shift context before each trial
                #####
                if self.phase == "pretrial":
                    source = None
                    self.beta = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.beta_source = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                #####
                # Present items (pairs)
                #####             
                if self.phase == "encoding":
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                        self.beta = self.params["beta_enc"]
                        self.beta_source = 0
                        if self.params["beta_enc_inpair"] is None:  # default is Gestalt
                            self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])
                        else:  # alternatively, we can let the context drift within a pair
                            self.present_item(pres_idx[0], source, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])
                            self.beta = self.params["beta_enc_inpair"]
                            self.present_item(pres_idx[1], source, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])

                #####
                # Shift context before recall phase (e.g., distractor)
                #####
                if self.phase == "prerecall":
                    self.beta = self.params["beta_distract"]
                    self.beta_source = self.params["beta_distract"]
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                #####
                # Simulate cued recall
                #####
                if self.phase == "recall":
                    for test_position in range(self.cues_indexes.shape[1]):
                        cue_idx = self.cues_indexes[trial_idx,test_position]
                        self.beta = self.params["beta_cue"]
                        self.beta_source = 0
                        self.simulate_cr(cue_idx)


    def run_success_single_sess(self, test1_num):
        """
        Simulates a session of successive test, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A pre-recall context shift [potentially distraction]
        4) Test 1
        5) A pre-recall context shift [potentially distraction]
        6) Test 2

        :param test1_num: Integer indicating the number of items tested in test1 during successive tests.
        [CMR-IA]
        """
        if "Recog-Recog" in self.mode:
            test1 = "recognition"
            test2 = "recognition"
        elif "Recog-CR" in self.mode:
            test1 = "recognition"
            test2 = "cued recall"
        elif "CR-Recog" in self.mode:
            test1 = "cued recall"
            test2 = "recognition"
        elif "CR-CR" in self.mode:
            test1 = "cued recall"
            test2 = "cued recall"
        phases = ["pretrial", "encoding", "prerecall", test1, "prerecall", test2]
        for trial_idx in range(self.nlists):
            is_test1 = True
            for self.phase in phases:

                #####
                # Shift context before each trial
                #####
                if self.phase == "pretrial":
                    source = None
                    self.beta = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.beta_source = 1 if trial_idx == 0 else self.params["beta_rec_post"]
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.init_csims_flag = False
                    self.list_items = []
                    self.list_pairs = []

                #####
                # Shift context before recall phase (e.g., distractor)
                #####
                if self.phase == "prerecall":
                    self.beta = self.params["beta_distract"]
                    self.beta_source = self.params["beta_distract"]
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)  # reset threshold
                    if not self.init_csims_flag:
                        self._init_recent_csims(do_item=True, do_pair=True)
                        self.init_csims_flag = True

                #####
                # Present items
                #####
                if self.phase == "encoding":
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                        self.beta = self.params["beta_enc"]
                        self.beta_source = 0
                        self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params["use_new_context"])
                        self._record_presented_items(pres_idx)

                #####
                # Simulate recognition
                #####            
                if self.phase == "recognition":
                    if is_test1:
                        cue_indexes = self.cues_indexes[trial_idx, :test1_num]
                        is_test1 = False
                    else:
                        cue_indexes = self.cues_indexes[trial_idx, test1_num:]
                    for cue_idx in cue_indexes:
                        if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                        self.beta = self.params["beta_cue"]
                        self.beta_source = 0
                        self.simulate_recog(cue_idx)  # can be pair, can be scalar
                        self._record_presented_items(cue_idx, as_list=False)

                #####
                # Simulate cued recall
                #####                 
                if self.phase == "cued recall":
                    if is_test1:
                        cue_indexes = self.cues_indexes[trial_idx, :test1_num]
                        is_test1 = False
                    else:
                        cue_indexes = self.cues_indexes[trial_idx, test1_num:]
                    for cue_idx in cue_indexes:
                        if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                        self.beta = self.params["beta_cue"]
                        self.beta_source = 0
                        self.simulate_cr(cue_idx)  # should be scalar


# ---------- Wrapper functions ---------- #

def run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode="IFR"):
    """
    Simulates a single session of free recall using the specified parameter set.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored, allowing zero-padding for varying list lengths.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: 3D array of source features for each presented word if not None. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i.
    :param mode: String indicating the type of free recall to simulate. "IFR" for immediate free recall or "DFR" for delayed recall.

    :returns: Two 2D arrays. The first contains the ID numbers of items the model recalled on each trial. The second contains the response times of each item relative to the start of the recall period.
    [CMR2]
    """
    ntrials = pres_mat.shape[0]

    # Simulate all trials of the session using CMR2
    cmr = CMR(params, pres_mat, sem_mat, source_mat=source_mat, mode=mode)
    for i in range(ntrials):
        cmr.run_fr_trial()

    # Get the model's simulated recall data
    rec_items = cmr.rec_items
    rec_times = cmr.rec_times

    # Identify the max number of recalls made on any trial
    max_recalls = max([len(trial_data) for trial_data in rec_times])

    # Zero-pad response data into an ntrials x max_recalls matrix
    rec_mat = np.zeros((ntrials, max_recalls), dtype=int)
    time_mat = np.zeros((ntrials, max_recalls))
    for i, trial_data in enumerate(rec_items):
        trial_nrec = len(trial_data)
        if trial_nrec > 0:
            rec_mat[i, :trial_nrec] = rec_items[i]
            time_mat[i, :trial_nrec] = rec_times[i]

    return rec_mat, time_mat


def run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode="IFR"):
    """
    Simulates multiple sessions of free recall using a single set of parameters.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored, allowing zero-padding for varying list lengths.
    :param identifiers: 1D array of session numbers, subject IDs, or other values indicating how the rows/trials in pres_mat and source_mat should be divided into sessions. For example, to simulate two four-trial sessions, set identifiers to np.array([0, 0, 0, 0, 1, 1, 1, 1]), indicating the latter four trials are from a different session than the first four.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: 3D array of source features for each presented word if not None. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i.
    :param mode: String indicating the type of free recall to simulate. "IFR" for immediate free recall or "DFR" for delayed recall.

    :returns: Two 2D arrays. The first contains the ID numbers of items the model recalled on each trial. The second contains the response times of each item relative to the start of the recall period.
    [CMR2]
    """
    now_test = time.time()

    # Split data based on identifiers provided
    pres, unique_ids, sources = split_data(pres_mat, identifiers, source_mat=source_mat)

    # Run CMR2 for each subject/session
    rec_items = []
    rec_times = []
    for i, sess_pres in enumerate(pres):
        sess_sources = None if sources is None else sources[i]
        out_tuple = run_cmr2_single_sess(params, sess_pres, sem_mat, source_mat=sess_sources, mode=mode)
        rec_items.append(out_tuple[0])
        rec_times.append(out_tuple[1])
    
    # Identify the maximum number of recalls made in any session
    max_recalls = max([sess_data.shape[1] for sess_data in rec_items])

    # Zero-pad response data into an total_trials x max_recalls matrix where rows align with those in the original data_mat
    total_trials = len(identifiers)
    rec_mat = np.zeros((total_trials, max_recalls), dtype=int)
    time_mat = np.zeros((total_trials, max_recalls))
    for i, uid in enumerate(unique_ids):
        sess_max_recalls = rec_items[i].shape[1]
        if sess_max_recalls > 0:
            rec_mat[identifiers == uid, :sess_max_recalls] = rec_items[i]
            time_mat[identifiers == uid, :sess_max_recalls] = rec_times[i]

    print("CMR Time: " + str(time.time() - now_test))

    return rec_mat, time_mat


def run_norm_recog_multi_sess(params, df_study, df_test, sem_mat, design=None, disable_tqdm=False):
    """
    Simulates multiple sessions of normal recognition (recognition after studying a list of items) using a single set of parameters. Only item recognition for now.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df_study: DataFrame containing the study list with columns "session" and "itemno".
    :param df_test: DataFrame containing the test list with columns "session" and "itemno".
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.

    :returns: DataFrame with columns "session", "list", and "test_itemno" in df_test plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for the test probe).
    [CMR-IA]
    """
    now_test = time.time()
    
    sessions = np.unique(df_study.session)
    list_num = len(np.unique(df_study.list))
    df_thin = df_test[["session", "list", "itemno1", "itemno2"]]

    resps, rts, csims, threshs, probs = [], [], [], [], []
    for sess in tqdm(sessions, disable=disable_tqdm):

        # Extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ["itemno1", "itemno2"]].to_numpy()
        pres_mat = np.reshape(pres_mat, (list_num, -1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, ["itemno1", "itemno2"]].to_numpy()
        cue_mat = np.reshape(cue_mat, (list_num, -1, 2))

        # Run CMR for each session
        cmr_model = CMR(params, pres_mat, sem_mat, cue_mat=cue_mat, mode="RecogNormal", design=design, seed=sess)
        cmr_model.run_norm_recog_single_sess()

        # Save results
        resps += cmr_model.rec_items
        rts += cmr_model.rec_times
        csims += cmr_model.recog_csims
        threshs += cmr_model.recog_threshs
        probs += cmr_model.recog_probs

    df_thin = df_thin.assign(s_resp=resps, s_rt=rts, csim=csims, thresh=threshs, prob=probs)
    print("CMR Time: " + str(time.time() - now_test))

    return df_thin


def run_conti_recog_multi_sess(params, df, sem_mat, design=None, disable_tqdm=False):
    """
    Simulates multiple sessions of continuous recognition using a single set of parameters.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df: DataFrame containing the study list with columns "session", "position", "study_itemno1", "study_itemno2", "test_itemno1", and "test_itemno2". For item recognition, "study_itemno2" and "test_itemno2" should be -1.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).
    :param design: String indicating the experimental design variant. Set "Hockley" for Hockley's variant of continuous recognition. None uses the standard paradigm.

    :returns: Dataframe with columns "session", "position", "study_itemno1", "study_itemno2", "test_itemno1", and "test_itemno2" in df plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for the test probe).
    [CMR-IA]
    """
    now_test = time.time()

    sessions = np.unique(df.session)
    df_thin = df[["session", "position", "study_itemno1", "study_itemno2", "test_itemno1", "test_itemno2"]]

    resps, rts, csims, threshs, probs = [], [], [], [], []
    for sess in tqdm(sessions, disable=disable_tqdm):

        # Extarct the session data
        pres_mat = df_thin.loc[df_thin.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy()
        pres_mat = np.reshape(pres_mat, (len(pres_mat), 1, 2))  # each presentation here is treated as a length-1 list
        cue_mat = df_thin.loc[df_thin.session == sess, ["test_itemno1", "test_itemno2"]].to_numpy()

        # Run CMR for each session
        cmr_model = CMR(params, pres_mat, sem_mat, cue_mat=cue_mat, mode="RecogContinuous", design=design, seed=sess)
        cmr_model.run_conti_recog_single_sess()

        # Save results
        resps += cmr_model.rec_items
        rts += cmr_model.rec_times
        csims += cmr_model.recog_csims
        threshs += cmr_model.recog_threshs
        probs += cmr_model.recog_probs

    df_thin = df_thin.assign(s_resp=resps, s_rt=rts, csim=csims, thresh=threshs, prob=probs)
    print("CMR Time: " + str(time.time() - now_test))

    return df_thin


def run_norm_cr_multi_sess(params, df_study, df_test, sem_mat, disable_tqdm=False):
    """
    Simulates multiple sessions of cued recall using a single set of parameters.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df_study: DataFrame containing the study list with columns "session", "list", "study_itemno1", and "study_itemno2".
    :param df_test: DataFrame containing the test list with columns "session", "list", and "test_itemno".    
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).
    
    :returns: 
    - df_thin: Dataframe with columns "session", "list", and "test_itemno" in df_test plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for intrusion filtering).
    - f_in_acc: List of Arrays of f_in for cued recalls (for developers).
    - f_in_dif: List of Arrays of f_in minus retrieval threshold for cued recalls (for developers).
    [CMR-IA]
    """
    now_test = time.time()

    sessions = np.unique(df_study.session)
    list_num = len(np.unique(df_study.list))
    df_thin = df_test[["session", "list", "test_itemno"]]

    resps, rts, csims = [], [], []
    f_in, f_dif = [], []
    for sess in tqdm(sessions, disable=disable_tqdm):

        # Extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy()
        pres_mat = np.reshape(pres_mat, (list_num, -1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, "test_itemno"].to_numpy()
        cue_mat = np.reshape(cue_mat, (list_num, -1))

        # Run CMR for each session
        cmr_model = CMR(params, pres_mat, sem_mat, cue_mat=cue_mat, mode="CRNormal", seed=sess)
        cmr_model.run_norm_cr_single_sess()

        # Save results
        resps += cmr_model.rec_items
        rts += cmr_model.rec_times
        csims += cmr_model.recog_csims
        f_in.append(cmr_model.f_in_acc)
        f_dif.append(cmr_model.f_in_dif)

    df_thin = df_thin.assign(s_resp=resps, s_rt=rts, csim=csims)
    print("CMR Time: " + str(time.time() - now_test))

    return df_thin, f_in, f_dif


def run_success_multi_sess(params, df_study, df_test, sem_mat, mode="Recog-CR", design=None, disable_tqdm=False):
    """
    Simulates multiple sessions of sucessitve tests using a single set of parameters.

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df_study: DataFrame containing the study list with columns "session", "list", "study_itemno1", and "study_itemno2".
    :param df_test: DataFrame containing the test list with columns "session", "list", "test_itemno1", and "test_itemno2".
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).
    :param mode: String indicating the type of successive tests to simulate. Set "Recog-Recog" for recognition-recognition, "Recog-CR" for recognition-cued recall, "CR-Recog" for cued recall-recognition, or "CR-CR" for cued recall-cued recall.

    :returns: 
    - df_thin: Dataframe with columns "session", "list", "test_itemno1", and "test_itemno2" in df_test plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for the test probe in recognition or intrusion filtering in cued recall).
    - f_in_acc: List of Arrays of f_in for cued recalls (for developers).
    - f_in_dif: List of Arrays of f_in minus retrieval threshold for cued recalls (for developers).
    [CMR-IA]
    """
    now_test = time.time()

    sessions = np.unique(df_study.session)
    list_num = len(np.unique(df_study.list))
    df_thin = df_test[["session", "list", "test_itemno1", "test_itemno2"]]
    test1_num = sum(df_test.query("session == 0 and list == 0").test == 1)

    resps, rts, csims, threshs = [], [], [], []
    f_in, f_dif = [], []
    for sess in tqdm(sessions, disable=disable_tqdm):

        # Extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ["study_itemno1", "study_itemno2"]].to_numpy()
        pres_mat = np.reshape(pres_mat, (list_num, -1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, ["test_itemno1", "test_itemno2"]].to_numpy()
        cue_mat = np.reshape(cue_mat, (list_num, -1, 2))

        # Run CMR for each session
        cmr_model = CMR(params, pres_mat, sem_mat, cue_mat=cue_mat, mode=mode, design=design, seed=sess)
        cmr_model.run_success_single_sess(test1_num=test1_num)

        # Save results
        resps += cmr_model.rec_items
        rts += cmr_model.rec_times
        csims += cmr_model.recog_csims
        threshs += cmr_model.recog_threshs
        f_in.append(cmr_model.f_in_acc)
        f_dif.append(cmr_model.f_in_dif)

    df_thin = df_thin.assign(s_resp=resps, s_rt=rts, csim=csims, thresh=threshs)
    print("CMR Time: " + str(time.time() - now_test))

    return df_thin, f_in, f_dif