from __future__ import print_function
# import mkl
# mkl.set_num_threads(1)
import os
import sys
import math
import time
import json
import numpy as np
import scipy.io
from tqdm import tqdm
from glob import glob
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport log, sqrt
cimport numpy as np
cimport cython

"""
Known differences from Lynn Lohnas's code:
1) Cycle counter starts at 0 in this code instead of 1 during leaky accumulator.
2) No empty feature vector is presented at the end of the recall period.
"""


# Credit to "senderle" for the cython random number generation functions used below. Original code can be found at:
# https://stackoverflow.com/questions/42767816/what-is-the-most-efficient-and-portable-way-to-generate-gaussian-random-numbers
@cython.cdivision(True)
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


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
    out[assign_ix + 1] = x2 * 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef cython_randn(int n):
    cdef int i
    np_result = np.zeros(n, dtype='f8', order='C')
    cdef double[:] result = np_result
    for i in range(n // 2):  # Int division ensures trailing index if n is odd.
        assign_random_gaussian_pair(result, i * 2)
    if n % 2 == 1:
        result[n - 1] = random_gaussian()

    return result


class CMR(object):

    def __init__(self, params, pres_mat, sem_mat,
                 source_mat=None, rec_mat=None, ffr_mat=None, cue_mat=None,
                 task='FR', mode='IFR', test1_num=None):
        """
        Initializes a CMR object and prepares it to simulate the session defined by pres_mat. [Modified]

        :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
        :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored. If presenting word pairs, use a 3D array with the length of the third dimension equals to 2.
        :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
        :param source_mat: 3D array of source features for each presented word. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i. If None, no source features are used.
        :param rec_mat: 2D array of ID numbers of words recalled by real subjects in a free recall phase on each trial. Rows correspond to pres_mat.
        :param ffr_mat: 1D array of ID numbers of words recalled by real subjects in a final free recall phase.
        :param cue_mat: 1D array of ID numbers of words presented to the model in recognition and cued recall. If presenting word pairs, use a 2D array with the length of the second dimension equals to 2.
        :param task: String indicating the type of task to simulate. 'FR': free recall; 'CR': cued recall; 'Recog': recognition; 'Success': successive tests.
        :param mode: String indicating the task mode to simulate.
            - For 'FR': 'IFR' (immediate free recall), 'DFR' (delayed recall).
            - For 'CR': 'Final' (cued recall in final stage).
            - For 'Recog': 'Continuous' (continuous recognition), 'Hockley' (Hockley's variation of continuous recognition), 'Final' (recognition in final stage).
            - For 'Success': 'Recog-CR' (test1 recognition, test2 cued recall), 'Recog-Recog' (test1 & test2 both recognition), 'CR-Recog' (test1 cued recall, test2 recognition), 'CR-CR' (test1 & test2 both cued recall).
        :param test1_num: Integer indicating the number of items tested in test1 during successive tests.
        """
        ##########
        #
        # Set up model parameters and presentation data
        #
        ##########

        # Convert input parameters
        self.params = params  # Dictionary of model parameters
        self.pres_nos = np.array(pres_mat, dtype=np.int16)  # Presented item ID numbers (trial x serial position)
        self.sem_mat = np.array(sem_mat, dtype=np.float32)  # Semantic similarity matrix (e.g. Word2vec, LSA, WAS)
        self.extra_distract = 0

        ### [bj] begin
        # input cue mat
        if cue_mat is None:
            self.have_cue = False
        else:
            self.have_cue = True
            self.cues_nos = np.array(cue_mat, dtype=np.int16)
        # input recall mat
        if rec_mat is None:
            self.have_rec = False
        else:
            self.have_rec = True
            self.rec_nos = np.array(rec_mat, dtype=np.int16)
        # input final free recall mat
        if ffr_mat is None:
            self.have_ffr = False
        else:
            self.have_ffr = True
            self.ffr_nos = np.array(ffr_mat, dtype=np.int16)
        # input source
        if source_mat is None:
            self.nsources = 0
        else:
            self.sources = np.atleast_3d(source_mat).astype(np.float32)
            self.nsources = self.sources.shape[2]
            if self.sources.shape[0:2] != self.pres_nos.shape[0:2]:
                raise ValueError('Source matrix must have the same number of rows and columns as the presented item matrix.')
        # input task
        if task not in ('FR', 'CR', 'Recog', 'Success'): # Task must in FR or CR or Recog
            raise ValueError('Task must be "FR" or "CR" or "Recog" or "Success", not %s.' % task)
        if (task is 'CR' or task is 'Recog') and cue_mat is None:
            raise ValueError('%s Must input a cue matrix.' % task)
        self.task = task
        # input mode
        if mode not in ('IFR', 'DFR', 'Continuous', 'Final', 'Hockley', 'Recog-Recog', 'Recog-CR', 'CR-Recog', 'CR-CR'):
            raise ValueError('Mode %s is invalid.' % mode)
        self.mode = mode
        # input learn_while_retrieving
        self.learn_while_retrieving = self.params['learn_while_retrieving'] if 'learn_while_retrieving' in self.params else False
        # necessary for successive tests
        self.test1_num = test1_num

        # Determine the number of lists and the maximum list length (how many words or word-pairs)
        self.nlists = self.pres_nos.shape[0]
        self.max_list_length = self.pres_nos.shape[1]

        # Create arrays of sorted and unique (nonzero) items
        self.pres_nonzero_mask = self.pres_nos > 0
        self.pres_nos_nonzero = self.pres_nos[self.pres_nonzero_mask] # reduce to 1D
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
            self.all_nos = np.concatenate((self.all_nos,self.cues_nos_nonzero),axis=None)
        self.all_nos_sorted = np.sort(self.all_nos)
        self.all_nos_unique = np.unique(self.all_nos_sorted)  # 1D, order in feature vector

        # Convert presented item and cue item ID numbers to indexes within the feature vector
        indexer = lambda x: np.searchsorted(self.all_nos_unique, x) if x > 0 else x
        indexer_func = np.vectorize(indexer)
        self.pres_indexes = indexer_func(self.pres_nos)
        if self.have_rec:
            self.rec_indexes = indexer_func(self.rec_nos)
        if self.have_ffr:
            self.ffr_indexes = indexer_func(self.ffr_nos)
        if self.have_cue:
            self.cues_indexes = indexer_func(self.cues_nos)
        ### [bj] end
        
        # Make sure items' associations with themselves are set to 0
        np.fill_diagonal(self.sem_mat, 0)

        # [bj] average semantic association with other items (for attention and criteria shift) 
        self.sem_mean = np.sum(self.sem_mat, axis=1)/(np.shape(self.sem_mat)[1] - 1)
        
        # Cut down semantic matrix to contain only the items in the session
        self.sem_mat = self.sem_mat[self.all_nos_unique - 1, :][:, self.all_nos_unique - 1]

        # Initial phase
        self.phase = None

        # Initial beta and encoding variability [bj]
        self.beta = 0
        self.beta_source = 0
        self.var_enc_p = self.params['var_enc']
        self.changestate_rng = np.random.default_rng(seed=42)

        ##########
        #
        # Set up context and feature vectors
        #
        ##########

        # Determine number of cells in each region of the feature/context vectors
        # self.nitems = self.pres_nos.size # [bj]
        self.nitems_unique = len(self.all_nos_unique) # [bj]
        if self.mode == 'Final':  # for PEERS task
            self.extra_distract += 1
        if self.task == 'Success':  # for successive tests
            self.extra_distract += 2*self.nlists
        if self.task == 'CR':  # for norm cued recall
            self.extra_distract += self.nlists
        self.ndistractors = self.nlists + self.extra_distract # One distractor prior to each list + ffr + recog
        if self.mode == 'DFR':
            self.ndistractors += self.nlists  # One extra distractor before each recall period if running DFR
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
        self.prim_vec = self.params['phi_s'] * np.exp(-1 * self.params['phi_d'] * np.arange(self.max_list_length)) + 1

        # Set up learning rate matrix for M_FC (dimensions are context x features)
        self.L_FC = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_FC.fill(self.params['gamma_fc']) # if no source, uniformly gamma_fc
        else:
            # Temporal Context x Item Features (items reinstating their previous temporal contexts)
            self.L_FC[:self.ntemporal, :self.ntemporal] = self.params['L_FC_tftc']
            # Temporal Context x Source Features (sources reinstating previous temporal contexts)
            self.L_FC[:self.ntemporal, self.ntemporal:] = self.params['L_FC_sftc']
            # Source Context x Item Features (items reinstating previous source contexts)
            self.L_FC[self.ntemporal:, :self.ntemporal] = self.params['L_FC_tfsc']
            # Source Context x Source Features (sources reinstating previous source contexts)
            self.L_FC[self.ntemporal:, self.ntemporal:] = self.params['L_FC_sfsc']

        # Set up learning rate matrix for M_CF (dimensions are features x context)
        self.L_CF = np.empty((self.nelements, self.nelements), dtype=np.float32)
        if self.nsources == 0:
            self.L_CF.fill(self.params['gamma_cf']) # if no source, uniformly gamma_cf
        else:
            # Item Features x Temporal Context (temporal context cueing retrieval of items)
            self.L_CF[:self.ntemporal, :self.ntemporal] = self.params['L_CF_tctf']
            # Item Features x Source Context (source context cueing retrieval of items)
            self.L_CF[:self.ntemporal, self.ntemporal:] = self.params['L_CF_sctf']
            # Source Features x Temporal Context (temporal context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, :self.ntemporal] = self.params['L_CF_tcsf']
            # Source Features x Source Context (source context cueing retrieval of sources)
            self.L_CF[self.ntemporal:, self.ntemporal:] = self.params['L_CF_scsf']

        # Initialize weight matrices as identity matrices
        self.M_FC = np.identity(self.nelements, dtype=np.float32)
        self.M_CF = np.identity(self.nelements, dtype=np.float32)

        # Scale the semantic similarity matrix by s_fc (Healey et al., 2016) and s_cf (Lohnas et al., 2015)
        fc_sem_mat = self.params['s_fc'] * self.sem_mat
        cf_sem_mat = self.params['s_cf'] * self.sem_mat

        # Complete the pre-experimental associative matrices by layering on the scaled semantic matrices
        self.M_FC[:self.nitems_unique, :self.nitems_unique] += fc_sem_mat # Elements include distractors and items, sem_mat just apply to items
        self.M_CF[:self.nitems_unique, :self.nitems_unique] += cf_sem_mat

        # Scale pre-experimental associative matrices by 1 - gamma
        self.M_FC *= 1 - self.L_FC
        self.M_CF *= 1 - self.L_CF

        #####
        #
        # Initialize leaky accumulator and recall variables
        #
        #####

        self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)  # Retrieval thresholds
        if self.params['ban_recall'] is not None: # [bj] items that should not be recalled, necessary for simu8
            no_recall_items = self.params['ban_recall']
            self.ret_thresh[no_recall_items] = np.inf
        self.nitems_in_race = self.params['nitems_in_accumulator']  # Number of items in accumulator
        self.rec_items = []  # Recalled items from each trial
        self.rec_times = []  # Rectimes of recalled items from each trial

        # Calculate dt_tau and its square root based on dt
        self.params['dt_tau'] = self.params['dt'] / 1000.
        self.params['sq_dt_tau'] = np.sqrt(self.params['dt_tau'])

        # Set up random seed
        srand(12345)

        ##########
        #
        # Initialize variables for tracking simulation progress
        #
        ##########

        self.trial_idx = 0  # Current trial number (0-indexed)
        self.serial_position = 0  # Current serial position (0-indexed)
        self.distractor_idx = self.nitems_unique  # Current distractor index
        self.first_source_idx = self.ntemporal  # Index of the first source feature

        # [bj] For analysis
        self.f_in_acc = []
        self.f_in_dif = []
        self.recog_similarity = []

        ##########
        #
        # [bj] Set up elevated-attention, criteria-shift, retrieval variability
        #
        ##########
        
        # Set up elevated-attention scaling vector for all itemno
        self.att_vec = self.params['psi_s'] * self.sem_mean + self.params['psi_c']
        self.att_vec[self.att_vec > 1/self.params['gamma_fc']] = 1/self.params['gamma_fc']
        self.att_vec[self.att_vec < 0] = 0

        # Set up c_thresh vector for all itemno, allowing criterion shifting for different items
        self.c_vec = self.params['c_s'] * self.sem_mean + self.params['c_thresh_itm']

        # [bj] extract model-calculated word frequency
        self.b0 = 6.8657
        self.b1 = -12.7856
        self.cal_word_freq = np.exp(self.b0 + self.sem_mean * self.b1)

        # Set up random mechanism for threshold
        self.thresh_rng = np.random.default_rng(87)
        self.thresh_sigma = self.params['thresh_sigma']


    def present_item(self, item_idx, source=None, update_context=True, update_weights=True, use_new_context=False):
        """
        Presents a single item (or distractor) to the model by updating the feature vector. Options are provided to update context and the model's associative matrices after presentation. [Modified]

        :param item_idx: Index of the cell within the feature vector to be activated by the presented item.
        :param source: If None, no source features are activated. If a 1D array, the source features in the feature vector are set to match the numbers in the source array.
        :param update_context: If True, the context vector updates after the feature vector is updated.
        :param update_weights: If True, the model's weight matrices update to strengthen the association between the presented item and the context state.
        :param use_new_context: If True, use the updated context vector to update the weight matrices (used in the paper). If False, use the old context vector (used in CMR2).
        """
        ##########
        #
        # Activate item's features
        #
        ##########

        paired_pres = np.logical_not(np.isscalar(item_idx))

        # Activate the presented item itself
        self.f.fill(0)
        if item_idx is not None:
            self.f[item_idx] = 1

        # Activate the source feature(s) of the presented item
        if self.nsources > 0 and source is not None:
            self.f[self.first_source_idx:, 0] = np.atleast_1d(source)

        # [bj] copy c_old
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
        # Update weight matrices
        #
        ##########

        if update_weights:
            if use_new_context:  # [bj] use updated c, as in the paper
                if self.phase == 'encoding':
                    # [bj] only apply elevated-attention and primacy during encoding
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c[:self.nitems_unique], self.f[:self.nitems_unique].T) \
                        * np.mean(self.att_vec[self.all_nos_unique[item_idx]-1])  # [bj] mean for pair presentation, not used
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
            else:  # [bj] alternatively, use c_old, as in previous CMR models
                if self.phase == 'encoding':
                    self.M_FC[:self.nitems_unique,:self.nitems_unique] \
                        += self.L_FC[:self.nitems_unique,:self.nitems_unique] \
                        * np.dot(self.c_old[:self.nitems_unique], self.f[:self.nitems_unique].T) \
                        * np.mean(self.att_vec[self.all_nos_unique[item_idx]-1])
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
                if paired_pres: # [bj] direct association for pair, not used in the paper
                    pair_ass = self.params['d_assoc'] * np.dot(self.f, self.f.T)
                    np.fill_diagonal(pair_ass, 0)
                    self.M_FC += self.L_FC * pair_ass
                    self.M_CF += self.L_CF * self.prim_vec[self.serial_position] * pair_ass


    def simulate_recall(self, time_limit=60000, max_recalls=np.inf):
        """
        Simulates a recall period starting from the current state of context. [Unchanged from CMR2]

        :param time_limit: Simulated duration of the recall period in milliseconds. Determines the number of cycles of the leaky accumulator before the recall period ends.
        :param max_recalls: Maximum number of retrievals (not overt recalls) the model is allowed to make. If this limit is reached, the recall period ends early. This setting prevents the model from consuming excessive runtime if its parameters cause it to make numerous recalls per trial.
        """
        cycles_elapsed = 0
        nrecalls = 0
        max_cycles = time_limit // self.params['dt']

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
                self.ret_thresh = 1 + self.params['alpha'] * (self.ret_thresh - 1)
                self.ret_thresh[item] = 1 + self.params['omega']

                # Present retrieved item to the model, with no source information
                if self.learn_while_retrieving:
                    self.present_item(item, source=None, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])  # [bj]
                else:
                    self.present_item(item, source=None, update_context=True, update_weights=False)

                # Filter intrusions using temporal context comparison, and log item if overtly recalled
                c_similarity = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
                if c_similarity >= self.params['c_thresh']:
                    rec_itemno = self.all_nos_unique[item] # [bj]
                    self.rec_items[-1].append(rec_itemno)
                    self.rec_times[-1].append(cycles_elapsed * self.params['dt'])


    def simulate_recog(self, cue_idx):
        """
        Simulate a recognition. [Newly added]

        :param cue_idx: The index of the provided cue in the feature vector.
        """
        # Present cue and update the context
        paired_cue = np.logical_not(np.isscalar(cue_idx))
        if self.mode == "Final":
            self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
        elif self.mode == "Continuous":
            self.present_item(cue_idx, source=None, update_context=False, update_weights=False)
        elif self.mode == "Hockley" or self.mode == "Recog-CR" or self.mode == "Recog-Recog" or self.mode == "CR-Recog":
            if paired_cue: # pair cue
                self.present_item(cue_idx[0], source=None, update_context=True, update_weights=False)
                self.present_item(cue_idx[1], source=None, update_context=True, update_weights=False)
                # self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
            else:
                self.present_item(cue_idx, source=None, update_context=True, update_weights=False)

        # calculate context similarity
        # c_similarity, rt = self.diffusion(self.c_old[:self.nitems_unique], self.c_in[:self.nitems_unique], max_time=self.params['rec_time_limit'])
        c_similarity = np.dot(self.c_old[:self.nitems_unique].T, self.c_in[:self.nitems_unique]) # [bj] similarity should not include distractors
        rt = self.params['a'] * np.exp(-1 * self.params['b'] * np.abs(c_similarity - self.params['c_thresh_itm'])) # [bj] under-developed, not used in the paper
        self.recog_similarity.append(c_similarity.item())
        self.rec_times.append(rt.item())

        # get recognition threshold
        if paired_cue:
            thresh = self.params['c_thresh_assoc'] + self.thresh_rng.uniform(-self.thresh_sigma, self.thresh_sigma)
        else:
            thresh = self.c_vec[self.all_nos_unique[cue_idx] - 1] + self.thresh_rng.uniform(-self.thresh_sigma, self.thresh_sigma)

        if c_similarity >= thresh:
            self.rec_items.append(1)  # YES
            if self.learn_while_retrieving:  # output encoding for judged-as-old items or pairs
                self.present_item(cue_idx, source=None, update_context=False, update_weights=True, use_new_context=self.params['use_new_context'])
        else:
            self.rec_items.append(0)  # NO


    def simulate_cr(self, cue_idx, time_limit=5000):
        """
        Simulate a cued recall. [Newly added]

        :param cue_idx: The index of the provided cue in the feature vector.
        :param time_limit: The simulated duration of the recall period (in ms). Determines how many cycles of the leaky accumulator will run before the recall period ends.
        """
        cycles_elapsed = 0
        max_cycles = time_limit // self.params['dt']

        # present cue and update the context
        self.present_item(cue_idx, source=None, update_context=True, update_weights=False)
        if not np.isinf(self.ret_thresh[cue_idx]):
            self.ret_thresh[cue_idx] = 1 + self.params['omega']  # can't recall the cue!

        # Use context to cue items
        f_in = np.dot(self.M_CF, self.c)[:self.nitems_unique].flatten()
        self.f_in_acc.append(f_in)  # for testing
        self.f_in_dif.append(f_in - self.ret_thresh)  # for testing, distance to threshold

        # Identify set of items with the highest activation
        top_items = np.argsort(f_in)[self.nitems_unique - self.nitems_in_race:]  # returns the original index of the sorted order
        if self.params['ban_recall'] is not None:
            top_items = [x for x in top_items if x not in self.params['ban_recall']]
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
            self.ret_thresh = 1 + self.params['alpha'] * (self.ret_thresh - 1)

            # Present retrieved item to the model, with no source information
            self.beta = self.params['beta_rec']
            self.present_item(item, source=None, update_context=True, update_weights=False)

            # Filter intrusions using temporal context comparison, and log item if overtly recalled
            c_similarity = np.dot(self.c_old[:self.ntemporal].T, self.c_in[:self.ntemporal])
            self.recog_similarity.append(c_similarity.item())
            if c_similarity >= self.params['c_thresh']:

                # Set the retrieved item's threshold to maximum
                if not np.isinf(self.ret_thresh[item]):
                    self.ret_thresh[item] = 1 + self.params['omega']

                # output encoding for the pair of cue and recalled item
                if self.learn_while_retrieving:
                    self.present_item([item,cue_idx], source=None, update_context=False, update_weights=True, use_new_context=self.params['use_new_context'])

                rec_itemno = self.all_nos_unique[item]
                self.rec_items.append(rec_itemno)
                self.rec_times.append(cycles_elapsed * self.params['dt'])
            else:
                self.rec_items.append(-2) # reject
                self.rec_times.append(-2)

        else:
            self.rec_items.append(-1) # fail
            self.rec_times.append(-1)
            self.recog_similarity.append(-1)


    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing
    @cython.cdivision(True)  # Skip checks for division by zero
    def leaky_accumulator(self, float [:] in_act, float [:] x_thresholds, Py_ssize_t max_cycles):
        """
        Simulates the item retrieval process using a leaky accumulator. The process loops until an item is retrieved or the recall period ends. [Unchanged from CMR2]

        :param in_act: 1D array of incoming activation values for all items in the competition.
        :param x_thresholds: 1D array of activation thresholds required to retrieve each item in the competition.
        :param max_cycles: Maximum number of cycles the accumulator can run before the recall period ends.

        :returns: Tuple containing the index of the retrieved item (or -1 if no item was retrieved) and the number of cycles that elapsed before retrieval.
        """
        # Set up indexes
        cdef Py_ssize_t i, j, cycle = 0
        cdef Py_ssize_t nitems_in_race = in_act.shape[0]

        # Set up time constants
        cdef float dt_tau = self.params['dt_tau']
        cdef float sq_dt_tau = self.params['sq_dt_tau']

        # Pre-scale decay rate (kappa) based on dt
        cdef float kappa = self.params['kappa']
        kappa *= dt_tau
        # Pre-scale inhibition (lambda) based on dt
        cdef float lamb = self.params['lamb']
        lamb *= dt_tau
        # Take sqrt(eta) and pre-scale it based on sqrt(dt_tau)
        # Note that we do this because (for cythonization purposes) we multiply the noise
        # vector by sqrt(eta), rather than directly setting the SD to eta
        cdef float eta = self.params['eta'] ** .5
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
            # srand(time.time())
            rand_idx = rand() % nwinners  # see http://www.delorie.com/djgpp/doc/libc/libc_637.html
            winner = winner_vec[rand_idx]
        # If only one item crossed the retrieval threshold, we already set it as the winner above

        # Return winning item's index within in_act, as well as the number of cycles elapsed
        winner_and_cycle = (winner, cycle)
        return winner_and_cycle


    # def diffusion(self, c1, c2, max_time=5000):
    #     """
    #     An experimental mechanism to calculate RT. Not used. [Newly added]
    #     """
    #     if len(c1) != len(c2):
    #         print('err')
    #     len_c = len(c1)

    #     dt = self.params['dt']
    #     dot_order = np.random.permutation(len_c)
    #     total_time = 0
    #     c_similarity = 0

    #     for i in dot_order:
    #         if total_time > max_time:
    #             total_time = max_time
    #             break

    #         c_similarity += c1[i] * c2[i]
    #         total_time += dt

    #         if c_similarity >= self.params['c_thresh']:
    #             break

    #     return c_similarity, total_time


    def run_fr_trial(self):
        """
        Simulates an entire standard trial, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item presentations
        3) A pre-recall distractor (only if the mode was set to 'DFR')
        4) A recall period
        [Unchanged from CMR2]
        """
        ##########
        #
        # Shift context before start of new list
        #
        ##########

        # On first trial, present orthogonal item that starts the system;
        # On subsequent trials, present an interlist distractor item
        # Assume source context changes at same rate as temporal between trials
        # initialize context vector to have non-zero elements, no updating matrix
        self.phase = 'pretrial'
        self.serial_position = 0
        self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post'] # learn pre-trial distractor
        self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
        # Treat initial source and intertrial source as an even mixture of all sources
        #source = np.zeros(self.nsources) if self.nsources > 0 else None
        source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
        self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
        self.distractor_idx += 1

        ##########
        #
        # Present items
        #
        ##########

        self.phase = 'encoding'
        for self.serial_position in range(self.pres_indexes.shape[1]):
            # Skip over any zero-padding in the presentation matrix in order to allow variable list length
            if not self.pres_nonzero_mask[self.trial_idx, self.serial_position].all:
                continue
            pres_idx = self.pres_indexes[self.trial_idx, self.serial_position] # [bj] if word-pair, give a pair
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.beta = self.params['beta_enc']
            self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
            self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])  # [bj]

        ##########
        #
        # Pre-recall distractor (if delayed free recall)
        #
        ##########

        if self.mode == 'DFR':
            self.phase = 'distractor'
            self.beta = self.params['beta_distract']
            # Assume source context changes at the same rate as temporal during distractors
            self.beta_source = self.params['beta_distract']
            # By default, treat distractor source as an even mixture of all sources
            # [If your distractors and sources are related, you should modify this so that you can specify distractor source.]
            #source = np.zeros(self.nsources) if self.nsources > 0 else None
            source = self.sources[self.trial_idx, self.serial_position] if self.nsources > 0 else None
            self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
            self.distractor_idx += 1

        ##########
        #
        # Recall period
        #
        ##########

        self.phase = 'recall'
        self.beta = self.params['beta_rec']
        # Follow Polyn et al. (2009) assumption that beta_source is the same at encoding and retrieval
        self.beta_source = self.params['beta_source'] if self.nsources > 0 else 0
        self.rec_items.append([])
        self.rec_times.append([])

        if self.task == 'FR':
            if 'max_recalls' in self.params:  # Limit number of recalls per trial if user has specified a maximum
                self.simulate_recall(time_limit=self.params['rec_time_limit'], max_recalls=self.params['max_recalls'])
            else:
                self.simulate_recall(time_limit=self.params['rec_time_limit'])

        self.trial_idx += 1


    def run_norm_recog_single_sess(self):
        """
        Simulates a session of normal recognition, consisting of the following steps:
        1) Pre-trial context initialization / between-trial distractor
        2) Item presentation as encoding
        3) Pre-recog distractor
        4) Recognition simulation
        [Newly added]
        """
        phases = ['pretrial', 'encoding', 'prerecall', 'recognition']
        for self.trial_idx in range(self.nlists):
            for self.phase in phases:

                if self.phase == 'pretrial':
                    #####
                    # Shift context before start of new list
                    #####
                    # On first trial, present orthogonal item that starts the system;
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    self.serial_position = 0
                    source = None
                    self.beta = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
                    self.beta_source = 1 if self.trial_idx == 0 else self.params['beta_rec_post']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[self.trial_idx, self.serial_position] # [bj] if word-pair, give a pair
                        self.beta = self.params['beta_enc']
                        self.beta_source = 0
                        self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])

                if self.phase == 'prerecall':
                    #####
                    # Shift context before recall phase (e.g., distractor)
                    #####
                    self.beta = self.params['beta_distract']
                    self.beta_source = self.params['beta_distract']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                if self.phase == 'recognition':
                    #####
                    # Simulate recognition
                    #####
                    self.beta = self.params['beta_cue']
                    self.beta_source = 0
                    for self.cue_position in range(len(self.cues_indexes)):
                        cue_idx = self.cues_indexes[self.cue_position]
                        self.simulate_recog(cue_idx)


    def run_conti_recog_single_sess(self):
        """
        Simulates a session of continuous recognition, consisting of the following steps:
        1) Pre-session context initialization / between-trial distractor
        2) Recognition
        3) Item presentation as encoding
        4) Loop step 1-3
        For Hockley's variant, we changes the order of encoding and recognition.
        [Newly added]
        """

        if self.mode == 'Continuous':
            phases = ['pretrial','recognition','encoding']
        elif self.mode == 'Hockley':
            phases = ['pretrial','encoding','recognition']

        for trial_idx in range(self.nlists):
            for self.phase in phases:

                if self.phase == 'pretrial':
                    #####
                    # Shift context before each trial
                    #####
                    # On first trial, present orthogonal item that starts the system
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    self.beta = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.beta_source = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    source = None
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.serial_position = 0

                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                    self.beta = self.params['beta_enc']
                    self.beta_source = 0
                    if np.logical_not(np.isscalar(pres_idx)) and pres_idx[1] == -1:
                        pres_idx = pres_idx[0].astype(int)
                    self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])

                if self.phase == 'recognition':
                    #####
                    # Simulate recognition
                    #####
                    self.beta = self.params['beta_cue']
                    self.beta_source = 0
                    cue_idx = self.cues_indexes[trial_idx]
                    if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                        cue_idx = cue_idx[0].astype(int)
                    self.simulate_recog(cue_idx)

    def run_norm_cr_single_sess(self):
        """
        Simulates a standard session of cued recall, for each list (trial), consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A pre-recall context shift (distractor, beta_distractor = 0 to cancel)
        4) A cued recall period
        [Newly added]
        """

        phases = ['pretrial', 'encoding', 'prerecall', 'recall']
        for trial_idx in range(self.nlists):
            for self.phase in phases:

                if self.phase == 'pretrial':
                    #####
                    # Shift context before each trial
                    #####
                    # On first trial, present orthogonal item that starts the system
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    source = None
                    self.beta = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.beta_source = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.serial_position = 0
                
                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                        self.beta = self.params['beta_enc']
                        self.beta_source = 0
                        self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])

                if self.phase == 'prerecall':
                    #####
                    # Shift context before recall phase (e.g., distractor)
                    #####
                    self.beta = self.params['beta_distract']
                    self.beta_source = self.params['beta_distract']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1

                if self.phase == 'recall':
                    #####
                    # Simulate cued recall
                    #####
                    for test_position in range(self.cues_indexes.shape[1]):
                        cue_idx = self.cues_indexes[trial_idx,test_position]
                        self.beta = self.params['beta_cue']
                        self.beta_source = 0
                        self.simulate_cr(cue_idx)


    def run_success_single_sess(self):
        """
        Simulates a session of successive test, consisting of the following steps:
        1) A pre-trial context shift
        2) A sequence of item (word pair) presentations
        3) A pre-recall context shift [potentially distraction]
        4) Test 1
        5) A pre-recall context shift [potentially distraction]
        6) Test 2
        [Newly added]
        """

        if self.mode == 'Recog-Recog':
            test1 = 'recognition'
            test2 = 'recognition'
        elif self.mode == 'Recog-CR':
            test1 = 'recognition'
            test2 = 'cued recall'
        elif self.mode == 'CR-Recog':
            test1 = 'cued recall'
            test2 = 'recognition'
        elif self.mode == 'CR-CR':
            test1 = 'cued recall'
            test2 = 'cued recall'

        phases = ['pretrial', 'encoding', 'prerecall', test1, 'prerecall', test2]
        for trial_idx in range(self.nlists):
            is_test1 = True

            for self.phase in phases:

                if self.phase == 'pretrial':
                    #####
                    # Shift context before each trial
                    #####
                    # On first trial, present orthogonal item that starts the system
                    # On subsequent trials, present an interlist distractor item
                    # Assume source context changes at same rate as temporal between trials
                    source = None
                    self.beta = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.beta_source = 1 if trial_idx == 0 else self.params['beta_rec_post']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.serial_position = 0

                if self.phase == 'prerecall':
                    #####
                    # Shift context before recall phase (e.g., distractor)
                    #####
                    self.beta = self.params['beta_distract']
                    self.beta_source = self.params['beta_distract']
                    self.present_item(self.distractor_idx, source, update_context=True, update_weights=False)
                    self.distractor_idx += 1
                    self.ret_thresh = np.ones(self.nitems_unique, dtype=np.float32)  # reset threshold

                if self.phase == 'encoding':
                    #####
                    # Present items
                    #####
                    enc_state = 1  # 1 is good, 0 is bad, initially good state
                    for self.serial_position in range(self.pres_indexes.shape[1]):
                        pres_idx = self.pres_indexes[trial_idx, self.serial_position]
                        self.beta = self.params['beta_enc']
                        self.beta_source = 0
                        change_state = self.changestate_rng.choice([False,True], p=[self.var_enc_p,1-self.var_enc_p])
                        if change_state:
                            enc_state = 1 - enc_state
                            if enc_state == 1:
                                self.L_FC.fill(self.params['gamma_fc'])
                                self.L_CF.fill(self.params['gamma_cf'])
                            elif enc_state == 0:
                                self.L_FC.fill(self.params['gamma_fc'] * self.params['bad_enc_ratio'])
                                self.L_CF.fill(self.params['gamma_cf'] * self.params['bad_enc_ratio'])
                        self.present_item(pres_idx, source, update_context=True, update_weights=True, use_new_context=self.params['use_new_context'])
                
                if self.phase == 'recognition':
                    #####
                    # Simulate recognition
                    #####
                    if is_test1:
                        cue_indexes = self.cues_indexes[trial_idx, 0:self.test1_num]
                        is_test1 = False
                    else:
                        cue_indexes = self.cues_indexes[trial_idx, self.test1_num:]
                    for cue_idx in cue_indexes:
                        if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                        self.beta = self.params['beta_cue']
                        self.beta_source = 0
                        self.simulate_recog(cue_idx)  # can be pair, can be scalar
                    
                if self.phase == 'cued recall':
                    #####
                    # Simulate cued recall
                    #####
                    if is_test1:
                        cue_indexes = self.cues_indexes[trial_idx, 0:self.test1_num]
                        is_test1 = False
                    else:
                        cue_indexes = self.cues_indexes[trial_idx, self.test1_num:]
                    for cue_idx in cue_indexes:
                        if np.logical_not(np.isscalar(cue_idx)) and cue_idx[1] == -1:
                            cue_idx = cue_idx[0].astype(int)
                        self.beta = self.params['beta_cue']
                        self.beta_source = 0
                        self.simulate_cr(cue_idx)  # should be scalar


##########
#
# Code to load data and run model
#
##########

def make_params(source_coding=False):
    """
    Returns a dictionary containing all parameters that need to be defined in order for CMR to run. Can be used as a template for the "params" input. [Modified]

    :param source_coding: If True, parameter dictionary will contain the parameters required for the source coding version of the model. If False, the dictionary will only condain parameters required for the base version of the model.
    """
    param_dict = {
        # Beta parameters
        'beta_enc': None,  # Beta encoding
        'beta_rec': None,  # Beta recall
        'beta_cue': None,  # [bj] Beta for cue
        'beta_rec_post': None,  # Beta post-recall
        'beta_distract': None,  # Beta for distractor task

        # Primacy and semantic scaling
        'phi_s': None,
        'phi_d': None,
        's_cf': None,  # Semantic scaling in context-to-feature associations
        's_fc': 0,  # Semantic scaling in feature-to-context associations (Defaults to 0)

        # Recall parameters
        'kappa': None,
        'eta': None,
        'omega': None,
        'alpha': None,
        'lamb': None,
        'c_thresh': None,
        'c_thresh_itm': None,  # [bj] Threshold for item recognition
        'c_thresh_assoc': None,  # [bj] Threshold for associative recognition
        'd_assoc': None,  # [bj] Direct association, not used in the paper

        # Timing & recall settings
        'rec_time_limit': 60000.,  # Duration of recall period (in ms) (Defaults to 60000)
        'dt': 10,  # Number of milliseconds to simulate in each loop of the accumulator (Defaults to 10)
        'nitems_in_accumulator': 50,  # Number of items in accumulator (Defaults to 50)
        'max_recalls': 50,  # Maximum recalls allowed per trial (Defaults to 50)
        'learn_while_retrieving': False,  # Whether associations should be learned during recall (Defaults to False)
        'use_new_context': False, # [bj] Whether to use updated context for learning (Defaults to False to be consistent with CMR2, always True in the paper)

        # [bj] Elevated-attention parameters for WFE
        'psi_s': None,
        'psi_c': None,

        # [bj] Criteria-shift parameters for WFE
        'c_s': None,

        # [bj] Retrieval variability
        'thresh_sigma': None,

        # [bj] Items that should not be recalled
        'ban_recall': None,

        # [bj] Encoding variability, not used in the paper
        'var_enc': 1,
        'bad_enc_ratio': 1,

        # [bj] Parameters for exponential RT in recognition, not used in the paper
        'a': None,
        'b': None,
    }

    # If not using source coding, set up 2 associative scaling parameters (gamma)
    if not source_coding:
        param_dict['gamma_fc'] = None  # Gamma FC
        param_dict['gamma_cf'] = None  # Gamma CF

    # If using source coding, add an extra beta parameter and set up 8 associative scaling parameters
    else:
        param_dict['beta_source'] = None  # Beta source

        param_dict['L_FC_tftc'] = None  # Scale of items reinstating past temporal contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sftc'] = 0  # Scale of sources reinstating past temporal contexts (Defaults to 0)
        param_dict['L_FC_tfsc'] = None  # Scale of items reinstating past source contexts (Recommend setting to gamma FC)
        param_dict['L_FC_sfsc'] = 0  # Scale of sources reinstating past source contexts (Defaults to 0)

        param_dict['L_CF_tctf'] = None  # Scale of temporal context cueing past items (Recommend setting to gamma CF)
        param_dict['L_CF_sctf'] = None  # Scale of source context cueing past items (Recommend setting to gamma CF or fitting as gamma source)
        param_dict['L_CF_tcsf'] = 0  # Scale of temporal context cueing past sources (Defaults to 0, since model does not recall sources)
        param_dict['L_CF_scsf'] = 0  # Scale of source context cueing past sources (Defaults to 0, since model does not recall sources)

    return param_dict

def make_default_params():
    """
    Returns a dictionary containing all parameters that need to be defined in order for CMR to run, with default value. [Newly added]
    """
    param_dict = make_params()
    param_dict.update(
        beta_enc = 0.5,
        beta_rec = 0.5,
        beta_cue = 0.5,
        beta_rec_post = 0.5,
        phi_s = 2,
        phi_d = 0.5,
        s_cf = 0,
        s_fc = 0,
        kappa = 0.5,
        eta = 0.5,
        omega = 5,
        alpha = 1,
        c_thresh = 0.5,
        c_thresh_itm = 0.5,
        c_thresh_assoc = 0.5,
        d_assoc = 1,
        lamb = 0.5,
        gamma_fc = 0.5,
        gamma_cf = 0.5,
        psi_s = 0,
        psi_c = 1,
        c_s = 0,
        thresh_sigma = 0,
        var_enc = 1,
        bad_enc_ratio = 1,
        a = 2800,
        b = 20,
    )

    return param_dict


def load_pres(path):
    """
    Loads matrix of presented items from a .txt file, a .json behavioral data, file, or a .mat behavioral data file. Uses numpy's loadtxt function, json's load function, or scipy's loadmat function, respectively. [Unchanged from CMR2]

    :param path: The path to a .txt, .json, or .mat file containing a matrix where item (i, j) is the jth word presented on trial i.

    :returns: A 2D array of presented items.
    """
    if os.path.splitext(path) == '.txt':
        data = np.loadtxt(path)
    elif os.path.splitext(path) == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
            data = data['pres_nos'] if 'pres_nos' in data else data['pres_itemnos']
    elif os.path.splitext(path) == '.mat':
        data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)['data'].pres_itemnos
    else:
        raise ValueError('Can only load presented items from .txt, .json, and .mat formats.')
    return np.atleast_2d(data)


def split_data(pres_mat, identifiers, source_mat=None):
    """
    If data from multiple subjects or sessions are in one matrix, separate out the data into separate presentation and source matrices for each unique identifier. [Unchanged from CMR2]

    :param pres_mat: A 2D array of presented items from multiple consolidated subjects or sessions.
    :param identifiers: A 1D array with length equal to the number of rows in pres_mat, where entry i identifies the subject/session/etc. to which row i of the presentation matrix belongs.
    :param source_mat: (Optional) A trials x serial positions x nsources array of source information for each presented item in pres_mat.

    :returns: A list of presented item matrices (one matrix per unique identifier), an array of the unique identifiers, and a list of source information matrices (one matrix per subject, None if no source_mat provided).
    """
    # Make sure input matrices are numpy arrays
    pres_mat = np.array(pres_mat)
    if source_mat is not None:
        source_mat = np.atleast_3d(source_mat)

    # Get list of unique IDs
    unique_ids = np.unique(identifiers)

    # Split data up by each unique identifier
    data = []
    sources = None if source_mat is None else []
    for i in unique_ids:
        mask = identifiers == i
        data.append(pres_mat[mask, :])
        if source_mat is not None:
            sources.append(source_mat[mask, :, :])

    return data, unique_ids, sources


def run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates a single session of free recall using the specified parameter set. [Unchanged from CMR2]

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored, allowing zero-padding for varying list lengths.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: 3D array of source features for each presented word if not None. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i.
    :param mode: String indicating the type of free recall to simulate. 'IFR' for immediate free recall or 'DFR' for delayed recall.

    :returns: Two 2D arrays. The first contains the ID numbers of items the model recalled on each trial. The second contains the response times of each item relative to the start of the recall period.
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


def run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR'):
    """
    Simulates multiple sessions of free recall using a single set of parameters. [Unchanged from CMR2]

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param pres_mat: 2D array specifying the ID numbers of words presented to the model on each trial. Row i, column j holds the ID number of the jth word on the ith trial. ID numbers range from 1 to N (number of words in sem_mat). 0s are treated as padding and ignored, allowing zero-padding for varying list lengths.
    :param identifiers: 1D array of session numbers, subject IDs, or other values indicating how the rows/trials in pres_mat and source_mat should be divided into sessions. For example, to simulate two four-trial sessions, set identifiers to np.array([0, 0, 0, 0, 1, 1, 1, 1]), indicating the latter four trials are from a different session than the first four.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: 3D array of source features for each presented word if not None. One row per trial, one column per serial position, and the third dimension for the number of source features. Cell (i, j, k) contains the kth source feature of the jth item on list i.
    :param mode: String indicating the type of free recall to simulate. 'IFR' for immediate free recall or 'DFR' for delayed recall.

    :returns: Two 2D arrays. The first contains the ID numbers of items the model recalled on each trial. The second contains the response times of each item relative to the start of the recall period.
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


def run_norm_recog_multi_sess(params, df_study, df_test, sem_mat, source_mat=None):
    """
    Simulates multiple sessions of normal recognition (recognition after studying a list of items) using a single set of parameters. Only item recognition for now. [Newly added]

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df_study: DataFrame containing the study list with columns "session" and "itemno".
    :param df_test: DataFrame containing the test list with columns "session" and "itemno".
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).

    :returns: DataFrame with columns "session", "list", and "test_itemno" in df_test plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for the test probe).
    """
    now_test = time.time()
    task = 'Recog'
    mode = 'Final'

    sessions = np.unique(df_study.session)
    df_thin = df_test[['session', 'itemno']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)

    for sess in tqdm(sessions):
        # extarct the session data
        pres_mat = df_study.loc[df_study.session==sess, 'itemno'].to_numpy()
        pres_mat = np.reshape(pres_mat,(1, len(pres_mat)))
        cue_mat = df_thin.loc[df_thin.session==sess, 'itemno'].to_numpy()

        # run CMR for each session
        cmr = CMR(params, pres_mat, sem_mat, source_mat=source_mat, rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_norm_recog_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))

        df_thin.loc[df_thin.session==sess, ['s_resp', 's_rt', 'csim']] = result

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin

def run_conti_recog_multi_sess(params, df, sem_mat, source_mat=None, mode='Continuous'):
    """
    Simulates multiple sessions of continuous recognition using a single set of parameters. [Newly added]

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df: DataFrame containing the study list with columns "session", "position", "study_itemno1", "study_itemno2", "test_itemno1", and "test_itemno2". For item recognition, "study_itemno2" and "test_itemno2" should be -1.
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).
    :param mode: String indicating the type of continuous recognition to simulate. Set "Continuous" for the standard continuous recognition paradigm, or "Hockley" for the Hockley's variant.

    :returns: Dataframe with columns "session", "position", "study_itemno1", "study_itemno2", "test_itemno1", and "test_itemno2" in df plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for the test probe).
    """
    now_test = time.time()
    task = 'Recog'

    sessions = np.unique(df.session)
    df_thin = df[['session', 'position', 'study_itemno1', 'study_itemno2', 'test_itemno1', 'test_itemno2']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)

    for sess in tqdm(sessions):
        # extarct the session data
        pres_mat = df_thin.loc[df_thin.session == sess, ['study_itemno1', 'study_itemno2']].to_numpy()
        pres_mat = np.reshape(pres_mat, (len(pres_mat), 1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, ['test_itemno1', 'test_itemno2']].to_numpy()

        # run CMR for each session
        cmr = CMR(params, pres_mat, sem_mat, source_mat=source_mat, rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_conti_recog_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))

        df_thin.loc[df_thin.session==sess, ['s_resp', 's_rt', 'csim']] = result

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin

def run_norm_cr_multi_sess(params, df_study, df_test, sem_mat, source_mat=None):
    """
    Simulates multiple sessions of cued recall using a single set of parameters. [Newly added]

    :param params: Dictionary of model parameters and settings for the simulation. Use CMR_IA.make_params() to get a template dictionary.
    :param df_study: DataFrame containing the study list with columns "session", "list", "study_itemno1", and "study_itemno2".
    :param df_test: DataFrame containing the test list with columns "session", "list", and "test_itemno".    
    :param sem_mat: 2D array of pairwise semantic similarities between all words in the word pool. The order of words must match the word ID numbers, with scores for word k located along row k-1 and column k-1.
    :param source_mat: If None, source coding will not be used (as in the paper).
    
    :returns: 
    - df_thin: Dataframe with columns "session", "list", and "test_itemno" in df_test plus three additional columns: "s_resp" (simulated response), "s_rt" (simulated reaction time, not used in the paper), and "c_sim" (context similarity for intrusion filtering).
    - f_in_acc: List of Arrays of f_in for cued recalls (for developers).
    - f_in_dif: List of Arrays of f_in minus retrieval threshold for cued recalls (for developers).
    """
    now_test = time.time()
    task = 'CR'
    mode = 'Final'

    sessions = np.unique(df_study.session)
    list_num = len(np.unique(df_study.list))
    df_thin = df_test[['session', 'list', 'test_itemno']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan)
    f_in = []
    f_dif = []

    for sess in tqdm(sessions):
        # extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ['study_itemno1', 'study_itemno2']].to_numpy()
        pres_mat = np.reshape(pres_mat, (list_num, -1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, 'test_itemno'].to_numpy()
        cue_mat = np.reshape(cue_mat, (list_num, -1))

        # run CMR for each session
        cmr = CMR(params, pres_mat, sem_mat, source_mat=source_mat, rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode)
        cmr.run_norm_cr_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))
        df_thin.loc[df_thin.session==sess, ['s_resp', 's_rt', 'csim']] = result
        f_in.append(cmr.f_in_acc)
        f_dif.append(cmr.f_in_dif)

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin, f_in, f_dif

def run_success_multi_sess(params, df_study, df_test, sem_mat, source_mat=None, mode='Recog-CR'):
    """
    Simulates multiple sessions of sucessitve tests using a single set of parameters. [Newly added]

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
    """
    now_test = time.time()
    task = "Success"

    sessions = np.unique(df_study.session)
    list_num = len(np.unique(df_study.list))
    df_thin = df_test[['session', 'list', 'test_itemno1', 'test_itemno2']]
    df_thin = df_thin.assign(s_resp=np.nan, s_rt=np.nan, csim=np.nan)
    f_in = []
    f_dif = []
    test1_num = sum(df_test.query("session == 0 and list == 0").test == 1)

    for sess in tqdm(sessions):
        # extarct the session data
        pres_mat = df_study.loc[df_study.session == sess, ['study_itemno1', 'study_itemno2']].to_numpy()
        pres_mat = np.reshape(pres_mat, (list_num, -1, 2))
        cue_mat = df_thin.loc[df_thin.session == sess, ['test_itemno1', 'test_itemno2']].to_numpy()
        cue_mat = np.reshape(cue_mat, (list_num, -1, 2))

        # run CMR for each session
        cmr = CMR(params, pres_mat, sem_mat, source_mat=source_mat, rec_mat=None, ffr_mat=None, cue_mat=cue_mat, task=task, mode=mode, test1_num=test1_num)
        cmr.run_success_single_sess()

        recs = cmr.rec_items
        rts = cmr.rec_times
        csims = cmr.recog_similarity
        result = np.column_stack((recs,rts,csims))
        df_thin.loc[df_thin.session==sess, ['s_resp','s_rt','csim']] = result
        f_in.append(cmr.f_in_acc)
        f_dif.append(cmr.f_in_dif)

    print("CMR Time: " + str(time.time() - now_test))

    return df_thin, f_in, f_dif