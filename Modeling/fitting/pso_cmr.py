# import mkl
# mkl.set_num_threads(1)
import os
import sys
import time
import json
import errno
import numpy as np
import pickle as pkl
from glob import glob
from noise_maker_pso import make_noise
from optimization_utils import make_boundary
from object_funcs import obj_func_S1, obj_func_S2, obj_func_6b, obj_func_3, obj_func_1, obj_func_4ctrl, obj_func_4shift

# Set where you want your noise files and output files to be saved
global OUTDIR
global NOISE_DIR
OUTDIR = 'outfiles/'
NOISE_DIR = 'noise_files/'

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)
if not os.path.exists(NOISE_DIR):
    os.makedirs(NOISE_DIR)

"""
Dependencies: CMR_IA plus all the package imports above.

Modified by Beige Jin. You will need to modify run_pso() and the __main__ script to work for your own projects.
"""


def pso(func, lb, ub, df_study, df_test, sem_mat, sources, swarmsize=100,
        omega_min=.8, omega_max=.8, d_omega=.1, c1=2, c2=2, c3=0.5, c4=0.5, R=1,
        c2_min=.5, c2_max=2.5, hard_bounds=False, maxiter=100, algorithm='pso', optfile=None, sim_name=''):
    """
    Runs particle swarm optimization (PSO). Can be run using a variety of
    particle swarm algorithms, as described below.

    Parameters
    ==========
    func : function
        The objective function to be minimized. Usually a function that runs
        CMR2 to simulate a dataset and evaluate its fit to empirical data.
    lb : array
        The lower bounds of each dimension of the parameter space.
    ub : array
        The upper bounds of each dimension of the parameter space.
    data_pres : array
        A trials x items matrix of the ID numbers of items that will be
        presented to CMR2.
    identifiers: array
        An array of subject/session identifiers, used for separating data_pres
        into the presented item matrices for multiple sessions.
    w2v: array
        An items x items matrix of word2vec (or other) semantic similarity scores
        to be passed on to CMR2.
    sources: array
        A trials x items x feature array of source information to be passed to CMR2.
    target_stats: dictionary
        A dictionary containing the empirical recall performance stats that will
        be used by the objective function to determine the difference between
        CMR2's performance and actual human performance.

    Optional
    ========
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega_min : scalar
        Minimum value of the particles' inertia. Except in certain special PSO
        algorithms (e.g. chaotic particle swarm optimization), inertia will decrease
        linearly from omega_max to omega_min over the course of the optimization
        process. If omega_min == omega_max, inertia will be constant. (Default: 0.8)
    omega_max : scalar
        Maximum value of the particles' inertia. Except in certain special PSO
        algorithms (e.g. chaotic particle swarm optimization), inertia will decrease
        linearly from omega_max to omega_min over the course of the optimization
        process. If omega_min == omega_max, inertia will be constant. (Default: 0.8)
    d_omega : scalar
        The value by which to increase/decrease omega at each iteration in the APSO-VI algorithm.
    c1 : scalar
        Scaling factor to move towards the particle's best known position
        (Default: 2.0)
    c2 : scalar
        Scaling factor to move towards the swarm's best known position
        (Default: 2.0)
    c3 : scalar
        Scaling factor to move away from the particle's worst known position
        (Default: 0.5)
    c4 : scalar
        Scaling factor to move away from the swarm's worst known position
        (Default: 0.5)
    R : scalar
        Velocity will be capped such that R is the greatest fraction of a dimension
        that can be traveled over a single iteration. For example, if R=.5 the max
        velocity on each dimension is 50% of that dimension's range. (Default: 1)
    c2_min : scalar
        Minimum value of the social acceleration constant (c2) during SAPSO and DPSO.
        (Default: 0.5)
    c2_max :
        Maximum value of the social acceleration constant (c2) during SAPSO and DPSO.
        (Default: 2.5)
    hard_bounds :
        If True, particles cannot fly outside of the parameter space, and will stop upon hitting an edge. If False,
        particles will be allowed to fly outside of the parameter space, but will not be evaluated if they do.
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    algorithm : string
        Specifies which particle swarm optimization algorithm to use. Options are:
        'pso' : Particle Swarm Optimization with inertia (Kennedy & Eberhart, 1995; Shi & Eberhart, 1998)
        'pso2 : Particle Swarm Optimization with constriction (Eberhart & Shi, 2000; Clerc & Kennedy, 2002)
        'sapso' : Self-Adaptive Particle Swarm Optimization (Wu & Zhou, 2007)
        'dpso' : Dispersed Particle Swarm Optimization (Cai, Cui, Zeng, & Tan, 2008)
        'cpso' : Chaotic Particle Swarm Optimization (Chaunwen & Bompard, 2005)
        'npso' : New Particle Swarm Optimization (Selvakumar & Thanushkodi, 2007)
        'apso6' : Adaptive Particle Swarm Optimization VI (Xu, 2013)
        'awl' : Particle Swarm Optimization with Avoidance of Worst Location (Mason & Howley, 2016)
    optfile : string or None
        If a file path is provided, the best known parameter set and score will be initialized as the parameters listed
        in that file. This allows a new particle swarm to begin with knowledge of a previous swarm's best position.

    Returns
    =======
    gb : array
        The swarm's best known position, i.e. the best-fitting parameter set identified.
    fgb : scalar
        The goodness-of-fit score of the best-fitting parameter set identified.

    """
    global OUTDIR
    global NOISE_DIR
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'

    ##########
    #
    # Initialization
    #
    ##########

    lb = np.array(lb)
    ub = np.array(ub)
    unused_dim = ub == lb
    algorithm = algorithm.lower()
    S = swarmsize  # Number of particles
    D = len(lb)  # Number of dimensions

    # Initialize global best and worst fitness values such that any new value will replace them
    fgb = np.inf
    fgw = -np.inf

    # Initialize particle best and worst fitness values such that any new value will replace them
    fpb = np.full(S, np.inf)
    fpw = np.full(S, -np.inf)

    # Initialize best and worst particle positions to NaN
    pb = np.full((S, D), np.nan)  # Best known position of each particle
    pw = np.full((S, D), np.nan)  # Worst known position of each particle

    if isinstance(optfile, str) and os.path.exists(optfile):
        old_best = np.loadtxt(optfile)
        gb = old_best[:-1]
        fgb = old_best[-1]
        print('Loaded best known parameter location from file:', gb)
        print('Best known parameter RMSD:', fgb)

    # Define maximum positive and negative velocities for each dimension
    vhigh = (ub - lb) * R
    vlow = -1 * vhigh

    # os.O_CREAT --> create file if it does not exist
    # os.O_EXCL --> error if create and file exists
    # os.O_WRONLY --> open for writing only
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    ##########
    #
    # PSO
    #
    ##########

    # Run PSO for maxiter iterations
    for it in range(1, maxiter + 1):
        print('Starting PSO iteration %s...' % it)

        ##########
        #
        # Update particle positions & velocities
        #
        ##########

        # If it is the first iteration, load initial particle positions and initialize velocities to zero
        if it == 1:
            print('Initializing particle locations and velocities...')
            x = np.loadtxt(NOISE_DIR + 'rx')
            v = np.zeros((S, D))

        # If it is any iteration beyond the first, update particle positions and velocities
        else:
            print('Updating particle locations and velocities...')

            # Read in the noise files for this iteration
            r1 = np.loadtxt(NOISE_DIR + 'r1_iter' + str(it))
            r2 = np.loadtxt(NOISE_DIR + 'r2_iter' + str(it))
            r3 = np.loadtxt(NOISE_DIR + 'r3_iter' + str(it))
            r4 = np.loadtxt(NOISE_DIR + 'r4_iter' + str(it))

            # Read in the position, best/worst location, & velocity files from previous iteration
            # If the number of values in the file does not match the number of particles, it means
            # another job is in the process of writing that file. Wait for it to finish and then
            # try again after a couple seconds.
            while True:
                try:
                    x = np.loadtxt(OUTDIR + str(it - 1) + 'xfile.txt')
                    v = np.loadtxt(OUTDIR + str(it - 1) + 'vfile.txt')
                    pb = np.loadtxt(OUTDIR + str(it - 1) + 'pfile.txt')
                    pw = np.loadtxt(OUTDIR + str(it - 1) + 'pwfile.txt')
                except ValueError:  # May throw a ValueError if it tries to read one of these files before it is fully written
                    continue
                if (len(x) == S) and (len(v) == S) and (len(pb) == S) and (len(pw) == S):
                    break
                else:
                    time.sleep(2)

            # Update particle positions based on positions, velocities, and scores from last iteration

            # Basic PSO (Shi & Eberhart, 1998)
            if algorithm == 'pso':
                # Linearly decrease inertia over iterations and update velocities
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # Particle swarm with constriction
            elif algorithm == 'pso2':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega_max * (v[i, :] + t1[i, :] + t2)

            # Self-Adaptive PSO (Wu & Zhou, 2007)
            elif algorithm == 'sapso':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    # Calculate relational distance of current particle to best location
                    rd = (fx[i] - gb) / fx[i]

                    # Set inertia and social acceleration based on relational distance
                    # Note that these calculations were slightly modified to use max and min settings
                    omega = (omega_max - omega_min) * (1 - np.cos(.5 * np.pi * rd)) + omega_min
                    c2 = (c2_max - c2_min) * (1 - np.cos(.5 * np.pi * rd)) + c2_min

                    # Update velocities
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # Dispersed PSO (Cai, Cui, Zeng, & Tan, 2008)
            elif algorithm == 'dpso':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    # Set social acceleration based on current particle's performance relative to others
                    # in the current iteration
                    if it == 2:
                        c2 = c2_min
                    else:
                        grade = (fx.max() - fx[i]) / (fx.max() - fx.min()) if fx.max() != fx.min() else 1
                        c2 = c2_min + (c2_max - c2_min) * grade

                    # Update velocities
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

                    # On average, mutate one dimension of one particle per iteration
                    # The new velocity is some random value between vhigh and vlow (slightly modified from original)
                    for dim in range(D):
                        if r3[i, dim] < 1. / (S * D):
                            v[i, dim] = vlow[dim] + r4[i, dim] * (vhigh[dim] - vlow[dim])

            # Chaotic PSO (Chaunwen & Bompard, 2005)
            elif algorithm == 'cpso':
                # Shift inertia on every trial
                if it == 2:
                    omega = r3
                else:
                    omega = 4 * omega * (1 - omega)
                # Update velocities
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega[i, :] * v[i, :] + t1[i, :] + t2

            # New PSO (Selvakumar & Thanushkodi, 2007)
            elif algorithm == 'npso':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)

                # Move towards best locations while moving away from worst locations
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * (x - pw)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * (x[i, :] - gw)
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2 + t3[i, :] + t4

            # Adaptive PSO-VI (Xu, 2013)
            elif algorithm == 'apso6':
                if it == 2:
                    omega = omega_max
                # Get average absolute velocity on each dimension, normalize velocities by parameter ranges, and
                # then average across dimensions to get the average velocity in range [0, 1] (ignore flat dimensions)
                avg_v = np.mean(np.mean(np.abs(v), axis=0)[~unused_dim] / (ub - lb)[~unused_dim])
                # Calculate optimal velocity for current iteration
                opt_v = .5 * (1 + np.cos(np.pi * (it - 1) / (.95 * maxiter))) / 2
                # Adjust inertia to approach optimal velocity
                omega = max(omega - d_omega, omega_min) if avg_v >= opt_v else min(omega + d_omega, omega_max)

                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            # PSO AWL (Mason & Howley, 2016)
            elif algorithm == 'awl':
                # Linearly decrease inertia over iterations
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)

                # Move towards best locations, and move faster if near worst locations
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * t1 / (1 + np.abs(x - pw))
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * t2 / (1 + np.abs(x[i, :] - gw))
                    v[i, :] = omega * (v[i, :] + t1[i, :] + t2 + t3[i, :] + t4)

            else:
                raise ValueError('Unrecognized PSO algorithm "%s" -- see docstring '
                                 'for list of supported algorithms.' % algorithm)

            # Keep velocity within the bounds of [vlow, vhigh]
            for i in range(S):
                mask1 = v[i, :] < vlow
                mask2 = v[i, :] > vhigh
                v[i, mask1] = vlow[mask1]
                v[i, mask2] = vhigh[mask2]

            # Update all particles' positions
            x += v

            # If hard search bounds are enforced, keep the particles within bounds
            if hard_bounds:
                for i in range(S):
                    mask1 = x[i, :] < lb
                    mask2 = x[i, :] > ub
                    x[i, mask1] = lb[mask1]
                    x[i, mask2] = ub[mask2]
                    # If a particle runs into the wall, set its velocity in that dimension to 0 to prevent it from
                    # running into the wall for multiple iterations
                    v[i, mask1] = 0
                    v[i, mask2] = 0

        ##########
        #
        # Test model for each particle
        #
        ##########

        # If the error file for this iteration already exists, load the error values from that rather than from tempfiles
        if os.path.exists(OUTDIR + 'err_iter' + str(it)):
            while True:
                fx = np.loadtxt(OUTDIR + 'err_iter' + str(it))

                # If we got all the values we needed, break out of the loop, otherwise try again shortly
                if len(fx) == S:
                    break
                else:
                    time.sleep(2)

        else:
            # For each particle, test the model with parameters corresponding to that particle's location
            for i in range(S):

                # Determine which particles (if any) have flown out of bounds
                oob = np.any((x[i, :] < lb) | (x[i, :] > ub))

                match_file = OUTDIR + str(it) + 'tempfile' + str(i) + '.txt'
                try:
                    # Try to open the tempfile; if the file for this particle already exists, skip to the next particle
                    fd = os.open(match_file, flags)

                    # Run CMR2 using this particle's position and get out the fitness score (Core)
                    if not oob:
                        print('Running model for particle %s...' % i)
                        err, stats = func(x[i, :], df_study, df_test, sem_mat, sources)
                        print('Model finished with a fitness score of %s!' % err)
                    else:
                        print('Skipping out-of-bounds particle %s...' % i)
                        err = np.nan
                        stats = {}
                    # Save simulated behavioral stats from the particle
                    with open(OUTDIR + str(it) + 'data' + str(i) + '.pkl', 'wb') as f:
                        pkl.dump(stats, f, 2)

                    # Write the particle's fitness score to the tempfile
                    file_input = str(err)
                    os.write(fd, file_input.encode())
                    os.close(fd)

                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print('Model for particle %s already complete! Skipping...' % i)
                        continue
                    else:
                        raise

            # Wait until all parallel jobs have finished running the models from the current iteration
            # Check for each expected tempfile. If any are empty, wait 2 seconds before trying again. If all
            # are finished, break out of the loop and proceed.
            while True:
                for i in range(S):
                    path = OUTDIR + '%stempfile%s.txt' % (it, i)
                    if not (os.path.exists(path) and os.path.getsize(path) > 0.0):
                        break
                else:
                    break
                # sleep 2 seconds before we try again
                time.sleep(2)

            # Load the error values for this iteration from all tempfiles, once we have confirmed they are finished
            fx = np.zeros(S)
            for i in range(S):
                fx[i] = np.loadtxt(OUTDIR + '%stempfile%s.txt' % (it, i))

        ##########
        #
        # Search for new best/worst scores
        #
        ##########

        # Once all particles have finished for this iteration, collect all fitness values from the tempfiles
        # While doing this, check whether any new best and worst positions have been found
        print('Checking for new best/worst particle positions...')
        for i in range(S):

            # Skip particles that were out of bounds, and therefore were not scored
            if np.isnan(fx[i]):
                continue

            # Check whether the particle's current position is better than its previous best location
            if fx[i] < fpb[i]:
                print('New best location for particle %s!' % i)
                pb[i, :] = x[i, :].copy()
                fpb[i] = fx[i]

                # Check whether the particle's current position is better than the previous global best location
                if fx[i] < fgb:
                    print('Particle %s found a new best global location!' % i)
                    gb = x[i, :].copy()
                    fgb = fx[i]

            # Check whether the particle's current position is worse than its previous worst location
            if fx[i] > fpw[i]:
                print('New worst location for particle %s!' % i)
                pw[i, :] = x[i, :].copy()
                fpw[i] = fx[i]

                # Check whether the particle's current position is worse than the previous global worst location
                if fx[i] > fgw:
                    print('Particle %s found a new worst global location!' % i)
                    gw = x[i, :].copy()
                    fgw = fx[i]

        ##########
        #
        # Save results of iteration
        #
        ##########

        # Save the results from the current iteration before moving on to the next
        param_files = [OUTDIR + str(it) + 'xfile.txt', OUTDIR + str(it) + 'pfile.txt',
                       OUTDIR + str(it) + 'pwfile.txt', OUTDIR + str(it) + 'vfile.txt',
                       OUTDIR + 'err_iter' + str(it)]
        param_entries = [x, pb, pw, v, fx]
        for i in range(len(param_entries)):
            try:
                f = os.open(param_files[i], flags)
                os.close(f)
                np.savetxt(param_files[i], param_entries[i])
                print('Saved iteration results to %s!' % param_files[i])
            except OSError as e:
                if e.errno == errno.EEXIST:
                    continue
                else:
                    raise
        print('Iteration %s complete!' % it)

    return gb, fgb


def run_pso(df_study, df_test, sem_mat, sources, sim_name=''):

    global OUTDIR
    global NOISE_DIR    

    # Set PSO parameters
    alg = 'pso2'
    swarmsize = 100
    n_iter = 100
    omega_min = .72984 if alg in ('pso2', 'awl') else .3 if alg == 'apso6' else .4
    omega_max = .72984 if alg in ('pso2', 'awl') else .9
    d_omega = .1  # Delta omega for apso6 algorithm
    c1 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496172  # 1.496172 = 2.05 * .72984
    c2 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496172
    c3 = .205
    c4 = .205
    c2_min = .5
    c2_max = 2.5
    R = 1  # Maximum velocity constraint (max = R * (ub - lb))
    hard_bounds = False
    
    # Set parameter boundaries
    lb, ub, _ = make_boundary(sim_name)
    if sim_name == 'S1':
        func = obj_func_S1
    elif sim_name == 'S2':
        func = obj_func_S2
    elif sim_name == '6b':
        func = obj_func_6b
    elif sim_name == '3':
        func = obj_func_3
    elif sim_name == '1':
        func = obj_func_1
    elif sim_name == '4ctrl':
        func = obj_func_4ctrl
    elif sim_name == '4shift':
        func = obj_func_4shift
    else:
        raise ValueError('No objective function found!')
    
    print('Generating noise files...')
    make_noise(swarmsize, n_iter, lb, ub, NOISE_DIR)

    print('Initiating particle swarm optimization...')
    start_time = time.time()
    xopt, fopt = pso(func, lb, ub, df_study, df_test, sem_mat, sources, swarmsize=swarmsize, maxiter=n_iter,
                     omega_min=omega_min, omega_max=omega_max, d_omega=d_omega, c1=c1, c2=c2, c3=c3, c4=c4, R=R,
                     c2_min=c2_min, c2_max=c2_max, algorithm=alg, optfile=None, hard_bounds=hard_bounds, sim_name=sim_name)

    print(fopt, xopt)
    print("Run time: " + str(time.time() - start_time))
    sys.stdout.flush()

    np.savetxt(OUTDIR + 'xoptb.txt', xopt, delimiter=',', fmt='%f')


if __name__ == "__main__":
    
    # define simulation name
    # SIM = '1'
    # SIM = 'S1'
    # SIM = 'S2'
    # SIM = '3'
    # SIM = '6b'
    # SIM = '4ctrl'
    SIM = '4shift'
    
    # semantic file in general, but could be different for specific simulations
    sem_file = '../../Analysis/wordpools/ltp_FR_similarity_matrix.npy' 

    if SIM == '1':
        
        with open("../../Analysis/simu1_recog_recsim/simu1_data/simu1_design.pkl", 'rb') as inp:
            df_test = pkl.load(inp)
        df_test = df_test.loc[df_test.session < 500]
        df_study = None
        sources = None
        sem_file = '../../Analysis/simu1_recog_recsim/simu1_data/simu1_300_smat.npy'
    
    elif SIM == 'S1':

        with open("../../Analysis/simuS1_recog_cr/simuS1_data/simuS1_design.pkl", 'rb') as inp:
            df_study = pkl.load(inp)
            df_test = pkl.load(inp)
        sources = None

    elif SIM == 'S2':
         
        with open("../../Analysis/simuS2_recog_recog/simuS2_data/simuS2_design.pkl", 'rb') as inp:
            df_study = pkl.load(inp)
            df_test = pkl.load(inp)
        sources = None

    elif SIM == '6b':

        with open("../../Analysis/simu6b_cr_sym/simu6b_data/simu6b_design.pkl", 'rb') as inp:
            df_study = pkl.load(inp)
            df_test = pkl.load(inp)
        df_study = df_study.loc[df_study.session < 100]
        df_test = df_test.loc[df_test.session < 100]
        sources = None
    
    elif SIM == '3':
        
        with open("../../Analysis/simu3_recog_forget/simu3_data/simu3_design.pkl", 'rb') as inp:
            df_test = pkl.load(inp)
        df_test = df_test.loc[df_test.session < 300]
        df_study = None
        sources = None
        
    elif SIM == "4ctrl" or SIM == "4shift":
        
        with open("../../Analysis/simu4_recog_wfe/simu4_data/simu4_design.pkl", 'rb') as inp:
            df_study = pkl.load(inp)
            df_test = pkl.load(inp)
        sources = None
        sem_file = "../../Analysis/simu4_recog_wfe/simu4_data/simu4_smat.npy"

    else:
        raise ValueError('Simulation name not recognized!')
        
    # Load semantic similarity matrix (word2vec)
    sem_mat = np.load(sem_file)

    run_pso(df_study, df_test, sem_mat, sources, sim_name=SIM)
