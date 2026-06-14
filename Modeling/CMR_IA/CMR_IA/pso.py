import os
import sys
import time
import errno
import functools
import numpy as np
import pickle as pkl
from CMR_IA.fitting import make_boundary, obj_func


def make_noise(S, max_iter, lb, ub, path):
    """
    Make the noise matrices ahead of time for the particle swarm so that all
    parallel instances perform the same operations on each parameter set.

    :param S: Particle swarm size.
    :param max_iter: Number of iterations to run.
    :param lb: Lower bounds of the parameter space.
    :param ub: Upper bounds of the parameter space.
    :param path: Directory path to write noise files into.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    D = len(lb)

    # Initialize particle locations
    try:
        f = os.open(path + 'rx', flags)
        os.close(f)
        rx = np.random.uniform(size=(S, D))
        lb_mat = np.atleast_2d(lb).repeat(S, axis=0)
        ub_mat = np.atleast_2d(ub).repeat(S, axis=0)
        rx = lb_mat + rx * (ub_mat - lb_mat)
        np.savetxt(path + 'rx', rx)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # Create r1 through r4 files for each iteration
    for it in range(1, max_iter + 1):
        for i in range(1, 5):
            try:
                f = os.open(path + 'r%i_iter%i' % (i, it), flags)
                os.close(f)
                r = np.random.uniform(size=(S, D))
                np.savetxt(path + 'r%i_iter%i' % (i, it), r)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise


def pso(func, lb, ub, df_study, df_test, sem_mat, sources, swarmsize=100,
        omega_min=.8, omega_max=.8, d_omega=.1, c1=2, c2=2, c3=0.5, c4=0.5, R=1,
        c2_min=.5, c2_max=2.5, hard_bounds=False, maxiter=100, algorithm='pso',
        optfile=None, outdir='outfiles/', noise_dir='noise_files/'):
    """
    Runs particle swarm optimization (PSO).

    Parameters
    ==========
    func : callable
        Objective function to minimize. Signature:
        func(param_vec, df_study, df_test, sem_mat, sources) -> (err, stats)
    lb : array
        Lower bounds of the parameter space.
    ub : array
        Upper bounds of the parameter space.
    df_study, df_test : DataFrames
        Study and test data passed through to func.
    sem_mat : array
        Semantic similarity matrix passed through to func.
    sources : array or None
        Source information passed through to func.

    Optional
    ========
    swarmsize : int
        Number of particles in the swarm (Default: 100)
    omega_min : scalar
        Minimum inertia weight (Default: 0.8)
    omega_max : scalar
        Maximum inertia weight (Default: 0.8)
    d_omega : scalar
        Delta omega for apso6 algorithm (Default: 0.1)
    c1 : scalar
        Personal acceleration constant (Default: 2.0)
    c2 : scalar
        Social acceleration constant (Default: 2.0)
    c3 : scalar
        Avoidance of personal worst constant (Default: 0.5)
    c4 : scalar
        Avoidance of global worst constant (Default: 0.5)
    R : scalar
        Max velocity as fraction of dimension range (Default: 1)
    c2_min, c2_max : scalar
        Bounds for adaptive social acceleration (Default: 0.5, 2.5)
    hard_bounds : bool
        Whether to enforce hard bounds (Default: False)
    maxiter : int
        Maximum number of iterations (Default: 100)
    algorithm : string
        PSO variant: 'pso', 'pso2', 'sapso', 'dpso', 'cpso', 'npso', 'apso6', 'awl'
    optfile : string or None
        Path to file with a previous run's best parameters to warm-start from.
    outdir : string
        Directory for output files (Default: 'outfiles/')
    noise_dir : string
        Directory for pre-generated noise files (Default: 'noise_files/')

    Returns
    =======
    gb : array
        Best-fitting parameter set found.
    fgb : scalar
        Fitness score of the best-fitting parameter set.
    """
    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    assert hasattr(func, '__call__'), 'Invalid function handle'

    lb = np.array(lb)
    ub = np.array(ub)
    unused_dim = ub == lb
    algorithm = algorithm.lower()
    S = swarmsize
    D = len(lb)

    fgb = np.inf
    fgw = -np.inf
    fpb = np.full(S, np.inf)
    fpw = np.full(S, -np.inf)
    pb = np.full((S, D), np.nan)
    pw = np.full((S, D), np.nan)

    if isinstance(optfile, str) and os.path.exists(optfile):
        old_best = np.loadtxt(optfile)
        gb = old_best[:-1]
        fgb = old_best[-1]
        print('Loaded best known parameter location from file:', gb)
        print('Best known parameter RMSD:', fgb)

    vhigh = (ub - lb) * R
    vlow = -1 * vhigh

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    for it in range(1, maxiter + 1):
        print('Starting PSO iteration %s...' % it)

        # Update particle positions & velocities
        if it == 1:
            print('Initializing particle locations and velocities...')
            x = np.loadtxt(noise_dir + 'rx')
            v = np.zeros((S, D))

        else:
            print('Updating particle locations and velocities...')

            r1 = np.loadtxt(noise_dir + 'r1_iter' + str(it))
            r2 = np.loadtxt(noise_dir + 'r2_iter' + str(it))
            r3 = np.loadtxt(noise_dir + 'r3_iter' + str(it))
            r4 = np.loadtxt(noise_dir + 'r4_iter' + str(it))

            while True:
                try:
                    x = np.loadtxt(outdir + str(it - 1) + 'xfile.txt')
                    v = np.loadtxt(outdir + str(it - 1) + 'vfile.txt')
                    pb = np.loadtxt(outdir + str(it - 1) + 'pfile.txt')
                    pw = np.loadtxt(outdir + str(it - 1) + 'pwfile.txt')
                except ValueError:
                    continue
                if (len(x) == S) and (len(v) == S) and (len(pb) == S) and (len(pw) == S):
                    break
                else:
                    time.sleep(2)

            if algorithm == 'pso':
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            elif algorithm == 'pso2':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega_max * (v[i, :] + t1[i, :] + t2)

            elif algorithm == 'sapso':
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    rd = (fx[i] - gb) / fx[i]
                    omega = (omega_max - omega_min) * (1 - np.cos(.5 * np.pi * rd)) + omega_min
                    c2 = (c2_max - c2_min) * (1 - np.cos(.5 * np.pi * rd)) + c2_min
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            elif algorithm == 'dpso':
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    if it == 2:
                        c2 = c2_min
                    else:
                        grade = (fx.max() - fx[i]) / (fx.max() - fx.min()) if fx.max() != fx.min() else 1
                        c2 = c2_min + (c2_max - c2_min) * grade
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2
                    for dim in range(D):
                        if r3[i, dim] < 1. / (S * D):
                            v[i, dim] = vlow[dim] + r4[i, dim] * (vhigh[dim] - vlow[dim])

            elif algorithm == 'cpso':
                if it == 2:
                    omega = r3
                else:
                    omega = 4 * omega * (1 - omega)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega[i, :] * v[i, :] + t1[i, :] + t2

            elif algorithm == 'npso':
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * (x - pw)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * (x[i, :] - gw)
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2 + t3[i, :] + t4

            elif algorithm == 'apso6':
                if it == 2:
                    omega = omega_max
                avg_v = np.mean(np.mean(np.abs(v), axis=0)[~unused_dim] / (ub - lb)[~unused_dim])
                opt_v = .5 * (1 + np.cos(np.pi * (it - 1) / (.95 * maxiter))) / 2
                omega = max(omega - d_omega, omega_min) if avg_v >= opt_v else min(omega + d_omega, omega_max)
                t1 = c1 * r1 * (pb - x)
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    v[i, :] = omega * v[i, :] + t1[i, :] + t2

            elif algorithm == 'awl':
                omega = omega_max - (omega_max - omega_min) * (it - 1) / (maxiter - 1)
                t1 = c1 * r1 * (pb - x)
                t3 = c3 * r3 * t1 / (1 + np.abs(x - pw))
                for i in range(S):
                    t2 = c2 * r2[i, :] * (gb - x[i, :])
                    t4 = c4 * r4[i, :] * t2 / (1 + np.abs(x[i, :] - gw))
                    v[i, :] = omega * (v[i, :] + t1[i, :] + t2 + t3[i, :] + t4)

            else:
                raise ValueError('Unrecognized PSO algorithm "%s" -- see docstring '
                                 'for list of supported algorithms.' % algorithm)

            for i in range(S):
                mask1 = v[i, :] < vlow
                mask2 = v[i, :] > vhigh
                v[i, mask1] = vlow[mask1]
                v[i, mask2] = vhigh[mask2]

            x += v

            if hard_bounds:
                for i in range(S):
                    mask1 = x[i, :] < lb
                    mask2 = x[i, :] > ub
                    x[i, mask1] = lb[mask1]
                    x[i, mask2] = ub[mask2]
                    v[i, mask1] = 0
                    v[i, mask2] = 0

        # Test model for each particle
        if os.path.exists(outdir + 'err_iter' + str(it)):
            while True:
                fx = np.loadtxt(outdir + 'err_iter' + str(it))
                if len(fx) == S:
                    break
                else:
                    time.sleep(2)

        else:
            for i in range(S):
                oob = np.any((x[i, :] < lb) | (x[i, :] > ub))
                match_file = outdir + str(it) + 'tempfile' + str(i) + '.txt'
                try:
                    fd = os.open(match_file, flags)

                    if not oob:
                        print('Running model for particle %s...' % i)
                        err, stats = func(x[i, :], df_study, df_test, sem_mat, sources)
                        print('Model finished with a fitness score of %s!' % err)
                    else:
                        print('Skipping out-of-bounds particle %s...' % i)
                        err = np.nan
                        stats = {}

                    with open(outdir + str(it) + 'data' + str(i) + '.pkl', 'wb') as f:
                        pkl.dump(stats, f, 2)

                    file_input = str(err)
                    os.write(fd, file_input.encode())
                    os.close(fd)

                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print('Model for particle %s already complete! Skipping...' % i)
                        continue
                    else:
                        raise

            while True:
                for i in range(S):
                    path = outdir + '%stempfile%s.txt' % (it, i)
                    if not (os.path.exists(path) and os.path.getsize(path) > 0.0):
                        break
                else:
                    break
                time.sleep(2)

            fx = np.zeros(S)
            for i in range(S):
                fx[i] = np.loadtxt(outdir + '%stempfile%s.txt' % (it, i))

        # Search for new best/worst scores
        print('Checking for new best/worst particle positions...')
        for i in range(S):
            if np.isnan(fx[i]):
                continue

            if fx[i] < fpb[i]:
                print('New best location for particle %s!' % i)
                pb[i, :] = x[i, :].copy()
                fpb[i] = fx[i]

                if fx[i] < fgb:
                    print('Particle %s found a new best global location!' % i)
                    gb = x[i, :].copy()
                    fgb = fx[i]

            if fx[i] > fpw[i]:
                print('New worst location for particle %s!' % i)
                pw[i, :] = x[i, :].copy()
                fpw[i] = fx[i]

                if fx[i] > fgw:
                    print('Particle %s found a new worst global location!' % i)
                    gw = x[i, :].copy()
                    fgw = fx[i]

        # Save results of iteration
        param_files = [outdir + str(it) + 'xfile.txt', outdir + str(it) + 'pfile.txt',
                       outdir + str(it) + 'pwfile.txt', outdir + str(it) + 'vfile.txt',
                       outdir + 'err_iter' + str(it)]
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


def run_pso(simu_name, df_study, df_test, sem_mat, sources=None, outdir="outfiles/", noise_dir="noise_files/"):
    """
    Set up and run PSO for a given simulation.

    :param simu_name: Simulation name key (e.g. "S2", "1", "6b").
    :param df_study: Study DataFrame (or None for continuous-recognition sims).
    :param df_test: Test DataFrame.
    :param sem_mat: Semantic similarity matrix.
    :param sources: Source information.
    :param outdir: Directory for PSO output files.
    :param noise_dir: Directory for pre-generated noise files.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(noise_dir):
        os.makedirs(noise_dir)

    # Set PSO parameters
    alg = 'pso2'
    swarmsize = 500
    n_iter = 200
    omega_min = .72984 if alg in ('pso2', 'awl') else .3 if alg == 'apso6' else .4
    omega_max = .72984 if alg in ('pso2', 'awl') else .9
    d_omega = .1
    c1 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496172
    c2 = 2.05 if alg == 'pso2' else 1.845 if alg == 'awl' else 1.496172
    c3 = .205
    c4 = .205
    c2_min = .5
    c2_max = 2.5
    R = 1
    hard_bounds = False

    # Set parameter boundaries and bind simu_name into the objective function
    lb, ub, _ = make_boundary(simu_name)
    func = functools.partial(obj_func, simu_name=simu_name)
    print('Generating noise files...')
    make_noise(swarmsize, n_iter, lb, ub, noise_dir)

    print('Initiating particle swarm optimization...')
    start_time = time.time()
    xopt, fopt = pso(func, lb, ub, df_study, df_test, sem_mat, sources,
                     swarmsize=swarmsize, maxiter=n_iter,
                     omega_min=omega_min, omega_max=omega_max, d_omega=d_omega,
                     c1=c1, c2=c2, c3=c3, c4=c4, R=R,
                     c2_min=c2_min, c2_max=c2_max, algorithm=alg,
                     optfile=None, hard_bounds=hard_bounds,
                     outdir=outdir, noise_dir=noise_dir)

    print(fopt, xopt)
    print("Run time: " + str(time.time() - start_time))
    sys.stdout.flush()

    np.savetxt(outdir + 'xoptb.txt', xopt, delimiter=',', fmt='%f')
