## MODELING FILES SUMMARY

`CMR_IA/` contains the core CMR-IA code. These include the following:
- `CMR_IA.pyx`: contains the Cython implementation of CMR-IA.
- `setup.py`: a script which can be used to build and install CMR-IA.
- `setup_env.sh`: a script which can be used to create a new conda environment able to run CMR-IA.

`fitting/` contains the code for fitting CMR-IA to data. These include the following:
- `pso_cmr.py`: contains a Python function for running particle swarm optimization on CMR-IA. Includes implementations of many different particle swarm variants (see its docstring for reference).
- `object_funcs.py`: contains some objective functions that compares the model's output to the data.
- `optimization_utils.py`: contains a variety of utility functions used by the optimization algorithms.
- `noise_maker_pso.py`: contains a function used by the PSO algorithm to generate random numbers used in particle initialization and motion.
- `pgo` and `runpyfile.sh`: scripts for running the particle swarm optimization in parallel.

## RUNNING CMR_IA

Once you have imported CMR_IA (see README in the main page), there are basically two approaches you can run simulations. The first is through the provided functions, which automatially initializes a CMR object and simulates a single session or multiple sessions using a given parameter set. This is the recommended approach. Another is to initialize a CMR object manually and then build your own simulation code around it.

CMR_IA is able to simulate many types of memory tasks.

A couple helpful tips before we begin:
- For the `params` input to the functions below, you can use the function `CMR_IA.make_default_params()` to create a dictionary containing all of the parameters with default values. Just update the default values with your own values as needed.
- For the `sem_mat` inputs to the functions below, if your simulation uses one of the wordpools provided in the wordpools subdirectory, you should be able to find .npy file in there containing the semantic similarity matrix for that wordpool (e.g. ltp_FR_similarity_matrix.npy for PEERS wordpool). Simply load it with `np.load()` and input it as your similarity matrix.

### Free Recall

For free recall, CMR_IA keeps the same code as CMR2 from Pazdera & Kahana (2022). We preserve the same function names and the same inputs. As in CMR2, you should input a 2D Array as the presented item matrix. Be very cautious that such input format is different from the recognition and cued recall tasks which is newly developed in CMR-IA.

To create a presented item matrix, you can use the function `CMR_IA.load_pres()` to load the presented item matrix from a variety of data files, including the behavioral matrix files from ltp studies in .json and .mat format.

There are two provided functions for free recall.

~~~
run_cmr2_single_sess(params, pres_mat, sem_mat, source_mat=None, mode='IFR')
~~~

This function simulates a single session with CMR2 and returns two numpy arrays containing the ID numbers of recalled items and the response times of each of those recalls, respectively. See the function's docstring for details on its inputs, and note that you can choose whether to include source featuresin your model and whether to simulate immediate or delayed free recall.

~~~
run_cmr2_multi_sess(params, pres_mat, identifiers, sem_mat, source_mat=None, mode='IFR')
~~~

This function simulates multiple sessions with CMR2, using the same parameter set for all sessions. Like its single-session counterpart it returns two numpy arrays containing the ID numbers of recalled items and the response times of each of those recalls. See the function's docstring for details on its inputs, and again note that you can choose whether to include source features in your model and whether to simulate immediate or delayed free recall.

### Recognition

CMR_IA can simulate two recognition paradigms. The first is the normal recognition paradigm where subjects first memorize a list and then perform the recognition task. The second is the continuous recognition paradigm where subjects memorize and response for each item presented continuously. Importantly, we change the input format to DataFrames that specify session, study item ID, test item ID, etc. See the function's docstring for details on required fields. We believe this change makes the simulation code more intuitive and easier to use.

There are two provided functions responsible for two paradigms respectively. These functions support both single session and multiple sessions. 

~~~
run_norm_recog_multi_sess(params, df_study, df_test, sem_mat, source_mat=None)
~~~

This function simulates multiple sessions of the normal recongition task.

~~~
run_conti_recog_multi_sess(params, df, sem_mat, source_mat=None, mode='Continuous')
~~~

This function simulates multiple sessions of the continuous recognition task. You can change the mode between 'Continuous' and 'Hockley' to simulate different variants of the continuous recognition task.

### Cued Recall

CMR_IA can simulate the cued recall paradigm where subjects first memorize a list of word pairs and then perform the cued recall task. This function supports both single session and multiple sessions.

~~~
run_norm_cr_multi_sess(params, df_study, df_test, sem_mat, source_mat=None)
~~~

### Sucessive Tests

CMR_IA can simulate the successive tests paradigm where subjects first memorize a list of word pairs and then perform two tests successively. These two tests could be any combination of recognition and cued recall. For example, set mode="CR-Recog" to simulate the cued recall followed by recognition. This function supports both single session and multiple sessions.

~~~
run_success_multi_sess(params, df_study, df_test, sem_mat, source_mat=None, mode='Recog-CR')
~~~

### Initialize a CMR Object

This section is for those who want to build its own simulation code rather than using provided functions. The inputs when creating a CMR object would be different. If desired, you can work with the CMR object directly rather than using one of the wrapper functions provided. You can then directly call the following methods of the CMR class:
- `run_trial()`: Simulates a standard free recall trial.
- `run_norm_recog_single_sess()`: Simulates a session of normal recognition task.
- `run_conti_recog_single_sess()`: Simulates a session of continuous recognition task.
- `run_norm_cr_single_sess()`:  Simulates a session of cued recall.
- `run_success_single_sess()`: Simulates a session of successive tests.
- `present_item()`: Presents a single item (or distractor) to the model.
- `simulate_recall()`: Simulates a freerecall period.
- `simulate_recog()`: Simulates a recognition judgment given a test probe.
- `simulate_cr()`: Simulates a cued recall given a cue.

## FITTING CMR_IA

In order to fit CMR-IA to a set of data, you will need to use some type ofoptimization algorithm to search the paramaeter space and evaluate the goodnessof fit of different parameter sets. Although you can choose to use any algorithm you like, included is a particle swarm algorithm that has been used in previous work. It can be found in the `fitting/` subdirectory. In order to make use of these functions, you will need to customize them for your purposes. You can find additional code in `object_funcs.py` and `optimization_utils.py` to help you design your actual goodness-of-fit test, score your model's recall performance, and more. You can run `pso_cmr.py` to run the particle swarm optimization algorithm, but the recommended way to run it is in parallel, as described below.

Regardless of which algorithm you are using, you can run your optimization jobs in parallel by running the following commands. First, edit the following line in the `runpyfile.sh` and change it to your python path in your environment with CMR-IA installed:

~~~
PY_COMMAND="/your/path/to/python"
~~~

Then, modify these two lines in `pgo`, where the second line is the command to submit a job to the cluster and the first line is to print the command to the terminal. `-t` is the time limit, `-c` is the number of cores, `--mem` is the memory limit, `-w` is the node you want to run on. These should be modified to fit your cluster's requirements. You could also delete the `-w` option if you don't have a specific node you want to run on. If your cluster is not using SLURM, however, you should use another command instead of `sbatch` to submit jobs and the format will probably be different.

~~~
echo sbatch -p RAM -t 7-00:00 -c 1 --mem=5G -w node26 -J $PNAME $SH_SCRIPT
sbatch -p RAM -t 7-00:00 -c 1 --mem=5G -w node26 -J $PNAME $SH_SCRIPT
~~~

Finally, run:

~~~
./pgo FILENAME N_JOBS
~~~

Where FILENAME is the optimization algorithm's file path (e.g., `pso_cmr.py`) and N_JOBS is the number of parallel jobs you wish to run. 

In the PSO script, parallel instances will automatically track one another's progress to make sure the next iteration starts once all jobs have finished evaluating the current iteration. You therefore only need to run pgo once, rather than once per iteration.

IMPORTANT: The particle swarm leaves behind many files tracking intermediate steps of the algorithm. Once the algorithm has finished, remember to delete all tempfiles and keep only the files with the goodness of fit scores and parameter values from each iteration. My practice is to backup the files in `outfiles/` and delete all files in `noise_files/`. `outfiles/` and `noise_files/` should be kept empty before running the next optimization job.

NOTE: Particle swarms are designed to test small numbers of parameter sets for hundreds/thousands of iterations. Parameter sets within an iteration can be tested in parallel. Each new iteration cannot start until all parameter sets from the current iteration have finished.