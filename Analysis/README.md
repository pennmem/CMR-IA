## ANALYSIS FILE SUMMARY

Each folder starting with simu corresponds to a subsection of simulation in the paper:
- `simu1_recog_recsim/`: Simulation 1, also contains data and analysis of Experiment 1
- `simu2_recog_conti/`: Simulation 2 Successive-probe Contiguity Effects
- `simu3_recog_forget/`: Simulation 3 Differential Forgetting of Items and Associations
- `simu4_recog_wfe/`: Simulation 4 Word Frequency Effects in Recognition Memory
- `simu5_cr_rec/`: Simulation 5 Serial Position Effects in Cued Recall
- `simu6a_cr_recsym/`: Simulation 6 Associative Symmetry Figure 7
- `simu6b_cr_sym/`: Simulation 6 Associative Symmetry Table 2
- `simu7_cr_pliili/`: Simulation 7 Prior-list and Intra-list Intrusions in Cued Recall
- `simu8_cr_sim/`: Simulation 8 Similarity Effects
- `simuS1_recog_cr/`: Simulating Experiment 2
- `simuS2_recog_recog/`: Simulating Experiment 3

In each folder:
- `simuX_main.ipynb`: the notebook for the simulation, contains the code for running the simulation in CMR-IA and analysis of the simulation results. Intersted readers can run the notebook to reproduce the simulation results.
- `simuX_{name}.ipynb`: the notebook for the corresponding real experiment.
- `simuX_design.ipynb`: the notebook for generating the design of the simulation (df_study and df_test).
- `simuX_smat.ipynb`: the notebook for generating the simulation-specific semantic association matrix (if applicable).
- `simuX_data/`: contains the data files for the simulation (e.g., study df, test df, simulation results df), sometimes include the data from the corresponding real experiment, and sometimes include the simulation-specific semantic association matrix.
- `simuX_fig/`: the generated figures, as in the paper.

For example, in `simu1_recog_recsim/`:
- `simu1_main.ipynb`: the notebook for running Simulation 1.
- `simu1_design.ipynb`: the notebook for generating the design dataframe of Simulation 1.
- `simu1_smat.ipynb`: the notebook for generating the semantic association matrix of Simulation 1.
- `simu1_David.ipynb`: the notebook for analyzing Experiment 1 (after running simu1_David_preprocess.ipynb).
- `simu1_R/`: the R scripts for statistical inference in Experiment 1.
- `simu1_data/`: contains the data files for and from the simulation, and the data from Experiment 1 (cr_preproc_data_mturk.csv).
- `simu1_fig/`: the generated figures.