import numpy as np
import pandas as pd
from CMR_IA.pso import run_pso


if __name__ == "__main__":

    SIMU_NAME = "8"
    ANAL_DIR = "../../Analysis"
    SEED_FILE = None

    if SIMU_NAME == "1":

        df_study = None
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu1_recog_recsim/data/simu1_test.parquet")
        sem_file = f"{ANAL_DIR}/simu1_recog_recsim/data/simu1_smat.npy"

    elif SIMU_NAME == "2":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu2_recog_conti/data/simu2_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu2_recog_conti/data/simu2_test.parquet")
        sem_file = f"{ANAL_DIR}/simu2_recog_conti/data/simu2_smat.npy"

    elif SIMU_NAME == "2b":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu2b_recog_assoc_conti/data/simu2b_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu2b_recog_assoc_conti/data/simu2b_test.parquet")
        sem_file = f"{ANAL_DIR}/simu2b_recog_assoc_conti/data/simu2b_smat.npy"

    elif SIMU_NAME == "3":

        df_study = None
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu3_recog_forget/data/simu3_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME in ["4", "4base", "4shift", "4attn"]:

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu4_recog_wfe/data/simu4_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu4_recog_wfe/data/simu4_test.parquet")
        sem_file = f"{ANAL_DIR}/simu4_recog_wfe/data/simu4_smat.npy"

    elif SIMU_NAME == "5":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu5_cr_rec/data/simu5_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu5_cr_rec/data/simu5_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME == "6a":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu6a_cr_recsym/data/simu6a_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu6a_cr_recsym/data/simu6a_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME == "6b":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu6b_cr_sym/data/simu6b_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu6b_cr_sym/data/simu6b_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME == "7":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu7_cr_pliili/data/simu7_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu7_cr_pliili/data/simu7_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME == "8":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simu8_cr_sim/data/simu8_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simu8_cr_sim/data/simu8_test.parquet")
        sem_file = f"{ANAL_DIR}/simu8_cr_sim/data/simu8_smat.npy"

    elif SIMU_NAME == "S1":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simuS1_recog_cr/data/simuS1_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simuS1_recog_cr/data/simuS1_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    elif SIMU_NAME == "S2":

        df_study = pd.read_parquet(f"{ANAL_DIR}/simuS2_recog_recog/data/simuS2_study.parquet")
        df_test = pd.read_parquet(f"{ANAL_DIR}/simuS2_recog_recog/data/simuS2_test.parquet")
        sem_file = f"{ANAL_DIR}/wordpools/ltp_FR_similarity_matrix.npy"

    else:

        raise ValueError("Simulation name not recognized!")

    # Load semantic similarity matrix
    sem_mat = np.load(sem_file)

    # Run PSO
    run_pso(simu_name=SIMU_NAME, df_study=df_study, df_test=df_test, sem_mat=sem_mat, swarm_size=200, n_iter=200, seed_file=SEED_FILE)
