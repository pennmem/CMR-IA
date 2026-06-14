"""
Extract semantic similarity matrix for simu4 words from the PEERS/LTP word vector matrix.
"""

import numpy as np
import pandas as pd
import ast
from striprtf.striprtf import rtf_to_text

# Load LTP word list
with open("../wordpools/ltpFR_words.rtf", "r") as f:
    words = ast.literal_eval(rtf_to_text(f.read()).split("\n")[2])

# Load LTP similarity matrix
w2v = np.load("../wordpools/ltp_FR_similarity_matrix.npy")

# Load simu4 word frequency data (contains itemno_old mapping into LTP)
df = pd.read_parquet("data/simu4_word_freq.parquet")

# Extract submatrix for the 984 simu4 words
select_idx = df.itemno_old.to_numpy() - 1
w2v_new = w2v[np.ix_(select_idx, select_idx)]

# Save
np.save("data/simu4_smat.npy", w2v_new)
print(f"Saved: simu4_smat.npy  shape={w2v_new.shape}")
