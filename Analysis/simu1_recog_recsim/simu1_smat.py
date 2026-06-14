"""
Generate semantic similarity matrix for simu1 items using w2v.
"""

import numpy as np
import pandas as pd
import gensim.downloader as api

# Load test df to get item list
df = pd.read_parquet("data/simu1_test.parquet")
items = np.unique(df.item.to_numpy())
item2no = {item: i + 1 for i, item in enumerate(items)}
wordlist = [w.lower() for w in item2no.keys()]

# Load fasttext model and compute similarity matrix
wv = api.load("fasttext-wiki-news-subwords-300")
print("wv loaded")
vecs = np.array([wv[w] for w in wordlist])
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_normed = vecs / norms
s_mat = vecs_normed @ vecs_normed.T

# Save
np.save("data/simu1_smat.npy", s_mat)
print(f"Saved: simu1_smat.npy  shape={s_mat.shape}")
