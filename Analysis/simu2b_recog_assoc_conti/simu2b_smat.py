"""
Generate semantic similarity matrix for simu2b items w2v and save the item-to-number mapping.
"""

import json
import numpy as np
import pandas as pd
import gensim.downloader as api

# Load raw data to get all unique words
df = pd.read_csv("data/exp1.csv")
items = np.unique(np.concatenate([df.word1.to_numpy(), df.word2.to_numpy()]))
item2no = {item.lower().strip(): i + 1 for i, item in enumerate(items)}
wordlist = list(item2no.keys())

# Load fasttext model and compute similarity matrix
wv = api.load("fasttext-wiki-news-subwords-300")
print("wv loaded")
vecs = np.array([wv[w] for w in wordlist])
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_normed = vecs / norms
s_mat = vecs_normed @ vecs_normed.T

# Save
np.save("data/simu2b_smat.npy", s_mat)
with open("data/item2no.json", "w") as f:
    json.dump(item2no, f)
print(f"Saved: simu2b_smat.npy  shape={s_mat.shape}")
