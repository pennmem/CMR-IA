"""
Self-generated simplified design from Hockley. Use PEERS semantic matrix.
Here each subject 1 list, origin each subject 2 lists * 5 sessions. both aggregate stats as a whole (so acceptable).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
simu_sess_num = 300
n = 160
wordpool = np.arange(1, 1639)
pos_lags = np.array([2, 4, 6, 8, 16])
parts = []
for sess in range(simu_sess_num):
    
    # Choose present words and order
    pres_words = rng.choice(wordpool, 2 * n, replace=False).reshape(n, 2)

    # Choose test type for each study position
    pres_type = rng.permutation(["single_new", "single_old", "pair_new", "pair_old"] * 40)
    while pres_type[0] == "pair_new":
        pres_type = rng.permutation(pres_type)

    # Choose test words corresponding to pres_words and pres_type
    new_words = rng.permutation(wordpool[~np.isin(wordpool, pres_words)])
    newidx = 0
    test_words = []
    for i in range(n):
        t = pres_type[i]
        tmp = pres_words[i]
        if t == "single_old":
            pick = rng.choice([0, 1])
            test_words.append([tmp[pick], -1])
        elif t == "single_new":
            test_words.append([new_words[newidx], -1])
            newidx += 1
        elif t == "pair_old":
            test_words.append(tmp.tolist())
        elif t == "pair_new":
            tmp_pre = pres_words[i - 1].tolist()
            order = rng.permutation([tmp, tmp_pre])
            test_words.append([order[0][0], order[1][1]])
    test_words = np.array(test_words)

    # Algorithm fitting lags to study positions
    presidx = np.arange(n, dtype=int)
    testidx = np.zeros(n, dtype=int)
    tested = np.zeros(n, dtype=int)
    test_lag = np.zeros(n, dtype=int)
    test_type = np.array(["no_fit"] * n, dtype="<U32")
    lags = pos_lags.copy()
    while True:
        if lags.size == 0:
            break
        lag = rng.choice(lags)
        for i in range(1, n):
            if tested[i] == 0 and i + lag <= n - 1 and test_type[i + lag] == "no_fit":
                testidx[i + lag] = presidx[i]
                test_type[i + lag] = pres_type[i]
                test_lag[i + lag] = lag
                tested[i] = 1
                break
        else:
            lags = np.delete(lags, np.argwhere(lags == lag))

    # Create test sequence
    test_seq = []
    for i in range(n):
        if test_type[i] == "no_fit":
            testidx[i] = -1
            test_seq.append([-1, -1])
        else:
            test_seq.append(test_words[testidx[i]])
    test_seq = np.array(test_seq)

    parts.append(pd.DataFrame({
        "position": presidx,
        "session": sess,
        "testidx": testidx,
        "lag": test_lag,
        "type": test_type,
        "study_itemno1": pres_words[:, 0],
        "study_itemno2": pres_words[:, 1],
        "test_itemno1": test_seq[:, 0],
        "test_itemno2": test_seq[:, 1],
    }))

df = pd.concat(parts).reset_index(drop=True)

# Save
df.to_parquet("data/simu3_test.parquet", index=False)
print(f"Saved: simu3_test.parquet  shape={df.shape}")
