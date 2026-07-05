"""
Self-generated simplified design from Hockley. Use PEERS semantic matrix.
Here each subject 1 list, origin each subject 2 lists * 5 sessions. both aggregate stats as a whole (so acceptable).
"""

import numpy as np
import pandas as pd

# Settings
rng = np.random.default_rng(seed=42)
simu_sess_num = 300
stream_len = 320  # ~160 study + ~160 test presentations
pos_lags = np.array([2, 4, 6, 8, 16])
n_rep = 8  # replications of each test type at each lag
test_types = ["single_old", "single_new", "pair_old", "pair_new"]
wordpool = np.arange(1, 1639)
min_new_pos = pos_lags.min() + 1  # keep new-item tests after the first few studies

# Draw one session's words without replacement, one word at a time
def take(k):
    return np.array([next(word_iter) for _ in range(k)], dtype=int)

parts = []
for sess in range(simu_sess_num):

    # Per-session stream state
    word_iter = iter(rng.permutation(wordpool))
    occupied = np.zeros(stream_len, dtype=bool)
    ev_type = np.array(["study"] * stream_len, dtype="<U16")
    ev_lag = np.zeros(stream_len, dtype=int)
    study_w = np.full((stream_len, 2), -1, dtype=int)
    test_w = np.full((stream_len, 2), -1, dtype=int)

    # Build and shuffle the test requests
    requests = [(t, lag) for t in test_types for lag in pos_lags for _ in range(n_rep)]
    rng.shuffle(requests)

    # Greedily fit each test (and its study presentation/s) into the earliest available positions
    for typ, lag in requests:
        if typ == "single_new":
            for t in range(min_new_pos, stream_len):
                if not occupied[t]:
                    occupied[t] = True
                    ev_type[t], ev_lag[t] = typ, lag
                    test_w[t] = [take(1)[0], -1]
                    break
        elif typ in ("single_old", "pair_old"):
            for s in range(stream_len):
                t = s + lag + 1  # intervening presentations between study s and test t equal lag
                if t >= stream_len:
                    break
                if not occupied[s] and not occupied[t]:
                    w = take(2)
                    occupied[s] = occupied[t] = True
                    study_w[s] = w
                    ev_type[t], ev_lag[t] = typ, lag
                    test_w[t] = [w[rng.integers(2)], -1] if typ == "single_old" else [w[0], w[1]]
                    break
        elif typ == "pair_new":
            for s in range(stream_len - 1):
                t = (s + 1) + lag + 1  # lag measured from the more recent study pair at s+1
                if t >= stream_len:
                    break
                if not occupied[s] and not occupied[s + 1] and not occupied[t]:
                    w_prev, w_recent = take(2), take(2)  # study pairs at s and s+1
                    occupied[s] = occupied[s + 1] = occupied[t] = True
                    study_w[s], study_w[s + 1] = w_prev, w_recent
                    ev_type[t], ev_lag[t] = typ, lag
                    test_w[t] = [w_recent[0], w_prev[1]] if rng.integers(2) == 0 else [w_prev[0], w_recent[1]]
                    break

    # Fill remaining free positions with filler study pairs
    for p in range(stream_len):
        if not occupied[p]:
            study_w[p] = take(2)

    parts.append(pd.DataFrame({
        "session": sess,
        "position": np.arange(stream_len),
        "lag": ev_lag,
        "type": ev_type,
        "study_itemno1": study_w[:, 0],
        "study_itemno2": study_w[:, 1],
        "test_itemno1": test_w[:, 0],
        "test_itemno2": test_w[:, 1],
    }))

df = pd.concat(parts).reset_index(drop=True)

# Save
df.to_parquet("data/simu3_test.parquet", index=False)
print(f"Saved: simu3_test.parquet  shape={df.shape}")
