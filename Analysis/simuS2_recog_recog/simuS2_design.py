"""
Self-generated design from exp3. Use PEERS semantic matrix.
"""

import numpy as np
import pandas as pd

# Load 1638-word pool
with open("../wordpools/wasnorm_wordpool.txt") as f:
    items = np.array([line.rstrip("\n") for line in f])

nsubj = 300
nlist = 3
n = 25
v = 5
n_exitem = 15
n_expair = 15
extra = 2 * (n_exitem + n_expair * 2)
wordpool = np.arange(1, 1639)


def repeat_6_no_conti(n, rng):
    """Repeat each of n items 6 times in random order without consecutive repeats."""
    idx_order = []
    repeat_cnt = np.zeros(n)
    number_left = np.arange(n)
    last_pick = -1
    for _ in range(n * 6):
        pick = rng.choice(number_left)
        loop_cnt = 0
        while pick == last_pick:
            pick = rng.choice(number_left)
            loop_cnt += 1
            if loop_cnt > 100:
                return idx_order, False
        idx_order.append(pick)
        repeat_cnt[pick] += 1
        if repeat_cnt[pick] == 6:
            number_left = np.delete(number_left, np.where(number_left == pick))
        last_pick = pick
    return idx_order, True


rng = np.random.default_rng(seed=42)
study_parts = []
test_parts = []
for subj in range(nsubj):

    all_words = rng.choice(wordpool, nlist * (2 * (n + v) + extra), replace=False)
    for lst in range(nlist):

        # Study
        base = lst * (2 * (n + v) + extra)
        pres_words = all_words[base : base + 2 * n].reshape(n, 2)
        tmp_study_base = pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": items[pres_words[:, 0] - 1],
            "study_item2": items[pres_words[:, 1] - 1],
            "pair_idx": np.arange(n) + lst * (n + v),
            "list": lst, "session": subj, "subject": subj,
        })

        # Repeat 6 times without consecutive repetitions
        idx_order, flag = repeat_6_no_conti(n, rng)
        while not flag:
            idx_order, flag = repeat_6_no_conti(n, rng)
        study_parts.append(tmp_study_base.iloc[idx_order])

        # Test conditions for n + v pairs
        valid_words = all_words[base : base + 2 * (n + v)].reshape(n + v, 2)
        conditions = rng.permutation(["Same_Item"] * 6 + ["Different_Item"] * 6 + ["Intact_Pair"] * 3 + ["Item_Pair"] * 9 + ["Pair_Item"] * 6)
        test1_probe, test2_probe = [], []
        for i, c in enumerate(conditions):
            order = rng.choice([0, 1])
            this_pair = valid_words[i].tolist()
            if c == "Same_Item":
                test1_probe.append([this_pair[order], -1])
                test2_probe.append([this_pair[order], -1])
            elif c == "Different_Item":
                test1_probe.append([this_pair[order], -1])
                test2_probe.append([this_pair[1 - order], -1])
            elif c == "Intact_Pair":
                test1_probe.append(this_pair)
                test2_probe.append(this_pair)
            elif c == "Item_Pair":
                test1_probe.append([this_pair[order], -1])
                test2_probe.append(this_pair)
            elif c == "Pair_Item":
                test1_probe.append(this_pair)
                test2_probe.append([this_pair[order], -1])
        test1_probe = np.array(test1_probe)
        test2_probe = np.array(test2_probe)

        # Test 1
        t1_extra = all_words[base + 2 * (n + v) : base + 2 * (n + v) + extra // 2]
        t1_ei = t1_extra[:n_exitem]
        t1_ep = t1_extra[n_exitem:].reshape(n_expair, 2)
        t1_itemno1 = np.concatenate((test1_probe[:, 0], t1_ei, t1_ep[:, 0]))
        t1_itemno2 = np.concatenate((test1_probe[:, 1], [-1] * n_exitem, t1_ep[:, 1]))
        tmp_test1 = pd.DataFrame({
            "test_itemno1": t1_itemno1, "test_itemno2": t1_itemno2,
            "test_item1": [items[i - 1] if i != -1 else "None" for i in t1_itemno1],
            "test_item2": [items[i - 1] if i != -1 else "None" for i in t1_itemno2],
            "correct_ans": [1] * n + [0] * (v + n_exitem + n_expair),
            "pair_idx": np.concatenate((np.arange(n + v) + lst * (n + v), [-1] * (n_exitem + n_expair))),
            "type": np.concatenate((conditions, ["extra"] * (n_exitem + n_expair))),
            "test": 1, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test1 = tmp_test1.iloc[rng.permutation(tmp_test1.index)].reset_index(drop=True)

        # Test 2
        t2_extra = all_words[(lst + 1) * (2 * (n + v) + extra) - extra // 2 : (lst + 1) * (2 * (n + v) + extra)]
        t2_ei = t2_extra[:n_exitem]
        t2_ep = t2_extra[n_exitem:].reshape(n_expair, 2)
        t2_itemno1 = np.concatenate((test2_probe[:, 0], t2_ei, t2_ep[:, 0]))
        t2_itemno2 = np.concatenate((test2_probe[:, 1], [-1] * n_exitem, t2_ep[:, 1]))
        tmp_test2 = pd.DataFrame({
            "test_itemno1": t2_itemno1, "test_itemno2": t2_itemno2,
            "test_item1": [items[i - 1] if i != -1 else "None" for i in t2_itemno1],
            "test_item2": [items[i - 1] if i != -1 else "None" for i in t2_itemno2],
            "correct_ans": [1] * n + [0] * (v + n_exitem + n_expair),
            "pair_idx": np.concatenate((np.arange(n + v) + lst * (n + v), [-1] * (n_exitem + n_expair))),
            "type": np.concatenate((conditions, ["extra"] * (n_exitem + n_expair))),
            "test": 2, "list": lst, "session": subj, "subject": subj,
        })
        tmp_test2 = tmp_test2.iloc[rng.permutation(tmp_test2.index)].reset_index(drop=True)
        test_parts.extend([tmp_test1, tmp_test2])

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simuS2_study.parquet", index=False)
df_test.to_parquet("data/simuS2_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
