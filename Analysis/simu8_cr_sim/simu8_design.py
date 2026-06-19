"""
Self-generated simplified design from Pantelis. Use semantic matrix from distance.
Only one list here.

Group 1: study 8 random pairs, cued recall.
Group 2: study all 16 pairs, associative recognition (test 1) + cued recall (test 2).
"""

import numpy as np
import pandas as pd

face_list = ["Face" + str(x) for x in range(1, 17)]
name_list = ["Jim", "John", "Rob", "Bill", "Dave", "Rich", "Charles", "Joe",
             "Tom", "Chris", "Dan", "Paul", "Mark", "Mike", "George", "Ken"]
item_list = np.array(face_list + name_list)

simu_sess_num = 10000
list_num = 1
n = 8
facepool = np.arange(1, 17)
namepool = np.arange(17, 33)
serial_pos = np.arange(8)


# --- Group 1: Cued Recall (8 random pairs) ---

rng = np.random.default_rng(seed=42)
g1_study_parts, g1_test_parts = [], []
for sess in range(simu_sess_num):

    sess_faces = rng.choice(facepool, n, replace=False)
    sess_names = rng.choice(namepool, n, replace=False)
    sess_pairs = np.vstack([sess_faces, sess_names]).T
    for lst in range(list_num):

        # Study
        pres_words = rng.permutation(sess_pairs)
        g1_study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": item_list[pres_words[:, 0] - 1],
            "study_item2": item_list[pres_words[:, 1] - 1],
            "serial_pos": serial_pos,
            "list": lst,
            "session": sess,
        }))

        # Test (cued recall)
        study_pos_sel = rng.permutation(serial_pos)
        test_probe = [pres_words[pos][0] for pos in study_pos_sel]
        correct_ans = [pres_words[pos][1] for pos in study_pos_sel]
        g1_test_parts.append(pd.DataFrame({
            "test_itemno1": test_probe,
            "test_itemno2": -1,
            "test_item1": item_list[np.array(test_probe) - 1],
            "test_item2": None,
            "correct_ans": correct_ans,
            "study_pos": study_pos_sel,
            "test": 1,
            "list": lst,
            "session": sess,
        }))

df_g1_study = pd.concat(g1_study_parts, ignore_index=True)
df_g1_test = pd.concat(g1_test_parts, ignore_index=True)
df_g1_study["group"] = 1
df_g1_test["group"] = 1


# --- Group 2: Associative Recognition + Cued Recall (all 16 pairs) ---

n2 = 16
serial_pos2 = np.arange(n2)

rng = np.random.default_rng(seed=42)
g2_study_parts, g2_test_parts = [], []
for sess in range(simu_sess_num):

    sess_faces = rng.permutation(facepool)
    sess_names = rng.permutation(namepool)
    sess_pairs = np.vstack([sess_faces, sess_names]).T

    # Study
    pres_words = rng.permutation(sess_pairs)
    g2_study_parts.append(pd.DataFrame({
        "study_itemno1": pres_words[:, 0],
        "study_itemno2": pres_words[:, 1],
        "study_item1": item_list[pres_words[:, 0] - 1],
        "study_item2": item_list[pres_words[:, 1] - 1],
        "serial_pos": serial_pos2,
        "list": 0,
        "session": sess,
    }))

    # Test 1: associative recognition
    half = n2 // 2
    intact_mask = np.zeros(n2, dtype=bool)
    intact_mask[rng.choice(n2, half, replace=False)] = True

    intact_pairs = pres_words[intact_mask].copy()
    rearrange_pairs = pres_words[~intact_mask].copy()
    rearrange_pairs[:, 1] = rng.permutation(rearrange_pairs[:, 1])

    test1_pairs = np.concatenate([intact_pairs, rearrange_pairs], axis=0)
    test1_correct = np.array([1] * half + [0] * half)
    shuffle_idx = rng.permutation(n2)
    test1_pairs = test1_pairs[shuffle_idx]
    test1_correct = test1_correct[shuffle_idx]

    g2_test_parts.append(pd.DataFrame({
        "test_itemno1": test1_pairs[:, 0],
        "test_itemno2": test1_pairs[:, 1],
        "test_item1": item_list[test1_pairs[:, 0] - 1],
        "test_item2": item_list[test1_pairs[:, 1] - 1],
        "correct_ans": test1_correct,
        "study_pos": -1,
        "test": 1,
        "list": 0,
        "session": sess,
    }))

    # Test 2: cued recall
    study_pos_sel = rng.permutation(serial_pos2)
    test2_probe = [pres_words[pos][0] for pos in study_pos_sel]
    test2_ans = [pres_words[pos][1] for pos in study_pos_sel]
    g2_test_parts.append(pd.DataFrame({
        "test_itemno1": test2_probe,
        "test_itemno2": -1,
        "test_item1": item_list[np.array(test2_probe) - 1],
        "test_item2": None,
        "correct_ans": test2_ans,
        "study_pos": study_pos_sel,
        "test": 2,
        "list": 0,
        "session": sess,
    }))

df_g2_study = pd.concat(g2_study_parts, ignore_index=True)
df_g2_test = pd.concat(g2_test_parts, ignore_index=True)
df_g2_study["group"] = 2
df_g2_test["group"] = 2


# Merge groups
df_study = pd.concat([df_g1_study, df_g2_study], ignore_index=True)
df_test = pd.concat([df_g1_test, df_g2_test], ignore_index=True)

# Save
df_study.to_parquet("data/simu8_study.parquet", index=False)
df_test.to_parquet("data/simu8_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
