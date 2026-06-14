"""
Self-generated simplified design from Pantelis. Use semantic matrix from distance.
Only one list here.
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

rng = np.random.default_rng(seed=42)
study_parts = []
test_parts = []
for sess in range(simu_sess_num):
    
    sess_faces = rng.choice(facepool, n, replace=False)
    sess_names = rng.choice(namepool, n, replace=False)
    sess_pairs = np.vstack([sess_faces, sess_names]).T
    for lst in range(list_num):
        
        # Study
        pres_words = rng.permutation(sess_pairs)
        study_parts.append(pd.DataFrame({
            "study_itemno1": pres_words[:, 0],
            "study_itemno2": pres_words[:, 1],
            "study_item1": item_list[pres_words[:, 0] - 1],
            "study_item2": item_list[pres_words[:, 1] - 1],
            "serial_pos": serial_pos,
            "list": lst,
            "session": sess,
        }))

        # Test
        study_pos_sel = rng.permutation(serial_pos)
        test_probe = [pres_words[pos][0] for pos in study_pos_sel]
        correct_ans = [pres_words[pos][1] for pos in study_pos_sel]
        test_parts.append(pd.DataFrame({
            "test_itemno": test_probe,
            "test_item": item_list[np.array(test_probe) - 1],
            "correct_ans": correct_ans,
            "study_pos": study_pos_sel,
            "list": lst,
            "session": sess,
        }))

df_study = pd.concat(study_parts, ignore_index=True)
df_test = pd.concat(test_parts, ignore_index=True)

# Save
df_study.to_parquet("data/simu8_study.parquet", index=False)
df_test.to_parquet("data/simu8_test.parquet", index=False)
print(f"Saved: {len(df_study)} study rows, {len(df_test)} test rows")
