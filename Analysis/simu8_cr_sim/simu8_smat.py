"""
Generate semantic similarity matrix for simu8 faces from coordinate-based cosine similarity.
Faces (items 1-16) use cosine similarity; names (items 17-32) use identity only.
"""

import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm

# Load face coordinates from experiment data
mat = loadmat("data/simu8_Experiment1.mat")
mdata = mat["events"]
face_corrd = mdata["facecoordinates"]
face_corrd_list = [x[0].tolist()[0] for x in face_corrd if len(x[0].tolist()) != 0]
face_corrd_16 = [np.array(x) for x in np.unique(face_corrd_list, axis=0)]

# Compute cosine similarity between all pairs of faces
face_sim = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        A, B = face_corrd_16[i], face_corrd_16[j]
        face_sim[i, j] = (np.dot(A, B) / (norm(A) * norm(B)) + 1) / 2

# Build 32x32 matrix: faces (1-16) have cosine sim; names (17-32) identity only
s_mat = np.zeros((32, 32))
np.fill_diagonal(s_mat, 1)
s_mat[:16, :16] = face_sim

# Save smat
np.save("data/simu8_smat.npy", s_mat)
print(f"Saved: simu8_smat.npy  shape={s_mat.shape}")

# Also save face distance matrix
face_distance = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        face_distance[i, j] = round(norm(face_corrd_16[i] - face_corrd_16[j]), 4)
np.save("data/simu8_distance.npy", face_distance)
print(f"Saved: simu8_distance.npy  shape={face_distance.shape}")
