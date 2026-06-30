"""
Generate semantic similarity matrix for simu8 faces from MDS-coordinate Euclidean distance.
Faces (items 1-16) use a logistic (sigmoid) generalization centered at the neighbour radius on
the 4-D MDS coordinates (Pantelis et al., 2008); names (items 17-32) use identity only.
A logistic kernel saturates within the confusion radius and falls off beyond it, so the summed
semantic norm tracks the *number* of neighbours, unlike Shepard's exponential (in
simu8_smat_exponential.py), whose squared norm is dominated by the single nearest face and
therefore anti-correlates with neighbour count.
"""

import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm

# Logistic kernel: D0 = neighbour/confusion radius (matches the neighbour-count threshold), K = falloff sharpness
D0 = 2.0  # 3.0
K = 1.0  # 1.5

# Load face coordinates from experiment data, keyed by face id (0-15) to preserve ordering
events = loadmat("data/original_experiments/Experiment1/Experiment1.mat")["events"][:, 0]
coords = {}
for r in events:
    f = np.array(r["face"]).squeeze()
    c = np.array(r["facecoordinates"]).squeeze()
    if f.size == 1 and c.size == 4:
        coords[int(f)] = c
face_corrd_16 = [coords[i] for i in range(16)]

# Compute Euclidean distance between all pairs of faces in MDS space
face_distance = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        face_distance[i, j] = norm(face_corrd_16[i] - face_corrd_16[j])

# Convert distance to similarity via logistic law (saturating, monotonic decreasing in distance), self-sim = 1
face_sim = 1.0 / (1.0 + np.exp(K * (face_distance - D0)))
np.fill_diagonal(face_sim, 1.0)

# Build 32x32 matrix: faces (1-16) have distance-based sim; names (17-32) identity only
s_mat = np.zeros((32, 32))
np.fill_diagonal(s_mat, 1)
s_mat[:16, :16] = face_sim

# Save smat
np.save("data/simu8_smat.npy", s_mat)
print(f"Saved: simu8_smat.npy  shape={s_mat.shape}")

# Also save face distance matrix
np.save("data/simu8_distance.npy", np.round(face_distance, 4))
print(f"Saved: simu8_distance.npy  shape={face_distance.shape}")
