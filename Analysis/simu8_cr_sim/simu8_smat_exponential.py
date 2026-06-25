"""
Generate semantic similarity matrix for simu8 faces from MDS-coordinate Euclidean distance.
Faces (items 1-16) use Shepard's exponential generalization s=exp(-c*d) on the 4-D MDS
coordinates (Pantelis et al., 2008); names (items 17-32) use identity only.
"""

import numpy as np
from scipy.io import loadmat
from numpy.linalg import norm

# Decay parameter for Shepard's exponential similarity (larger c -> faster falloff)
C = 1.0

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

# Convert distance to similarity via Shepard's exponential law (monotonic, diagonal=1)
face_sim = np.exp(-C * face_distance)

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
