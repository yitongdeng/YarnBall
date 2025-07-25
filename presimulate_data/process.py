import os
import numpy as np
import json5
import json
from scipy.spatial.transform import Rotation as R

from utils.arguments import parseArgs
from utils.geometry import load_strands
from utils.strands_to_frame import strands_to_frame
from utils.cubic_viz import interactive_cubic
from utils.spline_utils import evaluate_spline_batch

def write_obj_file_list(list_of_vertices, filename="output.obj"):
    with open(filename, 'w') as f:
        vertices_count = 0
        for vertices in list_of_vertices:
            # Write vertices to the .obj file
            for v in vertices:  # Transpose to iterate over columns
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # Write lines to the .obj file connecting consecutive vertices
            f.write("l")
            for i in range(1, vertices.T.shape[1] + 1):
                f.write(f" {i+vertices_count}")
            f.write("\n")
            vertices_count += vertices.shape[0]

# Process strands
n = 1
num_selected = n**2

poss = np.load("poss.npy")[:num_selected]
x_arr, y_arr, z_arr = poss[..., 0], poss[..., 1], poss[..., 2]
# Sarah's code
args = parseArgs("Helix Generation")
args.n_fine_sampling = poss.shape[1]
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
x_fine_arr, y_fine_arr, z_fine_arr = evaluate_spline_batch(x_arr, y_arr, z_arr, args.n_fine_sampling)
e_1_arr, e_2_arr, e_3_arr, curvature_arr, torsion_arr = strands_to_frame(x_arr, y_arr, z_arr, args)

q_arr = []
for i, (e_1s, e_2s, e_3s) in enumerate(zip(e_1_arr, e_2_arr, e_3_arr)):
    qs = []
    for j, (e_1, e_2, e_3) in enumerate(zip(e_1s, e_2s, e_3s)):
        rot_mat = np.hstack((e_1.reshape(-1, 1), e_2.reshape(-1, 1), e_3.reshape(-1, 1)))
        rotation = R.from_matrix(rot_mat)
        q = rotation.as_quat()
        qs.append(q)
    q_arr.append(qs)

q_arr = np.array(q_arr)
q_arr_flat = q_arr.reshape((-1, 4))
# Frenet quaternion obtained

n = 1
num_selected = n**2
global_scale = 0.1

# 10 × 10 grid in the x‑ and y‑directions
x, y = np.meshgrid(np.arange(n),  # 0 … 9 (columns)
                   np.arange(n),  # 0 … 9 (rows)
                   indexing='xy')   # x varies fastest, y slowest

coords = global_scale * 100 * np.stack([x.ravel(), y.ravel(), 0*x.ravel()], axis=1)  # (100, 2) → [[0,0], … [9,9]]

poss = global_scale * poss

poss = poss[:num_selected] + coords[:, np.newaxis, :]

write_obj_file_list(poss, filename = "combined.obj")

# process the json
# # 1 . your (N, 3) NumPy array

# 2 . read the existing file (it contains // comments, so use json5/commentjson)
with open('template.json', 'r') as f:
    cfg = json5.load(f)

# 3 . replace the field
cfg["fixVertex"] = coords.tolist()      # shape (N, 3) → list‑of‑lists

cfg["frenetQ"] = q_arr_flat.tolist()      # shape (N, 3) → list‑of‑lists

# 4 . write it back (standard json is fine for output if you don’t need comments)
with open('my_hair.json', 'w') as f:
    json.dump(cfg, f, indent=4)