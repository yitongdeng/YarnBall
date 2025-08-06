import os
import numpy as np
import json5
import json
from scipy.spatial.transform import Rotation as R

#
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok = True)
out_dir = os.path.join(logs_dir, "processed")
os.makedirs(out_dir, exist_ok = True)

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
num_selected = 3
scale = 0.01
frame_scale = 0.01
offset = np.arange(num_selected)
offset = np.stack([offset, np.zeros_like(offset), np.zeros_like(offset)], axis = -1)[:, np.newaxis, :]
poss = scale * (np.load("logs/sampled/poss.npy")[:num_selected]) + offset
frames = frame_scale * np.load("logs/sampled/frames.npy")[:num_selected]
e1s = frames[..., 0, :]
e2s = frames[..., 1, :]
e3s = frames[..., 2, :]
poss_flat = poss.reshape(-1,3)
e1s_flat = e1s.reshape(-1,3)
e2s_flat = e2s.reshape(-1,3)
e3s_flat = e3s.reshape(-1,3)

write_obj_file_list(poss, filename = os.path.join(out_dir,"poss.obj"))
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e1s_flat)], filename = os.path.join(out_dir, "e1.obj"))
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e2s_flat)], filename = os.path.join(out_dir, "e2.obj"))
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e3s_flat)], filename = os.path.join(out_dir, "e3.obj"))

# # process the json
# # # 1 . your (N, 3) NumPy array

# # 2 . read the existing file (it contains // comments, so use json5/commentjson)
# with open('template.json', 'r') as f:
#     cfg = json5.load(f)

# # 3 . replace the field
# cfg["fixVertex"] = coords.tolist()      # shape (N, 3) → list‑of‑lists

# cfg["frenetQ"] = q_arr_flat.tolist()      # shape (N, 3) → list‑of‑lists

# # 4 . write it back (standard json is fine for output if you don’t need comments)
# with open('my_hair.json', 'w') as f:
#     json.dump(cfg, f, indent=4)