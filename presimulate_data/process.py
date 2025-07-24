import numpy as np
import json5
import json

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

n = 1
num_selected = n**2
global_scale = 0.1

# 10 × 10 grid in the x‑ and y‑directions
x, y = np.meshgrid(np.arange(n),  # 0 … 9 (columns)
                   np.arange(n),  # 0 … 9 (rows)
                   indexing='xy')   # x varies fastest, y slowest

coords = global_scale * 100 * np.stack([x.ravel(), y.ravel(), 0*x.ravel()], axis=1)  # (100, 2) → [[0,0], … [9,9]]

poss = global_scale * np.load("poss.npy")

poss = poss[:num_selected] + coords[:, np.newaxis, :]

write_obj_file_list(poss, filename = "combined.obj")

# process the json
# # 1 . your (N, 3) NumPy array

# 2 . read the existing file (it contains // comments, so use json5/commentjson)
with open('template.json', 'r') as f:
    cfg = json5.load(f)

# 3 . replace the field
cfg["fixVertex"] = coords.tolist()      # shape (N, 3) → list‑of‑lists

# 4 . write it back (standard json is fine for output if you don’t need comments)
with open('my_hair.json', 'w') as f:
    json.dump(cfg, f, indent=4)