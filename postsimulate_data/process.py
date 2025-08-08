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


def load_obj_lines(filename: str) -> np.ndarray:
    vertices = []          # will become (Nv, 3)
    line_idx = []          # list of lists – one per `l` record
    with open(filename, "r", encoding="utf8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue                         # skip comments / blanks
            head, *rest = raw.split()
            # ↪ v 1.23 4.56 7.89
            if head == "v":
                vertices.append([float(c) for c in rest])
            # ↪ l 10 20 21 22
            elif head == "l":
                # keep only the vertex index (OBJ can write "v/vt/vn")
                idx = [int(part.split("/")[0]) - 1  # OBJ is 1-based
                       for part in rest]
                line_idx.append(idx)

    # 1. vertices as (Nv, 3)
    V = np.asarray(vertices, dtype=np.float32)
    # 2. connectivity as (N, k)
    L = np.asarray(line_idx, dtype=np.int64)
    # 3. gather vertex positions for every line → (N, k, 3)
    line_positions = V[L]
    return line_positions

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

# poss = load_obj_lines("logs/helix/1.obj")
poss = load_obj_lines("frame_200.obj")

#
x_arr, y_arr, z_arr = poss[..., 0], poss[..., 1], poss[..., 2]
# Sarah's code
args = parseArgs("Helix Generation")
args.n_fine_sampling = 1 * poss.shape[1]

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
x_fine_arr, y_fine_arr, z_fine_arr = evaluate_spline_batch(x_arr, y_arr, z_arr, args.n_fine_sampling)
e_1_arr, e_2_arr, e_3_arr, curvature_arr, torsion_arr, v_arr, t_arr = strands_to_frame(x_arr, y_arr, z_arr, args)

poss_fine = np.array([np.stack([x_fine, y_fine, z_fine], axis = -1) for (x_fine, y_fine, z_fine) in zip(x_fine_arr, y_fine_arr, z_fine_arr)])
write_obj_file_list(poss_fine, "poss_fine.obj")
frame_scale = 0.01 
poss_flat = poss_fine.reshape(-1,3)
e1s_flat = frame_scale * np.array(e_1_arr).reshape(-1,3)
e2s_flat = frame_scale * np.array(e_2_arr).reshape(-1,3)
e3s_flat = frame_scale * np.array(e_3_arr).reshape(-1,3)
# poss_flat = poss.reshape(-1,3)
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e1s_flat)], filename = "e1.obj")
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e2s_flat)], filename = "e2.obj")
write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e3s_flat)], filename = "e3.obj")
#
# reconstruction
# during integration we assume kappa[i] is constant in the duration t[i] to t[i+1]
def integrate_tangent(e1, speed, t, x0):
    num_steps = t.shape[0]-1
    xs = [x0]
    #
    for i in range(num_steps):
        h = t[i+1] - t[i]
        speed_i = speed[i]
        e1_i = e1[i]
        # x
        x_next = xs[-1] + h * speed_i * e1_i
        xs.append(x_next)

    return np.stack(xs, axis = 0)

poss_recon = []
for i, (curvature, torsion, speed, x, y, z, e_1, e_2, e_3, t) in enumerate(zip(curvature_arr, torsion_arr, v_arr, x_fine_arr, y_fine_arr, z_fine_arr, e_1_arr, e_2_arr, e_3_arr, t_arr)):
    x = integrate_tangent(e1 = e_1, speed=speed, t=t, x0 = np.array([x[0], y[0], z[0]]))
    poss_recon.append(x)
poss_recon = np.array(poss_recon)

write_obj_file_list(poss_recon, "poss_recon.obj")


# during integration we assume kappa[i] is constant in the duration t[i] to t[i+1]
def integrate_frenet_serret(kappa, tau, speed, t, F0, x0):
    num_steps = kappa.shape[0]-1
    Fs = [F0]
    xs = [x0]
    #
    for i in range(num_steps):
        h = t[i+1] - t[i]
        kappa_i = kappa[i]
        tau_i = tau[i]
        speed_i = speed[i]
        #
        T = Fs[-1][:, 0]
        B = Fs[-1][:, 2]
        omega = speed_i * (-tau_i * T + kappa_i * B)
        displacement = h * omega
        angle = np.linalg.norm(displacement)
        if angle > 1.e-6:
            axis = displacement / angle
            skew = np.array([[0,-axis[2],axis[1]],
                        [axis[2],0,-axis[0]],
                        [-axis[1],axis[0],0]])
            R = (np.eye(3)+np.sin(angle)*skew + (1-np.cos(angle))*skew@skew)
        else:
            R = np.eye(3)
        F_next = R @ Fs[-1]
        Fs.append(F_next)
        # x
        x_next = xs[-1] + h * speed_i * T
        xs.append(x_next)

    return np.stack(Fs, axis = 0), np.stack(xs, axis = 0)

poss_recon2 = []
for i, (curvature, torsion, speed, x, y, z, e_1, e_2, e_3, t) in enumerate(zip(curvature_arr, torsion_arr, v_arr, x_fine_arr, y_fine_arr, z_fine_arr, e_1_arr, e_2_arr, e_3_arr, t_arr)):
    F, x = integrate_frenet_serret(kappa=curvature, tau=torsion, speed=speed, t=t, 
            F0 = np.hstack([e_1[0].reshape(-1, 1), e_2[0].reshape(-1, 1), e_3[0].reshape(-1, 1)]), x0 = np.array([x[0], y[0], z[0]]))
    poss_recon2.append(x)

write_obj_file_list(poss_recon2, "poss_recon2.obj")

# write_obj_file_list(poss_recon, "frame_200_2.obj")

# print(poss_recon.shape)
# for i in range(len(e_1_arr[0])):
#     print(np.linalg.norm(np.cross((poss_recon[0, 1:] - poss_recon[0, :-1])[i], e_1_arr[0][i])))
# #print(e_1_arr[0])
# exit()

# from utils.arguments import parseArgs
# from utils.geometry import load_strands
# from utils.strands_to_frame import strands_to_frame
# from utils.cubic_viz import interactive_cubic
# from utils.spline_utils import evaluate_spline_batch

# def write_obj_file_list(list_of_vertices, filename="output.obj"):
#     with open(filename, 'w') as f:
#         vertices_count = 0
#         for vertices in list_of_vertices:
#             # Write vertices to the .obj file
#             for v in vertices:  # Transpose to iterate over columns
#                 f.write(f"v {v[0]} {v[1]} {v[2]}\n")
#             # Write lines to the .obj file connecting consecutive vertices
#             f.write("l")
#             for i in range(1, vertices.T.shape[1] + 1):
#                 f.write(f" {i+vertices_count}")
#             f.write("\n")
#             vertices_count += vertices.shape[0]

# # during integration we assume kappa[i] is constant in the duration t[i] to t[i+1]
# def integrate_tangent(e1, speed, t, x0):
#     num_steps = e1.shape[0]-1
#     xs = [x0]
#     #
#     for i in range(num_steps):
#         h = t[i+1] - t[i]
#         speed_i = speed[i]
#         e1_i = e1[i]
#         # x
#         x_next = xs[-1] + h * speed_i * e1_i
#         xs.append(x_next)

#     return np.stack(xs, axis = 0)

# # during integration we assume kappa[i] is constant in the duration t[i] to t[i+1]
# def integrate_frenet_serret(kappa, tau, speed, t, F0, x0):
#     num_steps = kappa.shape[0]-1
#     print(F0)
#     Fs = [F0]
#     xs = [x0]
#     #
#     for i in range(num_steps):
#         h = t[i+1] - t[i]
#         kappa_i = kappa[i]
#         tau_i = tau[i]
#         speed_i = speed[i]
#         #
#         T = Fs[-1][:, 0]
#         B = Fs[-1][:, 2]
#         omega = speed_i * (tau_i * T + kappa_i * B)
#         displacement = h * omega
#         angle = np.linalg.norm(displacement)
#         if angle > 1.e-6:
#             axis = displacement / angle
#             skew = np.array([[0,-axis[2],axis[1]],
#                         [axis[2],0,-axis[0]],
#                         [-axis[1],axis[0],0]])
#             R = (np.eye(3)+np.sin(angle)*skew + (1-np.cos(angle))*skew@skew)
#         else:
#             R = np.eye(3)
#         F_next = R @ Fs[-1]
#         Fs.append(F_next)
#         # x
#         x_next = xs[-1] + h * speed_i * F_next[:, 0]
#         xs.append(x_next)

#     return np.stack(Fs, axis = 0), np.stack(xs, axis = 0)

# # Process strands
# n = 1
# num_selected = n**2

# poss = np.load("poss.npy")[:num_selected]
# x_arr, y_arr, z_arr = poss[..., 0], poss[..., 1], poss[..., 2]
# # Sarah's code
# args = parseArgs("Helix Generation")
# args.n_fine_sampling = poss.shape[1]
# if not os.path.exists(args.save_path):
#     os.makedirs(args.save_path)
# x_fine_arr, y_fine_arr, z_fine_arr = evaluate_spline_batch(x_arr, y_arr, z_arr, args.n_fine_sampling)
# e_1_arr, e_2_arr, e_3_arr, curvature_arr, torsion_arr, v_arr, t_arr = strands_to_frame(x_arr, y_arr, z_arr, args)

# # reconstruction
# poss_recon = []
# for i, (curvature, torsion, speed, x, y, z, e_1, e_2, e_3, t) in enumerate(zip(curvature_arr, torsion_arr, v_arr, x_fine_arr, y_fine_arr, z_fine_arr, e_1_arr, e_2_arr, e_3_arr, t_arr)):
#     t *= 5.9
#     # F, x = integrate_frenet_serret(kappa=curvature, tau=torsion, speed=speed, t=t, 
#     #         F0 = np.hstack([e_1[0].reshape(-1, 1), e_2[0].reshape(-1, 1), e_3[0].reshape(-1, 1)]), x0 = np.array([x[0], y[0], z[0]]))
#     x = integrate_tangent(e1 = e_1, speed=speed, t=t, x0 = np.array([x[0], y[0], z[0]]))
#     poss_recon.append(x)
# poss_recon = np.array(poss_recon)
# # print(poss_recon.shape)
# # for i in range(len(e_1_arr[0])):
# #     print(np.linalg.norm(np.cross((poss_recon[0, 1:] - poss_recon[0, :-1])[i], e_1_arr[0][i])))
# # #print(e_1_arr[0])
# # exit()

# poss = 10 * poss_recon #IMPORTANT
# #1poss = poss

# q_arr = []
# for i, (e_1s, e_2s, e_3s) in enumerate(zip(e_1_arr, e_2_arr, e_3_arr)):
#     qs = []
#     for j, (e_1, e_2, e_3) in enumerate(zip(e_1s, e_2s, e_3s)):
#         rot_mat = np.hstack((e_1.reshape(-1, 1), e_2.reshape(-1, 1), e_3.reshape(-1, 1)))
#         rotation = R.from_matrix(rot_mat)
#         q = rotation.as_quat()
#         qs.append(q)
#     q_arr.append(qs)

# q_arr = np.array(q_arr)
# q_arr_flat = q_arr.reshape((-1, 4))
# # Frenet quaternion obtained

# n = 1
# num_selected = n**2
# global_scale = 0.1

# # 10 × 10 grid in the x‑ and y‑directions
# x, y = np.meshgrid(np.arange(n),  # 0 … 9 (columns)
#                    np.arange(n),  # 0 … 9 (rows)
#                    indexing='xy')   # x varies fastest, y slowest

# coords = global_scale * 100 * np.stack([x.ravel(), y.ravel(), 0*x.ravel()], axis=1)  # (100, 2) → [[0,0], … [9,9]]

# poss = global_scale * poss

# poss = poss[:num_selected] + coords[:, np.newaxis, :]

# write_obj_file_list(poss, filename = "combined.obj")

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