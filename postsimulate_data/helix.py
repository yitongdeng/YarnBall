import numpy as np
import os
import copy
from scipy import linalg

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok = True)
out_dir = os.path.join(logs_dir, "helix")
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

def r_func(z):
    return 0.03 * np.ones_like(z)

def lambda_func(z):
    return 0.1 * np.ones_like(z)

# rotation matrix about u for angle theta
def rot(u, theta):
    u_hat = np.array([[0, u[2], -u[1]],[-u[2], 0, u[0]],[u[1], -u[0], 0]])
    R = linalg.expm(theta * u_hat)
    return R

def twist_helix(_helix, k, theta):
    helix = copy.deepcopy(_helix)
    pos_k = helix[k]
    u = helix[k+1] - pos_k
    u /= np.linalg.norm(u)
    R = rot(u, theta)
    helix[k:] = np.einsum('ij,kj->ki', R, helix[k:]-pos_k) + pos_k
    return helix


n_points = 100
z = np.linspace(0, 1, n_points)
x = r_func(z) * np.cos(2 * np.pi / lambda_func(z) * z)
y = r_func(z) * np.sin(2 * np.pi / lambda_func(z) * z)

helix = np.stack([x, y, z]).T

for i in range(1):
    twisted = twist_helix(helix, 50, i/100 * np.pi)
    write_obj_file_list([twisted], os.path.join(out_dir, f"{i+1}.obj"))
    