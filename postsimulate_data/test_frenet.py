import os
import numpy as np
import json5
import json
from scipy.spatial.transform import Rotation as R


from utils.spline_utils import get_frenet_frame

def write_strand_frames(poss, e1s, e2s, e3s, frame_scale = 1., filename_prefix = ""):
    poss_flat = poss.reshape(-1,3)
    e1s_flat = frame_scale * e1s.reshape(-1,3)
    e2s_flat = frame_scale * e2s.reshape(-1,3)
    e3s_flat = frame_scale * e3s.reshape(-1,3)
    write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e1s_flat)], filename = filename_prefix + "e1.obj")
    write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e2s_flat)], filename = filename_prefix + "e2.obj")
    write_obj_file_list([np.stack([start, end]) for start, end in zip(poss_flat, poss_flat+e3s_flat)], filename = filename_prefix + "e3.obj")

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

def make_helix_samples(R=1.0, c=0.5, t0=0.0, t1=4*np.pi, N=100):
    """
    Helix: r(t) = (R cos t, R sin t, c t)
    r'(t) = (-R sin t,  R cos t, c)
    r''(t)= (-R cos t, -R sin t, 0)
    r'''(t)= ( R sin t, -R cos t, 0)
    """
    t_fine = np.linspace(t0, t1, N)

    # Position
    x_fine = R * np.cos(t_fine)
    y_fine = R * np.sin(t_fine)
    z_fine = c * t_fine

    # First derivative
    x_prime_fine = -R * np.sin(t_fine)
    y_prime_fine =  R * np.cos(t_fine)
    z_prime_fine =  np.full_like(t_fine, c)

    # Second derivative
    x_second_fine = -R * np.cos(t_fine)
    y_second_fine = -R * np.sin(t_fine)
    z_second_fine = np.zeros_like(t_fine)

    # Third derivative
    x_third_fine =  R * np.sin(t_fine)
    y_third_fine = -R * np.cos(t_fine)
    z_third_fine = np.zeros_like(t_fine)

    return (x_fine, y_fine, z_fine,
            x_prime_fine, y_prime_fine, z_prime_fine,
            x_second_fine, y_second_fine, z_second_fine,
            x_third_fine, y_third_fine, z_third_fine,
            t_fine)

# Example usage
(x_fine, y_fine, z_fine,
 x_prime_fine, y_prime_fine, z_prime_fine,
 x_second_fine, y_second_fine, z_second_fine,
 x_third_fine, y_third_fine, z_third_fine,
 t_fine) = make_helix_samples(R=0.1, c=0.05, t0=0.0, t1=10*np.pi, N=100)

poss_fine = np.stack([x_fine, y_fine, z_fine], axis = -1)
write_obj_file_list([poss_fine], "tmp.obj")

# Now feed these into your get_frenet_frame(...)
T, N, B, curvature, torsion, speed = get_frenet_frame(
    x_fine, y_fine, z_fine,
    x_prime_fine, y_prime_fine, z_prime_fine,
    x_second_fine, y_second_fine, z_second_fine,
    x_third_fine, y_third_fine, z_third_fine,
    t_fine, visualize_all=False)

write_strand_frames(poss_fine, T, N, B, filename_prefix = "tmp_", frame_scale = 0.01)


def integrate_tangent(e1, speed, t, x0, eps=1e-12):
    """
    Integrate r'(t) = speed(t) * e1(t) using RK4 over discrete samples.

    Parameters
    ----------
    e1 : (N,3) array
        Tangent directions at each t[i] (ideally unit length).
    speed : (N,) array
        Speed ||r'(t)|| at each t[i].
    t : (N,) array
        Monotone parameter samples.
    x0 : (3,) array
        Initial position r(t[0]).
    eps : float
        Small threshold for safe normalization.

    Returns
    -------
    X : (N,3) array
        Integrated positions.
    """
    N = t.shape[0]
    X = np.empty((N, 3), dtype=float)
    X[0] = np.asarray(x0, dtype=float)

    def unit(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        return v / n

    for i in range(N - 1):
        h = t[i+1] - t[i]

        e1_i   = e1[i]
        e1_ip1 = e1[i+1]
        s_i    = speed[i]
        s_ip1  = speed[i+1]

        # Midpoint estimates (linear in time); renormalize e1 to avoid shrinkage
        e1_mid = unit(e1_i + e1_ip1)
        s_mid  = 0.5 * (s_i + s_ip1)

        k1 = s_i   * e1_i
        k2 = s_mid * e1_mid
        k3 = s_mid * e1_mid
        k4 = s_ip1 * e1_ip1

        X[i+1] = X[i] + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return X

def integrate_frenet_serret(kappa, tau, speed, t, F0, x0, reorthonormalize=True):
    """
    Integrate Frenet–Serret frames AND curve position for a non–arc-length parameterization.

    Inputs
    ------
    kappa, tau, speed : [N] arrays          # curvature, torsion, and ||r'(t)|| at t
    t : [N] array                            # parameter samples
    F0 : [3,3] array                         # initial frame, columns = [T0, N0, B0]
    x0 : [3] array                           # initial position r(t[0])
    reorthonormalize : bool                  # project frame to SO(3) each step

    Returns
    -------
    T, N, B : [N,3] arrays
    X       : [N,3] array  (reconstructed curve)
    """
    Nsamples = len(t)
    T = np.zeros((Nsamples, 3))
    Nn = np.zeros((Nsamples, 3))
    B = np.zeros((Nsamples, 3))
    X = np.zeros((Nsamples, 3))

    F = F0.copy()
    r = x0.astype(float).copy()

    T[0], Nn[0], B[0] = F[:,0], F[:,1], F[:,2]
    X[0] = r

    def S_mat(k, w):
        # Frenet–Serret generator (in the T,N,B basis)
        return np.array([[0.0, -k,   0.0],
                         [k,    0.0, -w ],
                         [0.0,  w,   0.0]])

    def rhs_F(Fmat, k, w, spd):
        # F' = F @ (spd * S(k,w))
        return Fmat @ (spd * S_mat(k, w))

    def rhs_r(Fmat, spd):
        # r' = spd * T  (T is first column of F)
        return spd * Fmat[:, 0]

    def reortho(Fmat):
        U, _, Vt = np.linalg.svd(Fmat, full_matrices=False)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    for i in range(Nsamples - 1):
        dt = t[i+1] - t[i]

        # Endpoint + midpoint values (piecewise linear in time)
        k1, k2 = kappa[i], kappa[i+1]
        w1, w2 = tau[i],   tau[i+1]
        s1, s2 = speed[i], speed[i+1]
        km, wm, sm = 0.5*(k1+k2), 0.5*(w1+w2), 0.5*(s1+s2)

        # RK4 for [F, r]
        K1F = rhs_F(F,            k1,  w1,  s1)
        K1r = rhs_r(F,            s1)

        F2  = F + 0.5*dt*K1F
        K2F = rhs_F(F2,           km,  wm,  sm)
        K2r = rhs_r(F2,           sm)

        F3  = F + 0.5*dt*K2F
        K3F = rhs_F(F3,           km,  wm,  sm)
        K3r = rhs_r(F3,           sm)

        F4  = F + dt*K3F
        K4F = rhs_F(F4,           k2,  w2,  s2)
        K4r = rhs_r(F4,           s2)

        F = F + (dt/6.0)*(K1F + 2*K2F + 2*K3F + K4F)
        r = r + (dt/6.0)*(K1r + 2*K2r + 2*K3r + K4r)

        if reorthonormalize:
            F = reortho(F)

        T[i+1], Nn[i+1], B[i+1] = F[:,0], F[:,1], F[:,2]
        X[i+1] = r

    return T, Nn, B, X

# # during integration we assume kappa[i] is constant in the duration t[i] to t[i+1]
# def integrate_frenet_serret(kappa, tau, speed, t, F0, x0):
#     num_steps = kappa.shape[0]-1
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
#         x_next = xs[-1] + h * speed_i * T
#         xs.append(x_next)

#     Fs = np.stack(Fs, axis = 0)
#     return Fs[..., 0], Fs[..., 1], Fs[..., 2], np.stack(xs, axis = 0)

F0 = np.column_stack([T[0], N[0], B[0]])  # shape [3,3]
x0 = np.array([x_fine[0], y_fine[0], z_fine[0]])
T_recon, N_recon, B_recon, poss_recon = integrate_frenet_serret(curvature, torsion, speed, t_fine, F0, x0)

write_obj_file_list([poss_recon], "tmp_recon.obj")
write_strand_frames(poss_recon, T_recon, N_recon, B_recon, filename_prefix = "tmp_recon_", frame_scale = 0.01)