import numpy as np
from scipy.interpolate import make_interp_spline
from utils.spline_utils import evaluate_spline_derivatives, get_scipy_spline
import matplotlib.pyplot as plt

def get_quintic_torsion(x, y, z):
    x_cs, y_cs, z_cs = get_scipy_spline(x, y, z, np.arange(len(x)), 5)
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, torsion = evaluate_quintic_spline(x_cs, y_cs, z_cs, np.arange(len(x)))
    return torsion

def evaluate_quintic_spline(x_cs, y_cs, z_cs, t_fine):
    x_fine = x_cs(t_fine)
    y_fine = y_cs(t_fine)
    z_fine = z_cs(t_fine)

    # Derivatives for Frenet frame and curvature
    x_prime_fine = x_cs.derivative()(t_fine)
    y_prime_fine = y_cs.derivative()(t_fine)
    z_prime_fine = z_cs.derivative()(t_fine)
    x_second_fine = x_cs.derivative(2)(t_fine)
    y_second_fine = y_cs.derivative(2)(t_fine)
    z_second_fine = z_cs.derivative(2)(t_fine)
    x_third_fine = x_cs.derivative(3)(t_fine)
    y_third_fine = y_cs.derivative(3)(t_fine)
    z_third_fine = z_cs.derivative(3)(t_fine)

    p = np.array([x_fine, y_fine, z_fine]).T
    v = np.array([x_prime_fine, y_prime_fine, z_prime_fine]).T
    a = np.array([x_second_fine, y_second_fine, z_second_fine]).T
    a_prime = np.array([x_third_fine, y_third_fine, z_third_fine]).T

    # Standard curvature formula
    v_cross_a = np.cross(v, a)
    v_norm = np.linalg.norm(v, axis=1)
    curvature = np.linalg.norm(v_cross_a, axis=1) / (v_norm**3)
 
    # Frenet frame: Tangent (T), Normal (N), Binormal (B)
    T = v / v_norm[:, np.newaxis]
    a_dot_T = np.sum(a * T, axis=1)[:, np.newaxis]
    N = a - a_dot_T * T
    N_norm = np.linalg.norm(N, axis=1)
    zero_mask = N_norm < 1e-10
    N[zero_mask] = np.array([1, 0, 0])
    N_norm[zero_mask] = 1.0
    N = N / N_norm[:, np.newaxis]
    B = np.cross(T, N)
    B_norm = np.linalg.norm(B, axis=1)
    B_norm = np.where(B_norm > 1e-10, B_norm, 1.0)
    B = B / B_norm[:, np.newaxis]

    # Frenet frame method for torsion
    # e_2: [N, 3], e_3: [N, 3], v: [N, 3], t_fine: parameter array
    dt = np.gradient(t_fine)
    e2_prime = np.gradient(N, axis=0) / dt[:, None]
    numerator_frenet = np.sum(e2_prime * B, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    torsion = numerator_frenet / v_norm

    return x_fine, y_fine, z_fine, x_prime_fine, y_prime_fine, z_prime_fine, x_second_fine, y_second_fine, z_second_fine, x_third_fine, y_third_fine, z_third_fine, T, N, B, curvature, torsion
