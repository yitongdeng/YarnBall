import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

def evaluate_spline_batch(x_arr, y_arr, z_arr, n_fine_sampling):
    x_fine_arr = []
    y_fine_arr = []
    z_fine_arr = []
    
    for i in range(len(x_arr)):
        x = x_arr[i]
        y = y_arr[i]
        z = z_arr[i]

        arc_len = parameterize_arc_length(x_arr[0], y_arr[0], z_arr[0])
        t_fine = np.linspace(0, arc_len[-1], n_fine_sampling)
        x_cs, y_cs, z_cs = get_scipy_spline(x, y, z, arc_len, 3)
        x_fine, y_fine, z_fine = evaluate_spline(x_cs, y_cs, z_cs, t_fine)
        x_fine_arr.append(x_fine)
        y_fine_arr.append(y_fine)
        z_fine_arr.append(z_fine)
    return x_fine_arr, y_fine_arr, z_fine_arr

def parameterize_arc_length(x, y, z):
    curve = np.array([x, y, z])
    arc_len = np.cumsum(np.linalg.norm(curve[:, 1:] - curve[:, :-1], axis=0))

    arc_len = np.concatenate([[0], arc_len])
    arc_len = arc_len / arc_len[-1]
    return arc_len


def get_scipy_spline(x, y, z, arc_len, degree):
    s = arc_len
    
    x_cs = make_interp_spline(s, x, k=degree)
    y_cs = make_interp_spline(s, y, k=degree)
    z_cs = make_interp_spline(s, z, k=degree)

    return x_cs, y_cs, z_cs

def evaluate_spline(x_cs, y_cs, z_cs, t_fine):
    x_fine = x_cs(t_fine)
    y_fine = y_cs(t_fine)
    z_fine = z_cs(t_fine)
    return x_fine, y_fine, z_fine

def evaluate_spline_derivatives(x_cs, y_cs, z_cs, t_fine):
    """
    Evaluate the first and second derivatives of the splines at t_fine.
    If smoothed arrays are provided for the current derivative, use them for the first derivative.
    Always use the spline's second derivative for the next order.
    Returns:
        x_prime_fine, y_prime_fine, z_prime_fine, x_second_fine, y_second_fine, z_second_fine
    """
    # For the next order, always use the derivative of the spline
    x_second_fine = x_cs.derivative()(t_fine)
    y_second_fine = y_cs.derivative()(t_fine)
    z_second_fine = z_cs.derivative()(t_fine)
    return x_second_fine, y_second_fine, z_second_fine

def get_frenet_frame(x_fine, y_fine, z_fine, x_prime_fine, y_prime_fine, z_prime_fine, x_second_fine, y_second_fine, z_second_fine, t_fine, visualize_all=False):
    p = np.array([x_fine, y_fine, z_fine]).T
    v = np.array([x_prime_fine, y_prime_fine, z_prime_fine]).T
    a = np.array([x_second_fine, y_second_fine, z_second_fine]).T
    if visualize_all:
        # Plot acceleration vector over parameter t
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(a[:,0], a[:,1], a[:,2], label='Acceleration Vector')
        ax.scatter([0], [0], [0], color='green', marker='*', s=100, label='Origin')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Acceleration Vector in 3D')
        
        # Set equal aspect ratio
        max_range = np.array([a[:,0].max()-a[:,0].min(), a[:,1].max()-a[:,1].min(), a[:,2].max()-a[:,2].min()]).max()
        mid_x = (a[:,0].max()+a[:,0].min()) * 0.5
        mid_y = (a[:,1].max()+a[:,1].min()) * 0.5
        mid_z = (a[:,2].max()+a[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
        ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
        ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
        ax.set_box_aspect([1,1,1])
        
        ax.legend()
        plt.show()

    # Frenet frame: Tangent (T), Normal (N), Binormal (B)
    v_norm = np.linalg.norm(v, axis=1)
    T = v / v_norm[:, np.newaxis]
    a_dot_T = np.sum(a * T, axis=1)[:, np.newaxis]
    N_unnormalized = a - a_dot_T * T
    N_norm = np.linalg.norm(N_unnormalized, axis=1)
    zero_mask = N_norm < 1e-10
    N_unnormalized[zero_mask] = np.array([1, 0, 0])
    N_norm[zero_mask] = 1.0
    N = N_unnormalized / N_norm[:, np.newaxis]
    B = np.cross(T, N)
    B_norm = np.linalg.norm(B, axis=1)
    B_norm = np.where(B_norm > 1e-10, B_norm, 1.0)
    B = B / B_norm[:, np.newaxis]
    
    # Standard curvature formula
    e_1_prime = N_unnormalized / np.linalg.norm(v, axis=1)[:, np.newaxis]
    curvature = np.linalg.norm(e_1_prime, axis=1) / np.linalg.norm(v, axis=1)
    
    if visualize_all:
        # Plot e_1_prime_dot_e_2, velocity, and curvature over t_fine
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        ax1.plot(t_fine, np.linalg.norm(v, axis=1))
        ax1.set_title('velocity')
        ax1.set_xlabel('t')
        ax1.set_ylabel('Value')
        ax1.grid(True)
        
        ax2.plot(t_fine, np.linalg.norm(a, axis=1))
        ax2.set_title("acceleration")
        ax2.set_xlabel('t') 
        ax2.set_ylabel('Value')
        ax2.grid(True)
        
        ax3.plot(t_fine, np.linalg.norm(e_1_prime, axis=1))
        ax3.set_title('e_1_prime')
        ax3.set_xlabel('t')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
    # Frenet frame method for torsion
    # e_2: [N, 3], e_3: [N, 3], v: [N, 3], t_fine: parameter array
    dt = np.gradient(t_fine)
    e2_prime = np.gradient(N, axis=0) / dt[:, None]
    numerator_frenet = np.sum(e2_prime * B, axis=1)
    v_norm = np.linalg.norm(v, axis=1)
    torsion = numerator_frenet / v_norm
    return T, N, B, curvature, torsion

def smooth_curvature_torsion(e_1, e_2, e_3, t_fine, v):
    e_1_prime = np.gradient(e_1, axis=0) / np.gradient(t_fine)[:, None]
    e_2_prime = np.gradient(e_2, axis=0) / np.gradient(t_fine)[:, None]
    curvature = np.sum(e_1_prime * e_2, axis=1) / np.linalg.norm(v, axis=1)
    torsion = np.sum(e_2_prime * e_3, axis=1) / np.linalg.norm(v, axis=1)
    return curvature, torsion

