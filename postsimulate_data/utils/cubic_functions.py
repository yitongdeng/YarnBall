import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline

    
def get_global_cubic_parameters(x, y, z, t):
    """
    x, y, z: (n_points,)
    t: (n_points,)
    Returns:
        q_list: (n_segments, 3)
        a_list: (n_segments, 3)
        b_list: (n_segments, 3)
        c_list: (n_segments, 3)
    """
    q_list = []
    a_list = []
    b_list = []
    c_list = []
    t_sections = [t[i:i+4] for i in range(len(t)-3)]
    x_sections = [x[i:i+4] for i in range(len(x)-3)]
    y_sections = [y[i:i+4] for i in range(len(y)-3)]
    z_sections = [z[i:i+4] for i in range(len(z)-3)]
    for i in range(len(t_sections)):
        t0 = t_sections[i][0]
        t1 = t_sections[i][1]
        t2 = t_sections[i][2]
        t3 = t_sections[i][3]
        pt0 = np.array([x_sections[i][0], y_sections[i][0], z_sections[i][0]])
        pt1 = np.array([x_sections[i][1], y_sections[i][1], z_sections[i][1]])
        pt2 = np.array([x_sections[i][2], y_sections[i][2], z_sections[i][2]])
        pt3 = np.array([x_sections[i][3], y_sections[i][3], z_sections[i][3]])
        
        A_x = np.array([
            [1, t0, t0**2, t0**3], 
            [1, t1, t1**2, t1**3], 
            [1, t2, t2**2, t2**3], 
            [1, t3, t3**2, t3**3]])
        A_y = np.array([
            [1, t0, t0**2, t0**3], 
            [1, t1, t1**2, t1**3], 
            [1, t2, t2**2, t2**3], 
            [1, t3, t3**2, t3**3]])
        A_z = np.array([
            [1, t0, t0**2, t0**3], 
            [1, t1, t1**2, t1**3], 
            [1, t2, t2**2, t2**3], 
            [1, t3, t3**2, t3**3]])
        B_x = np.array([pt0[0], pt1[0], pt2[0], pt3[0]])
        B_y = np.array([pt0[1], pt1[1], pt2[1], pt3[1]])
        B_z = np.array([pt0[2], pt1[2], pt2[2], pt3[2]])
        
        solution_x = np.linalg.solve(A_x, B_x)
        solution_y = np.linalg.solve(A_y, B_y)
        solution_z = np.linalg.solve(A_z, B_z)
        
        q = np.array([solution_x[0], solution_y[0], solution_z[0]])
        a = np.array([solution_x[1], solution_y[1], solution_z[1]])
        b = np.array([solution_x[2], solution_y[2], solution_z[2]])
        c = np.array([solution_x[3], solution_y[3], solution_z[3]])
        
        q_list.append(q)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
    q_list = np.concatenate((q_list[0].reshape(1, 3), q_list, q_list[-1].reshape(1, 3)), axis=0)
    a_list = np.concatenate((a_list[0].reshape(1, 3), a_list, a_list[-1].reshape(1, 3)), axis=0)
    b_list = np.concatenate((b_list[0].reshape(1, 3), b_list, b_list[-1].reshape(1, 3)), axis=0)
    c_list = np.concatenate((c_list[0].reshape(1, 3), c_list, c_list[-1].reshape(1, 3)), axis=0)
   
    return np.array(q_list), np.array(a_list), np.array(b_list), np.array(c_list)

def smooth_global_cubics(q_list, a_list, b_list, c_list):
    q_total = q_list[1:-1].copy()
    q_total[1:] += (q_list[1:-2] * .5)
    q_total[:-1] += (q_list[2:-1] * .5)
    q_total[1:-1] = q_total[1:-1] / 2
    q_total[0] = q_total[0] / 1.5
    q_total[-1] = q_total[-1] / 1.5
    q_list[1:-1] = q_total
    
    a_total = a_list[1:-1].copy()
    a_total[1:] += (a_list[1:-2] * .5)
    a_total[:-1] += (a_list[2:-1] * .5)
    a_total[1:-1] = a_total[1:-1] / 2
    a_total[0] = a_total[0] / 1.5
    a_total[-1] = a_total[-1] / 1.5
    a_list[1:-1] = a_total

    b_total = b_list[1:-1].copy()
    b_total[1:] += (b_list[1:-2] * .5)
    b_total[:-1] += (b_list[2:-1] * .5)
    b_total[1:-1] = b_total[1:-1] / 2
    b_total[0] = b_total[0] / 1.5
    b_total[-1] = b_total[-1] / 1.5
    b_list[1:-1] = b_total

    c_total = c_list[1:-1].copy()
    c_total[1:] += (c_list[1:-2] * .5)
    c_total[:-1] += (c_list[2:-1] * .5)
    c_total[1:-1] = c_total[1:-1] / 2
    c_total[0] = c_total[0] / 1.5
    c_total[-1] = c_total[-1] / 1.5
    c_list[1:-1] = c_total
    return q_list, a_list, b_list, c_list

def get_cubic_parameters_at_t(q_list, a_list, b_list, c_list, original_t_vals, new_t_vals):
    partitioned_t_vals = [] 
    partitioned_t_vals.append(new_t_vals[new_t_vals <= original_t_vals[1]])
    for i in range(1, len(original_t_vals)-1):
        partitioned_t_vals.append(new_t_vals[(new_t_vals > original_t_vals[i]) & (new_t_vals <= original_t_vals[i+1])])

    
    weights = []
    prev_start_idx = 0
    start_idx = 0
    for i in range(len(partitioned_t_vals)):
        weight_arr = np.zeros(len(new_t_vals))
        end_idx = start_idx + len(partitioned_t_vals[i])
        weight_arr[start_idx:end_idx] = 1.0
        if i > 0:
            weight_arr[prev_start_idx:start_idx] = np.linspace(0.0, 1.0, start_idx - prev_start_idx, endpoint=True)
        if i < len(partitioned_t_vals) - 1:
            weight_arr[end_idx:end_idx + len(partitioned_t_vals[i+1])] = np.linspace(1.0, 0.0, len(partitioned_t_vals[i+1]), endpoint=True)
        weights.append(weight_arr)
        prev_start_idx = start_idx
        start_idx = end_idx
    
    weights = np.array(weights)
        
def get_scipy_cubic_parameters(x, y, z):
    pts = np.array([x, y, z]).T
    s = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    s = np.array([0] + list(np.cumsum(s)))
    
    x_cs = CubicSpline(s, x)
    y_cs = CubicSpline(s, y)
    z_cs = CubicSpline(s, z)
    q_cs = np.array([x_cs.c[0], y_cs.c[0], z_cs.c[0]])
    a_cs = np.array([x_cs.c[1], y_cs.c[1], z_cs.c[1]])
    b_cs = np.array([x_cs.c[2], y_cs.c[2], z_cs.c[2]])
    c_cs = np.array([x_cs.c[3], y_cs.c[3], z_cs.c[3]])
    return q_cs.T, a_cs.T, b_cs.T, c_cs.T, s

def get_piecewise_cubic_parameters(x, y, z):
    cubic_segments = [np.stack([x[i:i+4], y[i:i+4], z[i:i+4]], axis=1) for i in range(len(x)-3)]
    # Initialize lists to store parameters for each cubic segment
    q_list = []
    a_list = []
    b_list = []
    c_list = []

    # For each cubic segment
    for segment in cubic_segments:
        # Get points p0, p1, p2, p3 from segment
        p0 = segment[0]
        p1 = segment[1] 
        p2 = segment[2]
        p3 = segment[3]

        
        # Solve this system for a, b, c
        q = p0
        
        # Coefficient matrix for the system
        A = np.array([
            [1/3, 1/9, 1/27],
            [2/3, 4/9, 8/27], 
            [1, 1, 1]
        ])
        
        # Right-hand side
        B = np.array([
            p1 - p0,
            p2 - p0,
            p3 - p0
        ])
        
        # Solve for a, b, c
        solution = np.linalg.solve(A, B)
        a = solution[0]
        b = solution[1] 
        c = solution[2]
 
        # Store parameters
        q_list.append(q)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)

    # Convert lists to arrays
    q = np.array(q_list)
    a = np.array(a_list)
    b = np.array(b_list) 
    c = np.array(c_list)

    return q, a, b, c

def prepare_global_cubic_samples(q, a, b, c, arc_len, n_samples):
    t = np.array([np.linspace(arc_len[i], arc_len[i+1], n_samples) for i in range(len(arc_len) - 1)]).reshape(-1,1)
    q = q.repeat(n_samples, axis=0)
    a = a.repeat(n_samples, axis=0)
    b = b.repeat(n_samples, axis=0)
    c = c.repeat(n_samples, axis=0)
    
    return t, q, a, b, c
    
def prepare_cubic_samples(q, a, b, c, n_samples):
    t = np.linspace(0, 1, n_samples).reshape(1, -1)
    t = t.repeat(len(q), axis=0)
    t = np.concatenate([t[0].reshape(1, -1) / 3, ((t + 1) / 3), (t[-1].reshape(1, -1) + 2) / 3], axis=0)

    q = np.concatenate([q[0].reshape(1, -1), q, q[-1].reshape(1, -1)], axis=0)
    a = np.concatenate([a[0].reshape(1, -1), a, a[-1].reshape(1, -1)], axis=0)
    b = np.concatenate([b[0].reshape(1, -1), b, b[-1].reshape(1, -1)], axis=0)
    c = np.concatenate([c[0].reshape(1, -1), c, c[-1].reshape(1, -1)], axis=0)
    return t, q, a, b, c

def get_cubic_points(t, q, a, b, c):
    # Generate points along the cubic curve
    x_cubic = q[:,0].reshape(-1, 1) + a[:,0].reshape(-1, 1)*t + b[:,0].reshape(-1, 1)*t**2 + c[:,0].reshape(-1, 1)*t**3
    y_cubic = q[:,1].reshape(-1, 1) + a[:,1].reshape(-1, 1)*t + b[:,1].reshape(-1, 1)*t**2 + c[:,1].reshape(-1, 1)*t**3
    z_cubic = q[:,2].reshape(-1, 1) + a[:,2].reshape(-1, 1)*t + b[:,2].reshape(-1, 1)*t**2 + c[:,2].reshape(-1, 1)*t**3
    return x_cubic, y_cubic, z_cubic
    
def get_e1_cubic(a, b, c, t):
    dx, dy, dz = get_cubic_prime(a, b, c, t)
    e1 = np.stack([dx, dy, dz], axis=1).reshape(-1, 3)
    return e1

def get_e2_cubic_unnormalized(b, c, t, e_1):
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    e_2 = np.concatenate([ddx, ddy, ddz], axis=1).reshape(-1, 3)
    e_2 = e_2 - np.sum(e_1 * e_2, axis=1)[:, np.newaxis] * e_1
    return e_2

def get_e2_cubic(b, c, t, e_1):
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    e_2 = np.concatenate([ddx, ddy, ddz], axis=1).reshape(-1, 3)
    e_2 = e_2 - np.sum(e_1 * e_2, axis=1)[:, np.newaxis] * e_1
    e_2 = e_2 / np.linalg.norm(e_2, axis=1)[:, np.newaxis]
    return e_2

def get_e3_cubic(e1, e2):
    e3 = np.cross(e1, e2)
    return e3

def get_cubic_prime(a, b, c, t):
    dx = a[:,0].reshape(-1, 1) + 2*b[:,0].reshape(-1, 1)*t + 3*c[:,0].reshape(-1, 1)*t**2
    dy = a[:,1].reshape(-1, 1) + 2*b[:,1].reshape(-1, 1)*t + 3*c[:,1].reshape(-1, 1)*t**2
    dz = a[:,2].reshape(-1, 1) + 2*b[:,2].reshape(-1, 1)*t + 3*c[:,2].reshape(-1, 1)*t**2
    # Normalize the tangent vectors
    norms = np.sqrt(dx**2 + dy**2 + dz**2)
    dx = dx / norms
    dy = dy / norms
    dz = dz / norms
    return dx, dy, dz

def get_cubic_second_derivative(b, c, t):
    ddx = 2*b[:,0].reshape(-1, 1) + 6*c[:,0].reshape(-1, 1)*t
    ddy = 2*b[:,1].reshape(-1, 1) + 6*c[:,1].reshape(-1, 1)*t
    ddz = 2*b[:,2].reshape(-1, 1) + 6*c[:,2].reshape(-1, 1)*t
    return ddx, ddy, ddz

def get_e1_prime_magnitude(a, b, c, t):
    dx, dy, dz = get_cubic_prime(a, b, c, t)
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    first_deriv = np.stack([dx, dy, dz], axis=2)
    second_deriv = np.stack([ddx, ddy, ddz], axis=2)
    first_deriv = first_deriv.reshape(-1, 3)
    second_deriv = second_deriv.reshape(-1, 3)
    
    e_1_prime = second_deriv - (first_deriv * np.sum(first_deriv * second_deriv, axis=1)[:,np.newaxis] / (np.linalg.norm(first_deriv, axis=1)**2).reshape(-1, 1))
    e_1_prime = e_1_prime / np.linalg.norm(first_deriv, axis=1).reshape(-1, 1)
    return np.linalg.norm(e_1_prime, axis=1).reshape(-1, 1)
   
def get_curvature(q, a, b, c, t):
    dx, dy, dz = get_cubic_prime(a, b, c, t)
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    first_deriv = np.stack([dx, dy, dz], axis=2)
    second_deriv = np.stack([ddx, ddy, ddz], axis=2)
    e_1 = get_e1_cubic(a, b, c, t)
    e_2 = get_e2_cubic_unnormalized(b, c, t, e_1)
    curvature_calc = np.cross(first_deriv, second_deriv)
    curvature_calc = np.linalg.norm(curvature_calc, axis=2)**2
    curvature_calc = curvature_calc / np.linalg.norm(first_deriv, axis=2)**4
    
    first_deriv = first_deriv.reshape(-1, 3)
    second_deriv = second_deriv.reshape(-1, 3)
    
    curvature_calc = curvature_calc / np.linalg.norm(e_2, axis=1).reshape(-1, 1)

    e_1_prime = second_deriv - (first_deriv * np.sum(first_deriv * second_deriv, axis=1)[:,np.newaxis] / (np.linalg.norm(first_deriv, axis=1)**2).reshape(-1, 1))
    e_1_prime = e_1_prime / np.linalg.norm(first_deriv, axis=1).reshape(-1, 1)
    curvature_wiki = np.linalg.norm(e_1_prime, axis=1).reshape(-1, 1) / np.linalg.norm(first_deriv, axis=1).reshape(-1, 1)
    
    diff = curvature_calc - curvature_wiki

    return curvature_calc


def get_torsion(q, a, b, c, t):
    dx, dy, dz = get_cubic_prime(a, b, c, t)
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    first_deriv = np.stack([dx, dy, dz], axis=2)
    second_deriv = np.stack([ddx, ddy, ddz], axis=2)
    e_1 = get_e1_cubic(a, b, c, t)
    e_2 = get_e2_cubic_unnormalized(b, c, t, e_1)
    torsion_calc = np.cross(first_deriv, second_deriv)
    torsion_calc = np.linalg.norm(torsion_calc, axis=2)**2
    torsion_calc = torsion_calc / np.linalg.norm(first_deriv, axis=2)**4
    torsion_calc = torsion_calc / np.linalg.norm(e_2, axis=1).reshape(-1, 1)
    return torsion_calc

def get_piecewise_hermite_parameters(x, y, z):
    """
    Returns piecewise Hermite cubic parameters for each segment using global parameterization.
    Returns:
        P0s, P1s: (n_segments, 3) endpoints for each segment
        m0s, m1s: (n_segments, 3) tangents at endpoints for each segment
        s: (n_points,) global parameter (normalized arc length)
    """
    points = np.stack([x, y, z], axis=1)
    n = len(points)
    # Compute global parameter s (normalized arc length)
    s = np.linalg.norm(points[1:] - points[:-1], axis=1)
    s = np.concatenate([[0], np.cumsum(s)])
    s = s / s[-1]
    # Compute tangents (finite difference)
    tangents = np.zeros_like(points)
    tangents[1:-1] = (points[2:] - points[:-2]) / (s[2:, None] - s[:-2, None])
    tangents[0] = (points[1] - points[0]) / (s[1] - s[0])
    tangents[-1] = (points[-1] - points[-2]) / (s[-1] - s[-2])
    # For each segment, store endpoints and tangents
    P0s = points[:-1]
    P1s = points[1:]
    m0s = tangents[:-1]
    m1s = tangents[1:]
    return P0s, P1s, m0s, m1s, s