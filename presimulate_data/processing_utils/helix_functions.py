import numpy as np
from utils.function_derivatives import (
    constant_function,
    constant_function_derivative,
    get_z,
    get_z_prime,
    get_x,
    get_y,
    get_x_twist,
    get_y_twist,
    get_z_twist,
    x_prime_twist,
    y_prime_twist,
    z_prime_twist,
    x_double_prime_twist,
    y_double_prime_twist,
    z_double_prime_twist,
    get_parameter_derivative,
    get_x_double_prime,
    get_y_double_prime,
    get_z_double_prime
)

def get_theta(wavelength, t):
    wavelength_at_t = constant_function(t, wavelength)
    z_at_t = get_z(t)
    return (2 * np.pi * z_at_t) / wavelength_at_t
   
def get_theta_prime(wavelength, t):
    z_at_t = get_z(t)
    wavelength_at_t = constant_function(t, wavelength)
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    return (2 * np.pi * (wavelength_at_t - (z_at_t * wavelength_prime_at_t))) / wavelength_at_t**2

def get_theta_double_prime(wavelength, t):
    z_at_t = get_z(t)
    wavelength_at_t = constant_function(t, wavelength)
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    wavelength_double_prime_at_t = get_parameter_derivative(t, wavelength_prime_at_t)
    z_at_t = z_at_t.reshape(-1, 1)
    wavelength_at_t = wavelength_at_t.reshape(-1, 1)
    wavelength_prime_at_t = wavelength_prime_at_t.reshape(-1, 1)
    wavelength_double_prime_at_t = wavelength_double_prime_at_t.reshape(-1, 1)
    
    s1 = -2 * np.pi * z_at_t * wavelength_double_prime_at_t * (wavelength_at_t ** 2)
    s2 = -4 * np.pi * wavelength_at_t * wavelength_prime_at_t
    s3 = wavelength_at_t - (z_at_t * wavelength_prime_at_t)
    s4 = wavelength_at_t ** 4

    return (s1 + (s2 * s3)) / s4

def get_e1_at_t(t, radius, wavelength):
    wavelength_at_t = constant_function(t, wavelength)
    radius_at_t = constant_function(t, radius)
    z_at_t = get_z(t)
    wavelength_prime_at_t = constant_function_derivative(t)
    radius_prime_at_t = constant_function_derivative(t)
    
    theta = (2 * np.pi * z_at_t) / wavelength_at_t
    theta_prime = (2 * np.pi * (wavelength_at_t - (z_at_t * wavelength_prime_at_t))) / wavelength_at_t**2
    
    x_prime = (radius_prime_at_t * np.cos(theta)) - (radius_at_t * np.sin(theta) * theta_prime)
    y_prime = (radius_prime_at_t * np.sin(theta)) + (radius_at_t * np.cos(theta) * theta_prime)
    z_prime = get_z_prime(t)
    
    x_prime = x_prime.reshape(-1, 1)
    y_prime = y_prime.reshape(-1, 1)
    z_prime = z_prime.reshape(-1, 1)
    
    e_1 = np.concatenate([x_prime, y_prime, z_prime], axis=1)
    e_1 = e_1 / np.linalg.norm(e_1, axis=1)[:, np.newaxis]
    return e_1

def get_e2_at_t(t, e_1, radius, wavelength):
    theta_at_t = get_theta(wavelength, t)
    theta_prime_at_t = get_theta_prime(wavelength, t)
    theta_double_prime_at_t = get_theta_double_prime(wavelength, t)
    
    x_2_prime = get_x_2_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    y_2_prime = get_y_2_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    z_2_prime = get_z_2_prime(t).reshape(-1, 1)

    e_2 = np.concatenate([x_2_prime, y_2_prime, z_2_prime], axis=1)
    e_2 = e_2 - np.sum(e_1 * e_2, axis=1)[:, np.newaxis] * e_1
    e_2 = e_2 / np.linalg.norm(e_2, axis=1)[:, np.newaxis]
    return e_2

def get_e3_at_t(e_1, e_2):
    e_3 = np.cross(e_1, e_2)
    return e_3
    
def get_e1_twist_at_t(t, radius, wavelength, twist_amt, twist_t, tangent_vec):
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    radius_prime_at_t = constant_function_derivative(t, radius)

    x_prime = x_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec)
    y_prime = y_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec)
    z_prime = z_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec)
    
    x_prime = x_prime.reshape(-1, 1)
    y_prime = y_prime.reshape(-1, 1)
    z_prime = z_prime.reshape(-1, 1)
    
    e_1 = np.concatenate([x_prime, y_prime, z_prime], axis=1)
    e_1 = e_1 / np.linalg.norm(e_1, axis=1)[:, np.newaxis]
    return e_1

def get_e2_twist_at_t(t, e_1, radius, wavelength, twist_amt, twist_t, tangent_vec):
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    radius_prime_at_t = constant_function_derivative(t, radius)
    theta_at_t = get_theta(wavelength, t)
    theta_prime_at_t = get_theta_prime(wavelength, t)
    theta_double_prime_at_t = get_theta_double_prime(wavelength, t) 
    theta_at_t = theta_at_t.reshape(-1, 1)
    theta_prime_at_t = theta_prime_at_t.reshape(-1, 1)
    theta_double_prime_at_t = theta_double_prime_at_t.reshape(-1, 1)
    t = t.reshape(-1, 1)
    x_2_prime = x_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec)
    print(x_2_prime.shape)
    y_2_prime = y_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec)
    print(y_2_prime.shape)  
    z_2_prime = z_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec)
    print(z_2_prime.shape)

    e_2 = np.concatenate([x_2_prime, y_2_prime, z_2_prime], axis=1)
    e_2 = e_2 - np.sum(e_1 * e_2, axis=1)[:, np.newaxis] * e_1
    e_2 = e_2 / np.linalg.norm(e_2, axis=1)[:, np.newaxis]
    return e_2

def generate_helix(z_vals, radius, wavelength):
    r = constant_function(z_vals, radius)
    w = constant_function(z_vals, wavelength)
    x = get_x(z_vals, r, w)
    y = get_y(z_vals, r, w)
    z = get_z(z_vals)
    return x, y, z

def generate_twisted_helix(z_vals, radius, wavelength, twist_amt, twist_t):
    
    # Get the tangent vectors (e_1) at all points
    e_1 = get_e1_at_t(z_vals, radius, wavelength)
    
    twist_t = int(twist_t * len(z_vals))
    # Use the explicit twist formulas
    x = get_x_twist(z_vals, twist_amt, twist_t, radius, wavelength, e_1)
    y = get_y_twist(z_vals, twist_amt, twist_t, radius, wavelength, e_1)
    z = get_z_twist(z_vals, twist_amt, twist_t, radius, wavelength, e_1)
    
    return x, y, z

def generate_piecewise_helix(total_points, helix):
    # Create evenly spaced points along the helix
    t = np.linspace(0, 1, total_points)
    # Linear interpolation between control points
    indices = np.minimum((t * (len(helix) - 1)).astype(int), len(helix) - 2)
    fracs = (t * (len(helix) - 1)) - indices
    
    # Get interpolated points
    p0 = helix[indices]
    p1 = helix[indices + 1]
    interpolated = p0 + (p1 - p0) * fracs.reshape(-1, 1)
    
    # Split back into x,y,z
    x = interpolated[:, 0:1]
    y = interpolated[:, 1:2] 
    z = interpolated[:, 2:3]
    return x, y, z

def add_helix_twist(x, y, z, tangent, twist):
    t = z.copy()
    # Get nonzero indices
    nonzero_indices = np.where(twist != 0)[0]

    # For each nonzero index, add the twist to the corresponding x, y, z
    for index in nonzero_indices:
        twist_amt = twist[index]
        twist_dir = tangent[index]
        twist_dir = twist_dir / np.linalg.norm(twist_dir)
        # Create quaternion from twist direction and amount
        angle = twist_amt / 2.0
        sin_angle = np.sin(angle)
        qw = np.cos(angle)
        qx = twist_dir[0] * sin_angle
        qy = twist_dir[1] * sin_angle 
        qz = twist_dir[2] * sin_angle
        
        # Create rotation matrix from quaternion
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        # Apply rotation to point
        point = np.array([x[index], y[index], z[index]])
        rotated = R @ point
        x[index] = rotated[0]
        y[index] = rotated[1] 
        z[index] = rotated[2]

    # Return the new x, y, z
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(twist.shape)
    exit()
    return x, y, z

def add_twist(x, y, z, e_1, twist_amt, twist_t):
    # Convert to numpy arrays if they aren't already
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    z = np.array(z).flatten()
    
    # Get the tangent vector at the twist point
    tangent = e_1[twist_t]
    tangent = tangent / np.linalg.norm(tangent)
    
    # EXPLICIT FORMULAS FOR TWISTED COORDINATES:
    # Given point P = (x, y, z) and rotation axis k = (kₓ, kᵧ, kᵤ) with angle θ:
    # 
    # x' = x + sin(θ)[kᵧ·z - kᵤ·y] + (1-cos(θ))[kₓ(kₓ·x + kᵧ·y + kᵤ·z) - x]
    # y' = y + sin(θ)[kᵤ·x - kₓ·z] + (1-cos(θ))[kᵧ(kₓ·x + kᵧ·y + kᵤ·z) - y]  
    # z' = z + sin(θ)[kₓ·y - kᵧ·x] + (1-cos(θ))[kᵤ(kₓ·x + kᵧ·y + kᵤ·z) - z]
    #
    # Where kₓ, kᵧ, kᵤ are the components of the normalized tangent vector
    # and θ is the twist amount (e.g., π for 180° twist)
    
    # EXPLICIT HELIX FORMULAS AFTER TWIST (for t > twist_t):
    # Let the original helix be: x₀(t) = r(t)cos(θ_helix(t)), y₀(t) = r(t)sin(θ_helix(t)), z₀(t) = z(t)
    # where θ_helix(t) = 2π·z(t)/λ(t) is the helix angle
    # Let the tangent at twist point be: k = (kₓ, kᵧ, kᵤ)
    # Let the reference point be: P_ref = (x₀(twist_t), y₀(twist_t), z₀(twist_t))
    # Let θ_twist be the twist amount (e.g., π for 180° twist)
    #
    # For t > twist_t:
    # x(t) = x₀(t) + sin(θ_twist)[kᵧ·z₀(t) - kᵤ·y₀(t)] + (1-cos(θ_twist))[kₓ(kₓ·x₀(t) + kᵧ·y₀(t) + kᵤ·z₀(t)) - x₀(t)]
    #        + (x₀(twist_t) - x₀'(twist_t))  # C₀ continuity correction
    #
    # y(t) = y₀(t) + sin(θ_twist)[kᵤ·x₀(t) - kₓ·z₀(t)] + (1-cos(θ_twist))[kᵧ(kₓ·x₀(t) + kᵧ·y₀(t) + kᵤ·z₀(t)) - y₀(t)]
    #        + (y₀(twist_t) - y₀'(twist_t))  # C₀ continuity correction
    #
    # z(t) = z₀(t) + sin(θ_twist)[kₓ·y₀(t) - kᵧ·x₀(t)] + (1-cos(θ_twist))[kᵤ(kₓ·x₀(t) + kᵧ·y₀(t) + kᵤ·z₀(t)) - z₀(t)]
    #        + (z₀(twist_t) - z₀'(twist_t))  # C₀ continuity correction
    #
    # Where x₀'(twist_t), y₀'(twist_t), z₀'(twist_t) are the rotated reference point coordinates
    
    # Create rotation matrix for rotation around tangent
    # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the rotation axis
    
    # Create skew-symmetric matrix K for the tangent vector
    kx, ky, kz = tangent
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]])
    
    # Identity matrix
    I = np.eye(3)
    
    # Rotation matrix using Rodrigues' formula
    R = I + np.sin(twist_amt) * K + (1 - np.cos(twist_amt)) * (K @ K)
    
    # Get the reference point (the point at twist_t) before rotation
    ref_point_before = np.array([x[twist_t], y[twist_t], z[twist_t]])
    
    # Apply rotation to all points from twist_t onwards
    for i in range(twist_t, len(x)):
        point = np.array([x[i], y[i], z[i]])
        rotated_point = R @ point
        x[i] = rotated_point[0]
        y[i] = rotated_point[1]
        z[i] = rotated_point[2]
    
    # Get the reference point after rotation
    ref_point_after = np.array([x[twist_t], y[twist_t], z[twist_t]])
    
    # Calculate the translation needed to maintain C₀ continuity
    translation = ref_point_before - ref_point_after
    
    # Apply the translation to all rotated points
    for i in range(twist_t, len(x)):
        x[i] += translation[0]
        y[i] += translation[1]
        z[i] += translation[2]
    
    return x, y, z