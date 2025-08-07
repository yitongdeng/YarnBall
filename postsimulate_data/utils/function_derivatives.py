import numpy as np

def get_x(z_0_hat, radius, wavelength):
    return radius * np.cos(2 * np.pi * z_0_hat / wavelength)

def get_x_prime(t, radius, wavelength):
    wavelength_at_t = constant_function(t, wavelength)
    radius_at_t = constant_function(t, radius)
    z_at_t = get_z(t)
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    radius_prime_at_t = constant_function_derivative(t, radius)
    
    s1 = np.cos(2 * np.pi * z_at_t / wavelength_at_t)
    s2 = np.sin(2 * np.pi * z_at_t / wavelength_at_t)
    s3 = 2 * np.pi * radius_at_t
    s4 = (wavelength_at_t - (z_at_t * wavelength_prime_at_t))
    s5 = 1 / wavelength_at_t**2
    return (radius_prime_at_t * s1) - (s2 * s3 * s4 * s5)

def get_x_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t):
    r_at_t = constant_function(t, radius)
    r_prime_at_t = constant_function_derivative(t, radius)
    r_double_prime_at_t = get_parameter_derivative(t, r_prime_at_t)
    
    r_at_t, r_prime_at_t, r_double_prime_at_t = \
        r_at_t.reshape(-1, 1), r_prime_at_t.reshape(-1, 1), r_double_prime_at_t.reshape(-1, 1)
    theta_at_t, theta_prime_at_t, theta_double_prime_at_t = \
        theta_at_t.reshape(-1, 1), theta_prime_at_t.reshape(-1, 1), theta_double_prime_at_t.reshape(-1, 1)
    
    s1 = r_double_prime_at_t * np.cos(theta_at_t)
    s2 = -2 * r_prime_at_t * np.sin(theta_at_t) * theta_prime_at_t
    s3 = -1 * r_at_t * np.cos(theta_at_t) * (theta_prime_at_t ** 2)
    s4 = -1 * r_at_t * np.sin(theta_at_t) * theta_double_prime_at_t
    return (s1 + s2 + s3 + s4)
    
def get_x_twist(t, twist_amt, twist_t, radius, wavelength, e_1):
    #x(t) &= x_0(t) + sin(twist_amt) *(k_y z_0(t) - k_z y_0(t))
    #       + (1 - cos(twist_amt)) * (k_x * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - x_0(t))
    #       + (x_0(twist_t) - x_0'(twist_t))
    x_0_at_t = get_x(t, radius, wavelength)
    y_0_at_t = get_y(t, radius, wavelength)
    z_0_at_t = get_z(t)
    k = e_1[twist_t]
    
    # Get reference point (unchanged)
    x_0_ref = x_0_at_t[twist_t]
    y_0_ref = y_0_at_t[twist_t]
    z_0_ref = z_0_at_t[twist_t]
    
    # Calculate the rotated reference point for continuity correction
    s1_ref = x_0_ref
    s2_ref = np.sin(twist_amt) * ((k[1] * z_0_ref) - (k[2] * y_0_ref))
    s3_ref = (1 - np.cos(twist_amt)) * ((k[0] * ((k[0] * x_0_ref) + (k[1] * y_0_ref) + (k[2] * z_0_ref))) - x_0_ref)
    x_0_rotated_ref = s1_ref + s2_ref + s3_ref
    
    # Initialize result with original coordinates
    result = x_0_at_t.copy()
    
    # Apply twist only to points after twist_t
    for i in range(twist_t + 1, len(t)):
        # Calculate twisted coordinates
        s1 = x_0_at_t[i]
        s2 = np.sin(twist_amt) * ((k[1] * z_0_at_t[i]) - (k[2] * y_0_at_t[i]))
        s3 = (1 - np.cos(twist_amt)) * ((k[0] * ((k[0] * x_0_at_t[i]) + (k[1] * y_0_at_t[i]) + (k[2] * z_0_at_t[i]))) - x_0_at_t[i])
        # double check this: why aren't we moving the reference point?
        s4 = (x_0_ref - x_0_rotated_ref)  # Continuity correction
        
        result[i] = s1 + s2 + s3 + s4
    
    return result

def x_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec):
    #dx/dt =
    #x'_0(t) 
    #+ sin(twist_amt) * (k_y * z_0_prime(t) - k_z * y_0_prime(t))
    #+ (1 - cos(twist_amt)) * (k_x * (k_x * x_0_prime(t) + k_y * y_0_prime(t) + k_z * z_0_prime(t)) - x_0_prime(t))
    x_0_prime_at_t = get_x_prime(t, radius, wavelength)
    y_0_prime_at_t = get_y_prime(t, radius, wavelength)
    z_0_prime_at_t = get_z_prime(t)
    k = tangent_vec
    s1 = x_0_prime_at_t
    s2 = np.sin(twist_amt) * ((k[1] * z_0_prime_at_t) - (k[2] * y_0_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[0] * ((k[0] * x_0_prime_at_t) + (k[1] * y_0_prime_at_t) + (k[2] * z_0_prime_at_t))) - x_0_prime_at_t)
    return s1 + s2 + s3

def x_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec):
    #d^2x/dt^2 =
    #x''_0(t)
    #+ sin(twist_amt) * (k_y * z_0''_prime(t) - k_z * y_0''_prime(t))
    #+ (1 - cos(twist_amt)) * (k_x * (k_x * x_0''_prime(t) + k_y * y_0''_prime(t) + k_z * z_0''_prime(t)) - x_0''_prime(t))
    x_0_double_prime_at_t = get_x_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    y_0_double_prime_at_t = get_y_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    z_0_double_prime_at_t = get_z_double_prime(t)
    k = tangent_vec
    
    s1 = x_0_double_prime_at_t
    s2 = np.sin(twist_amt) * ((k[1] * z_0_double_prime_at_t) - (k[2] * y_0_double_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[0] * ((k[0] * x_0_double_prime_at_t) + (k[1] * y_0_double_prime_at_t) + (k[2] * z_0_double_prime_at_t))) - x_0_double_prime_at_t)
    return s1 + s2 + s3

def get_y(z_0_hat, radius, wavelength):
    return radius * np.sin(2 * np.pi * z_0_hat / wavelength)

def get_y_prime(t, radius, wavelength):
    wavelength_at_t = constant_function(t, wavelength)
    radius_at_t = constant_function(t, radius)
    z_at_t = get_z(t)
    wavelength_prime_at_t = constant_function_derivative(t, wavelength)
    radius_prime_at_t = constant_function_derivative(t, radius)
    s1 = np.sin(2 * np.pi * z_at_t / wavelength_at_t)
    s2 = np.cos(2 * np.pi * z_at_t / wavelength_at_t)
    s3 = 2 * np.pi * radius_at_t
    s4 = wavelength_at_t - (z_at_t * wavelength_prime_at_t)
    s5 = 1 / wavelength**2
    return (radius_prime_at_t * s1) + (s2 * s3 * s4 * s5)

def get_y_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t):
    r_at_t = constant_function(t, radius)
    r_prime_at_t = constant_function_derivative(t, radius)
    r_double_prime_at_t = get_parameter_derivative(t, r_prime_at_t)
    
    r_at_t, r_prime_at_t, r_double_prime_at_t = \
        r_at_t.reshape(-1, 1), r_prime_at_t.reshape(-1, 1), r_double_prime_at_t.reshape(-1, 1)
    theta_at_t, theta_prime_at_t, theta_double_prime_at_t = \
        theta_at_t.reshape(-1, 1), theta_prime_at_t.reshape(-1, 1), theta_double_prime_at_t.reshape(-1, 1)
        
    s1 = r_double_prime_at_t * np.sin(theta_at_t)
    s2 = 2 * r_prime_at_t * np.cos(theta_at_t) * theta_prime_at_t
    s3 = -1 * r_at_t * np.sin(theta_at_t) * (theta_prime_at_t ** 2)
    s4 = r_at_t * np.cos(theta_at_t) * theta_double_prime_at_t
    
    return (s1 + s2 + s3 + s4)

def get_y_twist(t, twist_amt, twist_t, radius, wavelength, e_1):
    #y(t) &= y_0(t) + sin(twist_amt) *(k_z z_0(t) - k_x y_0(t))
    #       + (1 - cos(twist_amt)) * (k_y * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - y_0(t))
    #       + (y_0(twist_t) - y_0'(twist_t))
    y_0_at_t = get_y(t, radius, wavelength)
    z_0_at_t = get_z(t)
    x_0_at_t = get_x(t, radius, wavelength)
    k = e_1[twist_t]
    
    # Get reference point (unchanged)
    x_0_ref = x_0_at_t[twist_t]
    y_0_ref = y_0_at_t[twist_t]
    z_0_ref = z_0_at_t[twist_t]
    
    # Calculate the rotated reference point for continuity correction
    s1_ref = y_0_ref
    s2_ref = np.sin(twist_amt) * ((k[2] * z_0_ref) - (k[0] * y_0_ref))
    s3_ref = (1 - np.cos(twist_amt)) * ((k[1] * ((k[0] * x_0_ref) + (k[1] * y_0_ref) + (k[2] * z_0_ref))) - y_0_ref)
    y_0_rotated_ref = s1_ref + s2_ref + s3_ref
    
    # Initialize result with original coordinates
    result = y_0_at_t.copy()
    
    # Apply twist only to points after twist_t
    for i in range(twist_t + 1, len(t)):
        # Calculate twisted coordinates
        s1 = y_0_at_t[i]
        s2 = np.sin(twist_amt) * ((k[2] * z_0_at_t[i]) - (k[0] * y_0_at_t[i]))
        s3 = (1 - np.cos(twist_amt)) * ((k[1] * ((k[0] * x_0_at_t[i]) + (k[1] * y_0_at_t[i]) + (k[2] * z_0_at_t[i]))) - y_0_at_t[i])
        s4 = (y_0_ref - y_0_rotated_ref)  # Continuity correction
        
        result[i] = s1 + s2 + s3 + s4
    
    return result

def y_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec):
    #dy/dt =
    #y'_0(t)
    #+ sin(twist_amt) * (k_z * z_0_prime(t) - k_x * y_0_prime(t))
    #+ (1 - cos(twist_amt)) * (k_y * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - y_0(t))
    x_0_prime_at_t = get_x_prime(t, radius, wavelength)
    y_0_prime_at_t = get_y_prime(t, radius, wavelength)
    z_0_prime_at_t = get_z_prime(t)
    k = tangent_vec
    
    s1 = y_0_prime_at_t
    s2 = np.sin(twist_amt) * ((k[2] * z_0_prime_at_t) - (k[0] * y_0_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[1] * ((k[0] * x_0_prime_at_t) + (k[1] * y_0_prime_at_t) + (k[2] * z_0_prime_at_t))) - y_0_prime_at_t)
    return s1 + s2 + s3


def y_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec):
    #d^2y/dt^2 =
    #y''_0(t)
    #+ sin(twist_amt) * (k_z * z_0''_prime(t) - k_x * y_0''_prime(t))
    #+ (1 - cos(twist_amt)) * (k_y * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - y_0(t))
    x_0_double_prime_at_t = get_x_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    y_0_double_prime_at_t = get_y_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    z_0_double_prime_at_t = get_z_double_prime(t)
    k = tangent_vec
    
    s1 = y_0_double_prime_at_t
    s2 = np.sin(twist_amt) * ((k[2] * z_0_double_prime_at_t) - (k[0] * y_0_double_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[1] * ((k[0] * x_0_double_prime_at_t) + (k[1] * y_0_double_prime_at_t) + (k[2] * z_0_double_prime_at_t))) - y_0_double_prime_at_t)
    return s1 + s2 + s3

def get_z(t):
    return t 

def get_z_prime(t):
    return get_parameter_derivative(t, t)

def get_z_double_prime(t):
    return get_parameter_derivative(t, get_z_prime(t))

def get_z_twist(t, twist_amt, twist_t, radius, wavelength, e_1):
    #z(t) &= z_0(t) 
    #&+ sin(twist_amt) * (k_x * y_0(t) - k_y * x_0(t))
    #&+ (1 - cos(twist_amt)) * (k_z * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - z_0(t))
    #&+ (z_0(twist_t) - z_0'(twist_t))
    z_0_at_t = get_z(t)
    x_0_at_t = get_x(t, radius, wavelength)
    y_0_at_t = get_y(t, radius, wavelength)
    k = e_1[twist_t]
    
    # Get reference point (unchanged)
    x_0_ref = x_0_at_t[twist_t]
    y_0_ref = y_0_at_t[twist_t]
    z_0_ref = z_0_at_t[twist_t]
    
    # Calculate the rotated reference point for continuity correction
    s1_ref = z_0_ref
    s2_ref = np.sin(twist_amt) * ((k[0] * y_0_ref) - (k[1] * x_0_ref))
    s3_ref = (1 - np.cos(twist_amt)) * ((k[2] * ((k[0] * x_0_ref) + (k[1] * y_0_ref) + (k[2] * z_0_ref))) - z_0_ref)
    z_0_rotated_ref = s1_ref + s2_ref + s3_ref
    
    # Initialize result with original coordinates
    result = z_0_at_t.copy()
    
    # Apply twist only to points after twist_t
    for i in range(twist_t + 1, len(t)):
        # Calculate twisted coordinates
        s1 = z_0_at_t[i]
        s2 = np.sin(twist_amt) * ((k[0] * y_0_at_t[i]) - (k[1] * x_0_at_t[i]))
        s3 = (1 - np.cos(twist_amt)) * ((k[2] * ((k[0] * x_0_at_t[i]) + (k[1] * y_0_at_t[i]) + (k[2] * z_0_at_t[i]))) - z_0_at_t[i])
        s4 = (z_0_ref - z_0_rotated_ref)  # Continuity correction
        
        result[i] = s1 + s2 + s3 + s4
    
    return result

def z_prime_twist(t, twist_amt, twist_t, radius, wavelength, tangent_vec):
    #dz/dt =
    #z'_0(t)
    #+ sin(twist_amt) * (k_x * y_0_prime(t) - k_y * x_0_prime(t))
    #+ (1 - cos(twist_amt)) * (k_z * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - z_0(t))
    x_0_prime_at_t = get_x_prime(t, radius, wavelength)
    y_0_prime_at_t = get_y_prime(t, radius, wavelength)
    z_0_prime_at_t = get_z_prime(t)
    k = tangent_vec
    
    s1 = z_0_prime_at_t
    s2 = np.sin(twist_amt) * ((k[0] * y_0_prime_at_t) - (k[1] * x_0_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[2] * ((k[0] * x_0_prime_at_t) + (k[1] * y_0_prime_at_t) + (k[2] * z_0_prime_at_t))) - z_0_prime_at_t)
    return s1 + s2 + s3

def z_double_prime_twist(t, twist_amt, twist_t, radius, wavelength, theta_at_t, theta_prime_at_t, theta_double_prime_at_t, tangent_vec):
    #d^2z/dt^2 =
    #z''_0(t)
    #+ sin(twist_amt) * (k_x * y_0''_prime(t) - k_y * x_0''_prime(t))
    #+ (1 - cos(twist_amt)) * (k_z * (k_x * x_0(t) + k_y * y_0(t) + k_z * z_0(t)) - z_0(t))
    x_0_double_prime_at_t = get_x_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    y_0_double_prime_at_t = get_y_double_prime(t, radius, theta_at_t, theta_prime_at_t, theta_double_prime_at_t)
    z_0_double_prime_at_t = get_z_double_prime(t)
    k = tangent_vec
    
    s1 = z_0_double_prime_at_t
    s2 = np.sin(twist_amt) * ((k[0] * y_0_double_prime_at_t) - (k[1] * x_0_double_prime_at_t))
    s3 = (1 - np.cos(twist_amt)) * ((k[2] * ((k[0] * x_0_double_prime_at_t) + (k[1] * y_0_double_prime_at_t) + (k[2] * z_0_double_prime_at_t))) - z_0_double_prime_at_t)
    return s1 + s2 + s3

def constant_function(t, value):
    return np.ones_like(t) * value

def get_parameter_derivative(z_0_hat, parameter):
    parameter_value_diffs = (parameter[1:] - parameter[:-1]) / (z_0_hat[1:] - z_0_hat[:-1])
    parameter_value_avgs = (parameter_value_diffs[1:] + parameter_value_diffs[:-1]) / 2
    parameter_value_diffs = np.concatenate([parameter_value_diffs[0].reshape(-1, 1), parameter_value_avgs.reshape(-1,1), parameter_value_diffs[-1].reshape(-1, 1)])
    return parameter_value_diffs.reshape(-1,)

def constant_function_derivative(t):
    return np.zeros_like(t)

        
    