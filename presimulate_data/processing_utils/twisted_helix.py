import numpy as np
from utils.function_derivatives import (
    get_x, get_y, get_z,
    get_parameter_derivative,
    constant_function,
    constant_function_derivative
)

def apply_twist_to_helix(t_values, radius, wavelength, twist_specifications):
    """
    Apply twist transformations to a continuous helix at specified parameter values.
    
    Parameters:
    - t_values: array of parameter values along the helix
    - radius: helix radius (can be constant or array)
    - wavelength: helix wavelength (can be constant or array)
    - twist_specifications: list of tuples [(twist_angle1, t_position1), (twist_angle2, t_position2), ...]
    
    Returns:
    - x_twisted, y_twisted, z_twisted: arrays of twisted helix coordinates
    """
    
    # Start with original helix coordinates
    x_original = get_x(t_values, radius, wavelength)
    y_original = get_y(t_values, radius, wavelength)
    z_original = get_z(t_values)
    
    # Initialize twisted coordinates
    x_twisted = x_original.copy()
    y_twisted = y_original.copy()
    z_twisted = z_original.copy()
    
    # Sort twist specifications by t_position
    twist_specs_sorted = sorted(twist_specifications, key=lambda x: x[1])
    
    # Apply each twist transformation
    for twist_angle, t_twist in twist_specs_sorted:
        # Find the closest parameter value index
        twist_idx = np.argmin(np.abs(t_values - t_twist))
        
        # Get tangent vector at twist point
        tangent = get_tangent_at_t(t_twist, radius, wavelength)
        tangent = tangent / np.linalg.norm(tangent)  # normalize
        
        # Apply twist rotation to all points from this index onward
        for i in range(twist_idx, len(t_values)):
            # Translate point to origin at twist location
            point = np.array([
                x_twisted[i] - x_twisted[twist_idx],
                y_twisted[i] - y_twisted[twist_idx],
                z_twisted[i] - z_twisted[twist_idx]
            ])
            
            # Apply Rodrigues' rotation formula
            rotated_point = rodrigues_rotation(point, tangent, twist_angle)
            
            # Translate back
            x_twisted[i] = rotated_point[0] + x_twisted[twist_idx]
            y_twisted[i] = rotated_point[1] + y_twisted[twist_idx]
            z_twisted[i] = rotated_point[2] + z_twisted[twist_idx]
    
    return x_twisted, y_twisted, z_twisted

def get_tangent_at_t(t, radius, wavelength):
    """
    Calculate the tangent vector at parameter t for the helix.
    """
    # Convert scalar t to array for function compatibility
    t_array = np.array([t])
    
    # Get derivatives (tangent components)
    radius_at_t = constant_function(t_array, radius)
    wavelength_at_t = constant_function(t_array, wavelength)
    radius_prime = constant_function_derivative(t_array, radius)
    wavelength_prime = constant_function_derivative(t_array, wavelength)
    
    # Calculate theta and its derivative
    theta = (2 * np.pi * t) / wavelength_at_t[0]
    theta_prime = (2 * np.pi * (wavelength_at_t[0] - (t * wavelength_prime[0]))) / wavelength_at_t[0]**2
    
    # Tangent components
    x_prime = radius_prime[0] * np.cos(theta) - radius_at_t[0] * np.sin(theta) * theta_prime
    y_prime = radius_prime[0] * np.sin(theta) + radius_at_t[0] * np.cos(theta) * theta_prime
    z_prime = 1.0  # dz/dt = 1 since z = t
    
    return np.array([x_prime, y_prime, z_prime])

def rodrigues_rotation(vector, axis, angle):
    """
    Rotate a vector around an axis using Rodrigues' rotation formula.
    
    Parameters:
    - vector: 3D vector to rotate
    - axis: normalized rotation axis
    - angle: rotation angle in radians
    
    Returns:
    - rotated vector
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' formula: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    cross_product = np.cross(axis, vector)
    dot_product = np.dot(axis, vector)
    
    rotated = (vector * cos_angle + 
               cross_product * sin_angle + 
               axis * dot_product * (1 - cos_angle))
    
    return rotated

def create_piecewise_twisted_helix(t_values, radius, wavelength, twist_specifications):
    """
    Create a piecewise twisted helix with smooth transitions.
    
    Parameters:
    - t_values: array of parameter values
    - radius: helix radius
    - wavelength: helix wavelength  
    - twist_specifications: list of (twist_angle, t_position) tuples
    
    Returns:
    - Dictionary containing original and twisted coordinates, plus metadata
    """
    
    # Get original helix
    x_orig, y_orig, z_orig = get_x(t_values, radius, wavelength), get_y(t_values, radius, wavelength), get_z(t_values)
    
    # Apply twists
    x_twisted, y_twisted, z_twisted = apply_twist_to_helix(t_values, radius, wavelength, twist_specifications)
    
    # Calculate tangent vectors along the twisted helix
    tangents = []
    for i in range(len(t_values)):
        t = t_values[i]
        tangent = get_tangent_at_t(t, radius, wavelength)
        tangents.append(tangent)
    tangents = np.array(tangents)
    
    # Create piecewise segments
    segments = []
    twist_positions = sorted([spec[1] for spec in twist_specifications])
    
    segment_boundaries = [t_values[0]] + twist_positions + [t_values[-1]]
    
    for i in range(len(segment_boundaries) - 1):
        start_t = segment_boundaries[i]
        end_t = segment_boundaries[i + 1]
        
        # Find indices for this segment
        start_idx = np.argmin(np.abs(t_values - start_t))
        end_idx = np.argmin(np.abs(t_values - end_t))
        
        segment = {
            't_range': (start_t, end_t),
            'indices': (start_idx, end_idx),
            'x': x_twisted[start_idx:end_idx+1],
            'y': y_twisted[start_idx:end_idx+1], 
            'z': z_twisted[start_idx:end_idx+1],
            'tangents': tangents[start_idx:end_idx+1]
        }
        segments.append(segment)
    
    return {
        'original': {'x': x_orig, 'y': y_orig, 'z': z_orig},
        'twisted': {'x': x_twisted, 'y': y_twisted, 'z': z_twisted},
        'tangents': tangents,
        'segments': segments,
        'twist_specs': twist_specifications,
        't_values': t_values
    }

def visualize_twisted_helix(helix_data, title="Twisted Helix"):
    """
    Visualize the original and twisted helix for comparison.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # Original helix
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(helix_data['original']['x'], 
             helix_data['original']['y'], 
             helix_data['original']['z'], 'b-', linewidth=2, label='Original')
    ax1.set_title('Original Helix')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Twisted helix
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(helix_data['twisted']['x'], 
             helix_data['twisted']['y'], 
             helix_data['twisted']['z'], 'r-', linewidth=2, label='Twisted')
    
    # Mark twist points
    for twist_angle, t_pos in helix_data['twist_specs']:
        idx = np.argmin(np.abs(helix_data['t_values'] - t_pos))
        ax2.scatter(helix_data['twisted']['x'][idx], 
                   helix_data['twisted']['y'][idx], 
                   helix_data['twisted']['z'][idx], 
                   c='black', s=100, alpha=0.8)
    
    ax2.set_title('Twisted Helix')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y') 
    ax2.set_zlabel('Z')
    
    # Comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(helix_data['original']['x'], 
             helix_data['original']['y'], 
             helix_data['original']['z'], 'b-', linewidth=2, alpha=0.6, label='Original')
    ax3.plot(helix_data['twisted']['x'], 
             helix_data['twisted']['y'], 
             helix_data['twisted']['z'], 'r-', linewidth=2, alpha=0.8, label='Twisted')
    ax3.set_title('Comparison')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def get_twisted_frame_vectors(helix_data, frame_indices=None):
    """
    Calculate e1, e2, e3 frame vectors for the twisted helix.
    
    Parameters:
    - helix_data: result from create_piecewise_twisted_helix
    - frame_indices: specific indices to calculate frames for (None for all)
    
    Returns:
    - e1, e2, e3 arrays of frame vectors
    """
    if frame_indices is None:
        frame_indices = range(len(helix_data['t_values']))
    
    t_values = helix_data['t_values']
    radius = 0.5  # You may want to pass this as parameter
    wavelength = 1.0  # You may want to pass this as parameter
    
    e1_vectors = []
    e2_vectors = []
    e3_vectors = []
    
    for i in frame_indices:
        t = t_values[i]
        
        # For twisted sections, we need to account for the rotation
        # Find which segment this point belongs to
        segment_idx = 0
        for j, segment in enumerate(helix_data['segments']):
            if segment['t_range'][0] <= t <= segment['t_range'][1]:
                segment_idx = j
                break
        
        # Get original frame vectors
        from utils.helix_functions import get_e1_at_t, get_e2_at_t
        e1_orig = get_e1_at_t(np.array([t]), radius, wavelength)[0]
        e2_orig = get_e2_at_t(np.array([t]), np.array([e1_orig]), radius, wavelength)[0]
        e3_orig = np.cross(e1_orig, e2_orig)
        
        # Apply accumulated twist rotations up to this point
        accumulated_twist = 0
        tangent = get_tangent_at_t(t, radius, wavelength)
        tangent = tangent / np.linalg.norm(tangent)
        
        for twist_angle, t_twist in helix_data['twist_specs']:
            if t_twist <= t:
                accumulated_twist += twist_angle
        
        if accumulated_twist != 0:
            # Rotate frame vectors by accumulated twist
            e1_twisted = rodrigues_rotation(e1_orig, tangent, accumulated_twist)
            e2_twisted = rodrigues_rotation(e2_orig, tangent, accumulated_twist) 
            e3_twisted = rodrigues_rotation(e3_orig, tangent, accumulated_twist)
        else:
            e1_twisted = e1_orig
            e2_twisted = e2_orig
            e3_twisted = e3_orig
        
        e1_vectors.append(e1_twisted)
        e2_vectors.append(e2_twisted)
        e3_vectors.append(e3_twisted)
    
    return np.array(e1_vectors), np.array(e2_vectors), np.array(e3_vectors)

# Example usage
if __name__ == "__main__":
    # Define helix parameters
    height = 4.0
    n_points = 200
    radius = 0.5
    wavelength = 1.0
    
    # Create parameter values
    t_values = np.linspace(0, height, n_points)
    
    # Define twist specifications: (angle_in_radians, t_position)
    twist_specs = [
        (np.pi/4, 1.0),    # 45-degree twist at t=1.0
        (-np.pi/6, 2.5),   # -30-degree twist at t=2.5
        (np.pi/3, 3.5)     # 60-degree twist at t=3.5
    ]
    
    # Create twisted helix
    helix_data = create_piecewise_twisted_helix(t_values, radius, wavelength, twist_specs)

    # Get frame vectors for twisted helix
    e1, e2, e3 = get_twisted_frame_vectors(helix_data)
    
    # Visualize results
    visualize_twisted_helix(helix_data)
    
    print(f"Created twisted helix with {len(helix_data['segments'])} segments")
    print(f"Twist points: {[spec[1] for spec in twist_specs]}")
    print(f"Frame vectors calculated for {len(e1)} points")