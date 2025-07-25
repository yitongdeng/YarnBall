import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import dct, idct

def even_degree_natural_bc(k):
    # k: spline degree (must be even)
    n_bc = k - 1
    left = n_bc // 2 + n_bc % 2  # more on left if odd
    right = n_bc // 2
    left_bc = [(i, 0.0) for i in range(2, 2 + left)] if left > 0 else None
    right_bc = [(i, 0.0) for i in range(2, 2 + right)] if right > 0 else None
    return (left_bc, right_bc)

def fft_smooth(x_vals, y_vals, z_vals, dct_cutoff):
    dct_x = dct(x_vals, type=2, norm='ortho')
    dct_y = dct(y_vals, type=2, norm='ortho')
    dct_z = dct(z_vals, type=2, norm='ortho')
    filtered_dct_x = np.copy(dct_x)
    filtered_dct_y = np.copy(dct_y)
    filtered_dct_z = np.copy(dct_z)
    filtered_dct_x[int(len(x_vals) * dct_cutoff):] = 0
    filtered_dct_y[int(len(y_vals) * dct_cutoff):] = 0
    filtered_dct_z[int(len(z_vals) * dct_cutoff):] = 0
    smoothed_data_x = idct(filtered_dct_x, type=2, norm='ortho')
    smoothed_data_y = idct(filtered_dct_y, type=2, norm='ortho')
    smoothed_data_z = idct(filtered_dct_z, type=2, norm='ortho')
    
    return smoothed_data_x, smoothed_data_y, smoothed_data_z

def fix_discontinuities(x_vals, y_vals, z_vals, arc_len, degree, threshold_factor=3.0, base_sigma=1.0, max_sigma=5.0, general_smooth=False, verbose=False):
    """
    Fix discontinuities in 3D curve data by smoothing significant jumps.
    Args:
        x_vals, y_vals, z_vals: 1D numpy arrays of coordinate values
        degree: Degree of the spline (for reporting purposes)
        threshold_factor: Factor above mean + std to consider a jump significant
        base_sigma: Base sigma for Gaussian kernel
        max_sigma: Maximum sigma for Gaussian kernel
    Returns:
        Tuple of (smoothed arrays, splines):
        (x_smoothed, y_smoothed, z_smoothed, x_cs, y_cs, z_cs)
    """
    if verbose:
        print(f"    Smoothing {degree}th derivative components...")
    if general_smooth:
        x_vals, y_vals, z_vals = smooth_vector_components(
            np.column_stack([x_vals, y_vals, z_vals]),
            arc_len, sigma=len(arc_len) * 0.05, component_name=f"{degree}th derivative",
            visualize_all=False, verbose=verbose
        ).T
    x_vals, y_vals, z_vals = fix_zeros(x_vals, y_vals, z_vals, verbose=verbose)
    x_smoothed = smooth_jumps_advanced(x_vals, threshold_factor, base_sigma, max_sigma, verbose=verbose)
    y_smoothed = smooth_jumps_advanced(y_vals, threshold_factor, base_sigma, max_sigma, verbose=verbose)
    z_smoothed = smooth_jumps_advanced(z_vals, threshold_factor, base_sigma, max_sigma, verbose=verbose)

    bc_type = 'not-a-knot' if degree % 2 == 1 else even_degree_natural_bc(degree)
    from scipy.interpolate import make_interp_spline
    x_cs = make_interp_spline(arc_len, x_smoothed, k=degree, bc_type=bc_type) 
    y_cs = make_interp_spline(arc_len, y_smoothed, k=degree, bc_type=bc_type)
    z_cs = make_interp_spline(arc_len, z_smoothed, k=degree, bc_type=bc_type)
        
    return x_smoothed, y_smoothed, z_smoothed, x_cs, y_cs, z_cs

def visualize_smoothing_comparison(original, smoothed, title="Smoothing Comparison", component_name="Component"):
    """
    Visualize the original vs smoothed values to show the smoothing effect.
    
    Args:
        original: Original values array
        smoothed: Smoothed values array
        title: Plot title
        component_name: Name of the component being smoothed
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original vs smoothed
    plt.subplot(2, 1, 1)
    plt.plot(original, 'b-', alpha=0.7, label='Original', linewidth=1)
    plt.plot(smoothed, 'r-', alpha=0.8, label='Smoothed', linewidth=1.5)
    plt.title(f"{title} - {component_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the difference
    plt.subplot(2, 1, 2)
    difference = smoothed - original
    plt.plot(difference, 'g-', alpha=0.8, linewidth=1)
    plt.title(f"Difference (Smoothed - Original) - {component_name}")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def smooth_vector_components(e_1, t_fine, sigma=2.0, component_name="e_1", visualize_all=False, verbose=False):
    """
    Smooth the e_1 vector components (x, y, z) separately using Gaussian smoothing.
    
    Args:
        e_1: 2D numpy array of shape (n_points, 3) containing e_1 vectors
        t_fine: Parameter values corresponding to e_1 vectors
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Smoothed e_1 array with unit vectors
    """
    if verbose:
        print(f"    Smoothing {component_name} components with Gaussian kernel (sigma={sigma})...")
    
    # Extract individual components
    e_1_x = e_1[:, 0]
    e_1_y = e_1[:, 1] 
    e_1_z = e_1[:, 2]
    
    # Apply Gaussian smoothing to each component
    from scipy.ndimage import gaussian_filter1d
    e_1_x_smoothed = gaussian_filter1d(e_1_x, sigma=sigma)
    e_1_y_smoothed = gaussian_filter1d(e_1_y, sigma=sigma)
    e_1_z_smoothed = gaussian_filter1d(e_1_z, sigma=sigma)
    
    # Reconstruct the smoothed e_1 vector
    e_1_smoothed = np.column_stack([e_1_x_smoothed, e_1_y_smoothed, e_1_z_smoothed])
    
    # Normalize to maintain unit vector properties
    norms = np.linalg.norm(e_1_smoothed, axis=1)
    e_1_smoothed = e_1_smoothed / norms[:, np.newaxis]
    
    if visualize_all:
        # Optional: Visualize the smoothing effect for each component
        visualize_smoothing_comparison(e_1_x, e_1_x_smoothed, f"{component_name} X Component Smoothing", "X Component")
        visualize_smoothing_comparison(e_1_y, e_1_y_smoothed, f"{component_name} Y Component Smoothing", "Y Component") 
        visualize_smoothing_comparison(e_1_z, e_1_z_smoothed, f"{component_name} Z Component Smoothing", "Z Component")
    
    return e_1_smoothed

def fix_zeros(x_vals, y_vals, z_vals, verbose=False):
    """
    Fix zero crossings in 3D curve data by adjusting points around zero crossings.
    
    Args:
        x_vals, y_vals, z_vals: 1D numpy arrays of coordinate values
        arc_len: Arc length parameterization
        degree: Degree of the spline (for reporting purposes)
    
    Returns:
        Tuple of adjusted (x_vals, y_vals, z_vals)
    """
    x_zero_crossings = np.where(np.diff(np.sign(x_vals)) != 0)[0]
    y_zero_crossings = np.where(np.diff(np.sign(y_vals)) != 0)[0]
    z_zero_crossings = np.where(np.diff(np.sign(z_vals)) != 0)[0]
    x_zero_crossings = np.append(x_zero_crossings, np.where(np.abs(x_vals) < 1e-6)[0])
    y_zero_crossings = np.append(y_zero_crossings, np.where(np.abs(y_vals) < 1e-6)[0])
    z_zero_crossings = np.append(z_zero_crossings, np.where(np.abs(z_vals) < 1e-6)[0])
    
    # Find common indices across all three arrays
    common_crossings = np.intersect1d(np.intersect1d(x_zero_crossings, y_zero_crossings), z_zero_crossings)
    for i in common_crossings:
        if i == 0 or i == len(x_vals) - 1:
            continue  # skip boundaries to avoid IndexError
        x_curvature = np.abs(x_vals[i+1] - 2*x_vals[i] + x_vals[i-1])
        y_curvature = np.abs(y_vals[i+1] - 2*y_vals[i] + y_vals[i-1])
        z_curvature = np.abs(z_vals[i+1] - 2*z_vals[i] + z_vals[i-1])
        
        if x_curvature < y_curvature and x_curvature < z_curvature:
            if verbose:
                print(f"Adjusting x_vals[{i}] from {x_vals[i]} to {x_vals[i] + np.std(y_vals) * .1}")
            x_vals[i] += np.std(y_vals) * 3
        elif y_curvature < x_curvature and y_curvature < z_curvature:
            if verbose:
                print(f"Adjusting y_vals[{i}] from {y_vals[i]} to {y_vals[i] + np.std(z_vals) * .1}")
            y_vals[i] += np.std(z_vals) * 3
        elif z_curvature < x_curvature and z_curvature < y_curvature:
            if verbose:
                print(f"Adjusting z_vals[{i}] from {z_vals[i]} to {z_vals[i] + np.std(x_vals) * .1}")
            z_vals[i] += np.std(x_vals) * 3

    if len(common_crossings) > 0 and verbose:
        print(f"Common zero crossings found at indices: {common_crossings}")
    return x_vals, y_vals, z_vals
    
def detect_jumps(values, threshold_factor=2.0, min_jump_size=None):
    """
    Detect significant jumps in a sequence of values.
    
    Args:
        values: 1D numpy array of values
        threshold_factor: Factor above mean + std to consider a jump significant
        min_jump_size: Minimum absolute jump size to consider (if None, uses threshold)
    
    Returns:
        List of jump indices
    """
    diffs = np.diff(values)
    diff_mean = np.mean(diffs)
    diff_std = np.std(diffs)
    threshold = threshold_factor * diff_std + diff_mean
    
    if min_jump_size is not None:
        threshold = max(threshold, min_jump_size)
    
    jump_indices = np.where(np.abs(diffs) > threshold)[0]
    return jump_indices

def adaptive_gaussian_smoothing(values, jump_indices, base_sigma=1.0, max_sigma=5.0):
    """
    Apply adaptive Gaussian smoothing around detected jumps.
    
    Args:
        values: 1D numpy array of values to smooth
        jump_indices: Indices where jumps were detected
        base_sigma: Base sigma for Gaussian kernel
        max_sigma: Maximum sigma for Gaussian kernel
    
    Returns:
        Smoothed values array
    """
    smoothed = values.copy()
    n = len(values)
    
    if len(jump_indices) == 0:
        return smoothed
    
    # Create a mask for regions that need smoothing
    smooth_mask = np.zeros(n, dtype=bool)
    
    # Mark regions around jumps with adaptive width
    for jump_idx in jump_indices:
        # Calculate jump magnitude to determine smoothing width
        if jump_idx < n - 1:
            jump_magnitude = abs(values[jump_idx + 1] - values[jump_idx])
            # Adaptive sigma based on jump magnitude
            sigma = min(max_sigma, base_sigma + jump_magnitude / 2.0)
            width = int(3 * sigma)  # 3-sigma rule
            
            # Mark region for smoothing
            start_idx = max(0, jump_idx - width)
            end_idx = min(n, jump_idx + width + 1)
            smooth_mask[start_idx:end_idx] = True
    
    # Apply Gaussian smoothing only to marked regions
    if np.any(smooth_mask):
        # Create a temporary array for smoothing
        temp_values = smoothed.copy()
        
        # Apply different sigma values based on distance from jumps
        for i in range(n):
            if smooth_mask[i]:
                # Find the closest jump
                min_distance = float('inf')
                closest_sigma = base_sigma
                
                for jump_idx in jump_indices:
                    distance = abs(i - jump_idx)
                    if distance < min_distance:
                        min_distance = distance
                        # Adaptive sigma based on distance and jump magnitude
                        if jump_idx < n - 1:
                            jump_magnitude = abs(values[jump_idx + 1] - values[jump_idx])
                            closest_sigma = min(max_sigma, base_sigma + jump_magnitude / (distance + 1))
                
                # Apply local smoothing around point i
                window_size = int(3 * closest_sigma)
                start_window = max(0, i - window_size)
                end_window = min(n, i + window_size + 1)
                
                # Create local Gaussian kernel
                local_indices = np.arange(start_window, end_window)
                local_weights = np.exp(-0.5 * ((local_indices - i) / closest_sigma) ** 2)
                local_weights = local_weights / np.sum(local_weights)
                
                # Apply weighted average
                temp_values[i] = np.sum(smoothed[start_window:end_window] * local_weights)
        
        smoothed = temp_values
    
    return smoothed

def smooth_jumps_advanced(values, threshold_factor=2.0, base_sigma=1.0, max_sigma=5.0, 
                         max_iterations=5, convergence_threshold=1e-6, verbose=False):
    """
    Advanced smoothing that detects and smooths jumps with minimal disturbance to distant values.
    
    Args:
        values: 1D numpy array of values to smooth
        threshold_factor: Factor above mean + std to consider a jump significant
        base_sigma: Base sigma for Gaussian kernel
        max_sigma: Maximum sigma for Gaussian kernel
        max_iterations: Maximum number of smoothing iterations
        convergence_threshold: Threshold for convergence check
    
    Returns:
        Smoothed values array
    """
    smoothed = values.copy()
    n = len(values)
    
    for iteration in range(max_iterations):
        # Detect jumps in current smoothed values
        jump_indices = detect_jumps(smoothed, threshold_factor)
        
        if len(jump_indices) == 0:
            if verbose:
                print(f"    No jumps detected after {iteration + 1} iterations")
            break
        
        if verbose:
            print(f"    Iteration {iteration + 1}: Found {len(jump_indices)} jumps")
        
        # Store previous values for convergence check
        prev_smoothed = smoothed.copy()
        
        # Apply adaptive smoothing
        smoothed = adaptive_gaussian_smoothing(smoothed, jump_indices, base_sigma, max_sigma)
        
        # Check for convergence
        max_change = np.max(np.abs(smoothed - prev_smoothed))
        if max_change < convergence_threshold:
            print(f"    Converged after {iteration + 1} iterations (max change: {max_change:.6f})")
            break
    
    return smoothed

def visualize_smoothing_comparison(original, smoothed, title="Smoothing Comparison", component_name="Component"):
    """
    Visualize the original vs smoothed values to show the smoothing effect.
    
    Args:
        original: Original values array
        smoothed: Smoothed values array
        title: Plot title
        component_name: Name of the component being smoothed
    """
    plt.figure(figsize=(15, 10))
    
    # Plot original vs smoothed
    plt.subplot(3, 1, 1)
    plt.plot(original, 'b-', alpha=0.7, label='Original', linewidth=1)
    plt.plot(smoothed, 'r-', alpha=0.8, label='Smoothed', linewidth=1.5)
    plt.title(f"{title} - {component_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot the difference
    plt.subplot(3, 1, 2)
    difference = smoothed - original
    plt.plot(difference, 'g-', alpha=0.8, linewidth=1)
    plt.title(f"Difference (Smoothed - Original) - {component_name}")
    plt.grid(True, alpha=0.3)
    
    # Plot the derivatives to show jump reduction
    plt.subplot(3, 1, 3)
    original_deriv = np.diff(original)
    smoothed_deriv = np.diff(smoothed)
    plt.plot(original_deriv, 'b-', alpha=0.7, label='Original Derivative', linewidth=1)
    plt.plot(smoothed_deriv, 'r-', alpha=0.8, label='Smoothed Derivative', linewidth=1.5)
    plt.title(f"Derivatives Comparison - {component_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def test_advanced_smoothing():
    """Test the advanced smoothing functionality with various scenarios."""
    
    print("Testing advanced smoothing functionality...")
    
    # Test 1: Simple signal with artificial jumps
    t_test = np.linspace(0, 10, 200)
    test_signal = np.sin(t_test) + 0.1 * np.random.randn(len(t_test))
    
    # Add artificial jumps of different magnitudes
    jump_indices = [50, 100, 150]
    jump_magnitudes = [2.0, 1.5, 3.0]
    for idx, magnitude in zip(jump_indices, jump_magnitudes):
        test_signal[idx] += magnitude
    
    print("Test 1: Simple signal with jumps")
    print(f"  Original max jump: {np.max(np.abs(np.diff(test_signal))):.4f}")
    
    # Apply advanced smoothing
    smoothed_signal = smooth_jumps_advanced(test_signal, threshold_factor=1.5, 
                                          base_sigma=1.0, max_sigma=4.0)
    
    print(f"  Smoothed max jump: {np.max(np.abs(np.diff(smoothed_signal))):.4f}")
    
    # Visualize
    visualize_smoothing_comparison(test_signal, smoothed_signal, 
                                 "Advanced Smoothing Test 1", "Test Signal")
    
    # Test 2: Signal with multiple small jumps
    print("\nTest 2: Signal with multiple small jumps")
    t_test2 = np.linspace(0, 20, 400)
    test_signal2 = np.sin(t_test2) + 0.2 * np.random.randn(len(t_test2))
    
    # Add many small jumps
    for i in range(10, len(test_signal2), 30):
        test_signal2[i] += 0.5 * np.random.randn()
    
    print(f"  Original max jump: {np.max(np.abs(np.diff(test_signal2))):.4f}")
    
    smoothed_signal2 = smooth_jumps_advanced(test_signal2, threshold_factor=1.2, 
                                           base_sigma=0.8, max_sigma=3.0)
    
    print(f"  Smoothed max jump: {np.max(np.abs(np.diff(smoothed_signal2))):.4f}")
    
    # Visualize
    visualize_smoothing_comparison(test_signal2, smoothed_signal2, 
                                 "Advanced Smoothing Test 2", "Test Signal 2")
    
    # Test 3: Compare different smoothing approaches
    print("\nTest 3: Comparing smoothing approaches")
    
    # Standard Gaussian smoothing (for comparison)
    gaussian_smoothed = gaussian_filter1d(test_signal, sigma=2.0)
    
    # Our advanced smoothing
    advanced_smoothed = smooth_jumps_advanced(test_signal, threshold_factor=1.5, 
                                            base_sigma=1.0, max_sigma=4.0)
    
    # Compare all three
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.plot(test_signal, 'b-', alpha=0.7, label='Original', linewidth=1)
    plt.title("Original Signal")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    plt.plot(np.diff(test_signal), 'b-', alpha=0.7, label='Original Derivative', linewidth=1)
    plt.title("Original Derivative")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    plt.plot(gaussian_smoothed, 'g-', alpha=0.8, label='Gaussian Smoothed', linewidth=1.5)
    plt.title("Standard Gaussian Smoothing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    plt.plot(np.diff(gaussian_smoothed), 'g-', alpha=0.8, label='Gaussian Derivative', linewidth=1.5)
    plt.title("Gaussian Smoothed Derivative")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 5)
    plt.plot(advanced_smoothed, 'r-', alpha=0.8, label='Advanced Smoothed', linewidth=1.5)
    plt.title("Advanced Adaptive Smoothing")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    plt.plot(np.diff(advanced_smoothed), 'r-', alpha=0.8, label='Advanced Derivative', linewidth=1.5)
    plt.title("Advanced Smoothed Derivative")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nSmoothing Statistics:")
    print(f"  Original - Max jump: {np.max(np.abs(np.diff(test_signal))):.4f}")
    print(f"  Gaussian - Max jump: {np.max(np.abs(np.diff(gaussian_smoothed))):.4f}")
    print(f"  Advanced - Max jump: {np.max(np.abs(np.diff(advanced_smoothed))):.4f}")
    
    print("\nAdvanced smoothing test completed!")

if __name__ == "__main__":
    test_advanced_smoothing() 