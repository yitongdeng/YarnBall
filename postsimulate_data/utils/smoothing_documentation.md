# Advanced Smoothing for Curve Discontinuities

This document describes the advanced smoothing functionality designed to detect and smooth significant jumps in curve data with minimal disturbance to distant values.

## Overview

The smoothing system consists of several key functions that work together to:

1. **Detect jumps** in a sequence of values using statistical thresholds
2. **Apply adaptive Gaussian smoothing** only around detected jumps
3. **Minimize disturbance** to values far from the jumps
4. **Iteratively refine** the smoothing until convergence

## Key Functions

### `detect_jumps(values, threshold_factor=2.0, min_jump_size=None)`

Detects significant jumps in a sequence of values.

**Parameters:**
- `values`: 1D numpy array of values to analyze
- `threshold_factor`: Factor above mean + std to consider a jump significant (default: 2.0)
- `min_jump_size`: Minimum absolute jump size to consider (if None, uses threshold)

**Returns:**
- List of jump indices where significant discontinuities occur

**Example:**
```python
import numpy as np
from false_double import detect_jumps

# Create test signal with jumps
signal = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
signal[50] += 2.0  # Add a jump

# Detect jumps
jumps = detect_jumps(signal, threshold_factor=1.5)
print(f"Found {len(jumps)} jumps at indices: {jumps}")
```

### `adaptive_gaussian_smoothing(values, jump_indices, base_sigma=1.0, max_sigma=5.0)`

Applies adaptive Gaussian smoothing around detected jumps.

**Parameters:**
- `values`: 1D numpy array of values to smooth
- `jump_indices`: Indices where jumps were detected
- `base_sigma`: Base sigma for Gaussian kernel (default: 1.0)
- `max_sigma`: Maximum sigma for Gaussian kernel (default: 5.0)

**Returns:**
- Smoothed values array

**Key Features:**
- Uses adaptive sigma based on jump magnitude
- Applies smoothing only to regions around jumps
- Preserves values far from jumps

### `smooth_jumps_advanced(values, threshold_factor=2.0, base_sigma=1.0, max_sigma=5.0, max_iterations=5, convergence_threshold=1e-6)`

Advanced smoothing that detects and smooths jumps iteratively.

**Parameters:**
- `values`: 1D numpy array of values to smooth
- `threshold_factor`: Factor above mean + std to consider a jump significant
- `base_sigma`: Base sigma for Gaussian kernel
- `max_sigma`: Maximum sigma for Gaussian kernel
- `max_iterations`: Maximum number of smoothing iterations
- `convergence_threshold`: Threshold for convergence check

**Returns:**
- Smoothed values array

**Example:**
```python
import numpy as np
from false_double import smooth_jumps_advanced

# Create test signal
signal = np.sin(np.linspace(0, 10, 200)) + 0.1 * np.random.randn(200)
signal[50] += 2.0  # Add jumps
signal[100] += 1.5
signal[150] += 3.0

# Apply advanced smoothing
smoothed = smooth_jumps_advanced(signal, 
                                threshold_factor=1.5,
                                base_sigma=1.0, 
                                max_sigma=4.0)

print(f"Original max jump: {np.max(np.abs(np.diff(signal))):.4f}")
print(f"Smoothed max jump: {np.max(np.abs(np.diff(smoothed))):.4f}")
```

### `fix_discontinuities(x_vals, y_vals, z_vals, degree, threshold_factor=2.0, base_sigma=1.0, max_sigma=5.0)`

Fixes discontinuities in 3D curve data by smoothing significant jumps.

**Parameters:**
- `x_vals, y_vals, z_vals`: 1D numpy arrays of coordinate values
- `degree`: Degree of the spline (for reporting purposes)
- `threshold_factor`: Factor above mean + std to consider a jump significant
- `base_sigma`: Base sigma for Gaussian kernel
- `max_sigma`: Maximum sigma for Gaussian kernel

**Returns:**
- Tuple of smoothed (x_vals, y_vals, z_vals)

**Example:**
```python
from false_double import fix_discontinuities

# Smooth first derivative components
x_prime_smooth, y_prime_smooth, z_prime_smooth = fix_discontinuities(
    x_prime_fine, y_prime_fine, z_prime_fine, 
    degree=1, 
    threshold_factor=2.0, 
    base_sigma=1.0, 
    max_sigma=4.0
)
```

## Parameter Tuning Guide

### Threshold Factor (`threshold_factor`)

Controls how sensitive the jump detection is:

- **Lower values (1.0-1.5)**: More aggressive detection, smooths more jumps
- **Higher values (2.0-3.0)**: Conservative detection, only smooths major jumps
- **Default (2.0)**: Balanced approach

### Sigma Parameters (`base_sigma`, `max_sigma`)

Control the smoothing intensity:

- **`base_sigma`**: Minimum smoothing width (default: 1.0)
- **`max_sigma`**: Maximum smoothing width (default: 5.0)
- **Lower values**: Minimal smoothing, preserves more detail
- **Higher values**: More aggressive smoothing, removes more noise

### Iteration Control

- **`max_iterations`**: Maximum smoothing iterations (default: 5)
- **`convergence_threshold`**: Stop when max change < threshold (default: 1e-6)

## Usage Examples

### Example 1: Basic Signal Smoothing

```python
import numpy as np
import matplotlib.pyplot as plt
from false_double import smooth_jumps_advanced, visualize_smoothing_comparison

# Create test signal with jumps
t = np.linspace(0, 10, 200)
signal = np.sin(t) + 0.1 * np.random.randn(len(t))

# Add artificial jumps
signal[50] += 2.0
signal[100] += 1.5
signal[150] += 3.0

# Apply smoothing
smoothed = smooth_jumps_advanced(signal, threshold_factor=1.5)

# Visualize results
visualize_smoothing_comparison(signal, smoothed, "Test Signal Smoothing")
```

### Example 2: Derivative Smoothing

```python
from false_double import fix_discontinuities

# Smooth spline derivatives
x_prime_smooth, y_prime_smooth, z_prime_smooth = fix_discontinuities(
    x_prime_fine, y_prime_fine, z_prime_fine, 
    degree=1, 
    threshold_factor=2.0
)

# Visualize results
visualize_smoothing_comparison(x_prime_fine, x_prime_smooth, 
                             "First Derivative Smoothing", "X' Component")
```

### Example 3: Conservative vs Aggressive Smoothing

```python
# Conservative smoothing (preserves more detail)
conservative = smooth_jumps_advanced(signal, 
                                   threshold_factor=2.5, 
                                   base_sigma=0.5, 
                                   max_sigma=2.0)

# Aggressive smoothing (removes more noise)
aggressive = smooth_jumps_advanced(signal, 
                                 threshold_factor=1.0, 
                                 base_sigma=2.0, 
                                 max_sigma=8.0)
```

## Best Practices

1. **Start Conservative**: Begin with default parameters and adjust based on results
2. **Monitor Convergence**: Watch the iteration output to ensure proper convergence
3. **Visualize Results**: Always plot original vs smoothed to verify quality
4. **Test Parameters**: Try different threshold factors and sigma values
5. **Preserve Structure**: Ensure smoothing doesn't remove important features

## Performance Considerations

- **Computational Cost**: O(n × iterations × jump_count) per iteration
- **Memory Usage**: O(n) for storing smoothed arrays
- **Convergence**: Typically 3-5 iterations for most signals
- **Large Datasets**: Consider downsampling for very large arrays

## Troubleshooting

### Too Much Smoothing
- Increase `threshold_factor` (e.g., 2.5-3.0)
- Decrease `max_sigma` (e.g., 2.0-3.0)
- Reduce `base_sigma` (e.g., 0.5-1.0)

### Too Little Smoothing
- Decrease `threshold_factor` (e.g., 1.0-1.5)
- Increase `max_sigma` (e.g., 6.0-8.0)
- Increase `base_sigma` (e.g., 1.5-2.0)

### Slow Convergence
- Increase `convergence_threshold` (e.g., 1e-5)
- Reduce `max_iterations` (e.g., 3-4)
- Check for very large jumps that may need manual handling

## Integration with Existing Code

The smoothing functions are designed to work seamlessly with your existing spline evaluation code:

```python
# After evaluating spline derivatives
x_fine, y_fine, z_fine, x_prime_fine, y_prime_fine, z_prime_fine, ... = evaluate_quintic_spline(...)

# Apply smoothing to derivatives
x_prime_smooth, y_prime_smooth, z_prime_smooth = fix_discontinuities(
    x_prime_fine, y_prime_fine, z_prime_fine, degree=1
)

# Use smoothed derivatives for further calculations
```

This approach ensures that your curve analysis benefits from smooth derivatives while preserving the essential characteristics of your spline curves. 