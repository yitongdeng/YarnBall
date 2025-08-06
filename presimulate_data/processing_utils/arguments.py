import argparse
import numpy as np

def parseArgs(desc):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--visualize_all', default=False, type=bool, help='Show popup visualizations at every step')
    parser.add_argument('--save_visualizations', default=False, type=bool, help='Save visualizations to file')
    parser.add_argument('--sigma', default=0.05, type=float, help='Sigma for Gaussian smoothing')
    parser.add_argument('--save_path', default='./viz_results/', type=str, help='Path to save the visualization')
    parser.add_argument('--n_strands_to_load', default=-1, type=int, help='Number of strands to load from file. -1 loads all strands')
    parser.add_argument('--verbose', default=False, type=bool, help='Print verbose output')
    parser.add_argument('--save_np', default=False, type=bool, help='Save results as numpy arrays')
    parser.add_argument('--load_strands', default=False, type=bool, help='Load strands from file instead of generating them')
    
    # Starting strand parameters
    parser.add_argument('--load_path', default='./strands/canonical_strands.obj', type=str, help='Path to load the strands from')
    parser.add_argument('--primary_points', default=100, type=int, help='Number of points for the primary helix')
    parser.add_argument('--radius', default=10.0, type=float, help='Radius of the helix')
    parser.add_argument('--wavelength', default=1.0, type=float, help='Wavelength of the helix')
    parser.add_argument('--twist_amt', default=np.pi, type=float, help='Amount of twist in the helix')
    parser.add_argument('--twist_location', default=0.5, type=float, help='Time of twist in the helix')
    parser.add_argument('--noise_level', default=0.01, type=float, help='Noise level for the helix')
    parser.add_argument('--flatten_z', default=True, type=bool, help='Flatten the z-axis')
    
    # Smoothing parameters
    parser.add_argument('--smoothing_dct_cutoff', default=0.5, type=float, help='Cutoff for DCT smoothing')
    parser.add_argument('--smoothing_sigma', default=0.05, type=float, help='Sigma for Gaussian smoothing')
    parser.add_argument('--smoothing_iterations', default=5, type=int, help='Number of smoothing iterations')
    parser.add_argument('--smoothing_convergence_threshold', default=1e-6, type=float, help='Convergence threshold for smoothing')
    parser.add_argument('--smoothing_base_sigma', default=1.0, type=float, help='Base sigma for smoothing')
    parser.add_argument('--smoothing_max_sigma', default=5.0, type=float, help='Max sigma for smoothing')
    parser.add_argument('--smoothing_threshold_factor', default=2.0, type=float, help='Threshold factor for smoothing')
    parser.add_argument('--smoothing_min_jump_size', default=None, type=float, help='Min jump size for smoothing')
    parser.add_argument('--n_fine_sampling', default=512, type=int, help='Number of fine sampling points for the spline')
    
    return parser.parse_args()