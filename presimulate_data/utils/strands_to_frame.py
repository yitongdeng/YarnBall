import os
import numpy as np
from utils.arguments import parseArgs
from utils.geometry import load_strands
from utils.smoothing import fft_smooth, fix_discontinuities, smooth_vector_components
from utils.spline_utils import parameterize_arc_length, get_scipy_spline, evaluate_spline_derivatives, get_frenet_frame, evaluate_spline, smooth_curvature_torsion 
from utils.quintic_functions import get_quintic_torsion

import tqdm

def strands_to_frame(x_arr, y_arr, z_arr, args):
    """
    Converts strands to Frenet frame representation.
    """
    e_1_arr = []
    e_2_arr = []
    e_3_arr = []
    curvature_arr = []
    torsion_arr = []
    quintic_torsion_arr = []
    unsmooth_torsion_arr = []
    
    for x, y, z in tqdm.tqdm(zip(x_arr, y_arr, z_arr), total=len(x_arr), desc="Processing strands"):
        quintic_torsion_arr.append(get_quintic_torsion(x, y, z))
        
        x, y, z = fft_smooth(x, y, z, args.smoothing_dct_cutoff)
        arc_len = parameterize_arc_length(x, y, z)
        t_fine = np.linspace(0, arc_len[-1], args.n_fine_sampling)
        
        x_cs, y_cs, z_cs = get_scipy_spline(x, y, z, arc_len, 3)
        
        # Evaluate spline and smooth
        x_fine, y_fine, z_fine = evaluate_spline(x_cs, y_cs, z_cs, t_fine)
        x_fine, y_fine, z_fine, x_cs, y_cs, z_cs = fix_discontinuities(
            x_fine, y_fine, z_fine, t_fine, 3, threshold_factor=2.0, base_sigma=1.0, max_sigma=4.0, verbose=args.verbose)  
        x_fine, y_fine, z_fine = smooth_vector_components(
            np.stack([x_fine, y_fine, z_fine], axis=1), t_fine, sigma=len(t_fine) * .05, component_name="spline", visualize_all=args.visualize_all, verbose=args.verbose).T      
        
        # Evaluate first derivative and smooth
        x_prime_fine, y_prime_fine, z_prime_fine = evaluate_spline_derivatives(x_cs, y_cs, z_cs, t_fine)
        x_prime_fine, y_prime_fine, z_prime_fine, x_prime_cs, y_prime_cs, z_prime_cs = fix_discontinuities(
            x_prime_fine, y_prime_fine, z_prime_fine, t_fine, 3, threshold_factor=2.0, base_sigma=1.0, max_sigma=4.0, verbose=args.verbose)
        x_prime_fine, y_prime_fine, z_prime_fine = smooth_vector_components(
            np.stack([x_prime_fine, y_prime_fine, z_prime_fine], axis=1), t_fine, sigma=len(t_fine) * .05, component_name="first derivative", visualize_all=args.visualize_all, verbose=args.verbose).T
 
        # Evaluate second derivative and smooth
        x_second_fine, y_second_fine, z_second_fine = evaluate_spline_derivatives( x_prime_cs, y_prime_cs, z_prime_cs, t_fine)
        x_second_fine, y_second_fine, z_second_fine, x_second_cs, y_second_cs, z_second_cs = fix_discontinuities(
            x_second_fine, y_second_fine, z_second_fine, t_fine, 3, threshold_factor=2.0, base_sigma=1.0, max_sigma=4.0, verbose=args.verbose)
        x_second_fine, y_second_fine, z_second_fine = smooth_vector_components(np.stack([x_second_fine, y_second_fine, z_second_fine], axis=1), t_fine, sigma=len(t_fine) * .05, component_name="second derivative", visualize_all=args.visualize_all, verbose=args.verbose).T

        # Evaluate third derivative and smooth
        x_third_fine, y_third_fine, z_third_fine = evaluate_spline_derivatives(x_second_cs, y_second_cs, z_second_cs, t_fine)
        x_third_fine, y_third_fine, z_third_fine, x_third_cs, y_third_cs, z_third_cs = fix_discontinuities(
            x_third_fine, y_third_fine, z_third_fine, t_fine, 3, threshold_factor=2.0, base_sigma=1.0, max_sigma=4.0, verbose=args.verbose)
        x_third_fine, y_third_fine, z_third_fine = smooth_vector_components(np.stack([x_third_fine, y_third_fine, z_third_fine], axis=1), t_fine, sigma=len(t_fine) * .05, component_name="third derivative", visualize_all=args.visualize_all, verbose=args.verbose).T
        # Get Frenet frame and curvature/torsion
        e_1, e_2, e_3, curvature, torsion = get_frenet_frame(x_fine, y_fine, z_fine, x_prime_fine, y_prime_fine, z_prime_fine, x_second_fine, y_second_fine, z_second_fine, t_fine, args.visualize_all)
        unsmooth_torsion_arr.append(torsion)
        
        # Apply smoothing to e_1 components separately
        e_1_smoothed = smooth_vector_components(e_1, t_fine, sigma=len(t_fine) * .05, component_name="e_1", visualize_all=args.visualize_all)
        e_2_smoothed = smooth_vector_components(e_2, t_fine, sigma=len(t_fine) * .05, component_name="e_2", visualize_all=args.visualize_all)
        e_3_smoothed = smooth_vector_components(e_3, t_fine, sigma=len(t_fine) * .05, component_name="e_3", visualize_all=args.visualize_all)
        
        v = np.stack([x_prime_fine, y_prime_fine, z_prime_fine], axis=1)
        #smoothed_curvature, smoothed_torsion = smooth_curvature_torsion(e_1_smoothed, e_2_smoothed, e_3_smoothed, t_fine, v)
        e_1_arr.append(e_1)
        e_2_arr.append(e_2)
        e_3_arr.append(e_3)
        curvature_arr.append(curvature)
        torsion_arr.append(torsion)
    return e_1_arr, e_2_arr, e_3_arr, curvature_arr, torsion_arr
        
if __name__ == "__main__":
    args = parseArgs("Helix Generation")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    x_arr, y_arr, z_arr = load_strands(args.load_path, args.n_strands_to_load)
    
    e_1_arr, e_2_arr, e_3_arr, curvature_arr, torsion_arr = strands_to_frame(x_arr, y_arr, z_arr, args)
    
    if args.save_np:
        # Save results
        np.savez(os.path.join(args.save_path, 'frenet_frame.npz'), 
                e_1=np.array(e_1_arr), 
                e_2=np.array(e_2_arr), 
                e_3=np.array(e_3_arr), 
                curvature=np.array(curvature_arr), 
                torsion=np.array(torsion_arr))
            
        print(f"Frenet frame data saved to {os.path.join(args.save_path, 'frenet_frame.npz')}")