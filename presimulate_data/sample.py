from sampling_utils.rod.RodHelixConverter import * 
from sampling_utils.rod.rod_generator import * 
from scipy.stats import qmc
np.random.seed(42)
#from io_utils import *
import os

def is_proper_rotation(R, rtol=1e-5, atol=1e-8):
    """
    Parameters
    ----------
    R : (n, 3, 3) ndarray
        Stack of candidate rotation matrices.
    rtol, atol : float
        Relative / absolute tolerances for the orthogonality test.

    Returns
    -------
    mask : (n,) boolean ndarray
        True where R[i] is a proper rotation matrix.
    """
    # 1. orthogonality: RᵀR ≈ I
    RtR = np.matmul(R.transpose(0, 2, 1), R)        # shape (n, 3, 3)
    I = np.eye(3)
    ortho = np.allclose(RtR, I, rtol=rtol, atol=atol)

    # 2. right-handedness: det(R) ≈ +1
    dets = np.linalg.det(R)
    right_handed = np.isclose(dets, 1.0, rtol=rtol, atol=atol)

    return ortho & right_handed

# applying twist in gen. coord space, and changing scalp normal
def add_twist_tan(pos, theta, twist_freq): # twist_freq in units 1/length
    helix = RodHelixConverter.rod_to_helix(pos=pos, theta=theta)
    # q = [twist_{1}, curvature_{1, 1}, curvature_{1, 2}, ..., twist_{n}, curvature_{n, 1}, curvature_{n, 2}]
    # index into every 3rd element of q randomly and add a twist
    num_twists = len(helix.q) // 3
    total_twists = int(twist_freq * num_twists)
    twist_indices = np.random.choice(np.arange(num_twists), size=total_twists, replace=False)
    q_indices = 3 * twist_indices
    helix.q[q_indices] += np.random.uniform(-np.pi, np.pi, size=total_twists)

    tau = helix.q[::3] # the perturbation injected
    
    return *RodHelixConverter.helix_to_rod(helix), tau # returns pos, theta for DER

def extrapolate_segment(pos):
    # pos: [n, 3]
    return np.vstack([pos, 2*pos[[-1]]-pos[[-2]]]) 

if __name__ == "__main__": 
    #
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok = True)
    out_dir = os.path.join(logs_dir, "sampled")
    os.makedirs(out_dir, exist_ok = True)
    #
    height_scale = 0.5
    n=100 # number of segments

    # Define parameter ranges
    param_bounds = np.array([
        [0.3, 0.7],                       # radius (m)
        [0.01, 0.99], #[0.01, 1.0],       # curl frequency (m^-1)
        [0.01, 0.3], # [0.1, 1.0]          # twist frequency (m^-1)
    ])

    n_params = param_bounds.shape[0]
    n_strands = 3

    sampler = qmc.LatinHypercube(d=n_params, seed = 42)
    lhs_sample = sampler.random(n=n_strands)
    scaled_samples = qmc.scale(lhs_sample, param_bounds[:,0], param_bounds[:,1])

    poses, thetas = [], []
    sims = []

    strand_labels = []

    poss = []
    frames = []
    taus = []
    for i, sample in enumerate(scaled_samples):
        r, f, tf = list(sample)
        strand_labels.append({'r': r, 'f': f, 'tf': tf})
        pos, theta = RodGenerator.example_rod(n, r, f, height_scale)
        pos, theta, tau = add_twist_tan(pos, theta, tf)

        pos_extrap = extrapolate_segment(pos)
        t = pos_extrap[1:]-pos_extrap[:-1]
        t = t / np.linalg.norm(t, axis=-1, keepdims = True)
        t = t[:, np.newaxis, :]
        bishop = RodUtil.compute_bishop_frames(pos_extrap)

        material = RodUtil.compute_material_frames(theta=np.hstack([theta,theta[[-1]]]), bishop_frame=bishop)
 
        #nb = bishop
        nb = material
        frame = np.concatenate([t, nb], axis=-2)
        assert np.all(is_proper_rotation(frame)), "Not proper rotation!"

        poss.append(pos-pos[0])
        frames.append(frame)
        taus.append(tau)

    poss = np.array(poss)
    frames = np.array(frames)
    taus = np.array(taus)

    np.save(os.path.join(out_dir, f"poss.npy"), poss)
    np.save(os.path.join(out_dir, f"frames.npy"), frames)
    np.save(os.path.join(out_dir, f"taus.npy"), taus)
