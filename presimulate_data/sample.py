from sampling_utils.rod.RodHelixConverter import * 
from sampling_utils.rod.rod_generator import * 
from scipy.stats import qmc
np.random.seed(42)
#from io_utils import *
import os
from scipy.spatial.transform import Rotation as R

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

    return ortho, right_handed

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

def align_rod(positions, frames, target_frame=None):
    """
    Rotate a discrete Cosserat rod so that its ROOT material frame
    (index 0) aligns with `target_frame`, without moving the root vertex.

    Parameters
    ----------
    positions : (n, 3) ndarray
    frames    : (n, 3, 3) ndarray
    target_frame : (3, 3) ndarray or None
        Desired orientation for the root frame.  Identity if None.

    Returns
    -------
    pos_out : (n, 3) ndarray
    frame_out : (n, 3, 3) ndarray
    Q : (3, 3) ndarray
        Rotation that was applied.
    """
    pos   = np.asarray(positions, float)
    fr    = np.asarray(frames,    float)
    n     = pos.shape[0]

    if target_frame is None:
        target_frame = np.eye(3)
    target_frame = np.asarray(target_frame, float)

    # ------------------------------------------------------------------
    # 1. rotation that carries the current root frame to the target
    # ------------------------------------------------------------------
    R0 = fr[0]                      # current root frame (3×3)
    Q  = target_frame @ R0.T        # desired rotation (3×3, det ≈ +1)

    # ------------------------------------------------------------------
    # 2. translate-to-origin  → rotate → translate-back
    # ------------------------------------------------------------------
    p0          = pos[0]            # root position, shape (3,)
    shifted     = pos - p0          # broadcast, shape (n,3)
    pos_out     = np.einsum("ij, kj->ki", Q, shifted) + p0

    # ------------------------------------------------------------------
    # 3. rotate every local frame
    # ------------------------------------------------------------------
    frame_out   = Q[np.newaxis, :, :] @ fr   # (n,3,3)

    return pos_out, frame_out, Q

if __name__ == "__main__": 
    n_strands = 1

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
        [0.2, 0.99], #[0.01, 1.0],       # curl frequency (m^-1)
        [0.01, 0.3], # [0.1, 1.0]          # twist frequency (m^-1)
    ])

    n_params = param_bounds.shape[0]

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
        print(f"Working on {i}")
        r, f, tf = list(sample)
        strand_labels.append({'r': r, 'f': f, 'tf': tf})
        pos, theta = RodGenerator.example_rod(n, r, f, height_scale)
        pos, theta, tau = add_twist_tan(pos, theta, tf)

        # TESTING PURPOSE ONLY
        straight_line = np.zeros_like(pos)
        straight_line[:, 0] = np.arange(pos.shape[0])
        pos = straight_line# + 0.001 * np.random.randn(*straight_line.shape)

        pos_extrap = extrapolate_segment(pos)
        t = pos_extrap[1:]-pos_extrap[:-1]
        t = t / np.linalg.norm(t, axis=-1, keepdims = True)
        t = t[:, np.newaxis, :]
        bishop = RodUtil.compute_bishop_frames(pos_extrap) # compute bishop

        material = RodUtil.compute_material_frames(theta=np.hstack([theta,theta[[-1]]]), bishop_frame=bishop)
 
        nb = bishop
        #nb = material
        frame = np.concatenate([t, nb], axis=-2).transpose((0, 2, 1))

        # orthogonality check
        RTR = np.matmul(frame.transpose(0, 2, 1), frame)
        diff = RTR - np.eye(3)
        max_diff = np.max(np.linalg.norm(diff, axis = (-2, -1)))
        assert max_diff < 0.1, f"Not orthogonal for {i} with max diff: {max_diff}!"

        # is_orthogonal, no_reflection = is_proper_rotation(frame)
        #assert np.all(is_orthogonal & no_reflection), f"For {i},Is orthogonal:\n {np.matmul(frame.transpose(0, 2, 1), frame)}"

        # NO ALIGNMENT
        #pos_aligned, frame_aligned, align_rot = pos, frame, None 
        
        # WITH ALIGNMENT
        pos_aligned, frame_aligned, align_rot = align_rod(pos, frame)
        #R_random = R.random().as_matrix()          # target
        #pos_aligned, frame_aligned, align_rot = align_rod(pos, frame, R_random)
        #

        poss.append(pos_aligned-pos_aligned[0])
        frames.append(frame_aligned)
        taus.append(tau)

    poss = np.array(poss)
    frames = np.array(frames)
    taus = np.array(taus)

    np.save(os.path.join(out_dir, f"poss.npy"), poss)
    np.save(os.path.join(out_dir, f"frames.npy"), frames)
    np.save(os.path.join(out_dir, f"taus.npy"), taus)
