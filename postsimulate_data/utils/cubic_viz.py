import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.cubic_functions import *

def animate_vectors(x, y, z, e_1, e_2, e_3, gif_filename="helix_vectors.gif"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    print("animate_vectors")
    # Add every frame animation
    quivers = [None, None, None]
    def update(frame):
        if frame % 100 == 0:
            print("frame", frame)
        nonlocal quivers
        for quiver in quivers:
            if quiver is not None:
                quiver.remove()
        
        quivers[0] = ax.quiver(
            x[frame], y[frame], z[frame],
            e_1[frame, 0], e_1[frame, 1], e_1[frame, 2],
            length=10, normalize=True, color='r'
        )
        quivers[1] = ax.quiver(
            x[frame], y[frame], z[frame],
            e_2[frame, 0], e_2[frame, 1], e_2[frame, 2],
            length=10, normalize=True, color='g'
        )
        quivers[2] = ax.quiver(
            x[frame], y[frame], z[frame],
            e_3[frame, 0], e_3[frame, 1], e_3[frame, 2],
            length=10, normalize=True, color='b'
        )
        
        return quivers

    ani = animation.FuncAnimation(
        fig, update, frames=len(x), blit=False, repeat=False
    )
    
    # Set up the plot
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Helix')
    ani.save(gif_filename, writer='pillow', fps=500)
    print(f"Saved {gif_filename}")
    plt.close(fig)
    
def interactive_cubic(orig_x, orig_y, orig_z, x, y, z, e_1, e_2, e_3, values_to_graph):
    """
    Interactive cubic visualization with dynamic subplot grid for multiple values.
    
    Args:
        values_to_graph: List of arrays to plot, each will get its own subplot
    """
    # Debug: Check input data
    print(f"interactive_cubic called with {len(values_to_graph)} values to graph")
    print(f"Data shapes - x: {x.shape}, y: {y.shape}, z: {z.shape}")
    print(f"e_1 shape: {e_1.shape}, e_2 shape: {e_2.shape}, e_3 shape: {e_3.shape}")
    
    # Check for empty or invalid data
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        print("ERROR: Empty data arrays in interactive_cubic")
        return
    if len(e_1) == 0 or len(e_2) == 0 or len(e_3) == 0:
        print("ERROR: Empty vector arrays in interactive_cubic")
        return
    # Calculate optimal grid dimensions
    n_plots = len(values_to_graph)
    if n_plots == 0:
        n_plots = 1
        values_to_graph = [np.zeros_like(x)]
    
    # Calculate optimal grid layout
    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))
    
    # Create figure with appropriate size
    fig_width = max(12, 4 * cols)
    fig_height = max(8, 3 * rows + 2)  # Extra space for 3D plot and slider
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create 3D main plot (takes up left half)
    ax_3d = fig.add_subplot(rows, cols + 1, 1, projection='3d')
    ax_3d.set_position([0.0, 0.1, 0.4, 0.6])
    ax_3d.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    ax_3d.scatter(orig_x, orig_y, orig_z, c='b', s=10)
    
    # Create subplots for each value to graph
    subplot_axes = []
    for i in range(n_plots):
        # Calculate position in the grid (skip the first position which is 3D plot)
        grid_pos = i + 2  # Start from position 2
        
        # Check if this value needs 3D plotting
        value_data = values_to_graph[i]
        if isinstance(value_data, tuple):
            values = value_data[0]
        else:
            values = value_data
            
        # Create 3D subplot if values have shape (N, 3)
        if len(values.shape) == 2 and values.shape[1] == 3:
            ax = fig.add_subplot(rows, cols + 1, grid_pos, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols + 1, grid_pos)
        subplot_axes.append(ax)
    
    # Initialize quivers
    quivers = [None, None, None]
    
    # Create slider axis
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    t_slider = plt.Slider(slider_ax, 't', 0, (len(x)-1)/700*16, valinit=0, valstep=.01)
    
    def update(val):
        actual_t = t_slider.val * 700 / 16
        frame = int(actual_t)
        
        # Update each subplot
        for i, (ax, value_data) in enumerate(zip(subplot_axes, values_to_graph)):
            ax.clear()
            # Handle both tuple format (values, name) and array format
            if isinstance(value_data, tuple):
                values, name = value_data
            else:
                values = value_data
                name = f'Plot {i+1}'
            
            # Check if values are 3D (shape (N, 3))
            if len(values.shape) == 2 and values.shape[1] == 3:
                # 3D plot
                #ax.set_xlim(values[:, 0].min(), values[:, 0].max())
                #ax.set_ylim(values[:, 1].min(), values[:, 1].max())
                #ax.set_zlim(values[:, 2].min(), values[:, 2].max())
                ax.set_title(name)
                ax.plot(values[:frame, 0], values[:frame, 1], values[:frame, 2], c='r', linestyle='-', alpha=0.5)
            else:
                # 2D plot
                ax.set_xlim(0, len(values))
                ax.set_ylim(0, np.max(values))
                ax.set_title(name)
                ax.plot(values[:frame], c='r', linestyle='-', alpha=0.5)

        # Remove old quivers
        for quiver in quivers:
            if quiver is not None:
                quiver.remove()
        
        # Draw new quivers
        quivers[0] = ax_3d.quiver(
            x[frame], y[frame], z[frame],
            e_1[frame, 0], e_1[frame, 1], e_1[frame, 2],
            length=10, normalize=True, color='r'
        )
        quivers[1] = ax_3d.quiver(
            x[frame], y[frame], z[frame],
            e_2[frame, 0], e_2[frame, 1], e_2[frame, 2],
            length=10, normalize=True, color='g'
        )
        quivers[2] = ax_3d.quiver(
            x[frame], y[frame], z[frame],
            e_3[frame, 0], e_3[frame, 1], e_3[frame, 2],
            length=10, normalize=True, color='b'
        )
        fig.canvas.draw_idle()

    # Create text annotations for q, a, b, c values
    text_t = ax_3d.text2D(0.0, 0.75, '', transform=ax_3d.transAxes)
    text_n = ax_3d.text2D(0.0, 0.70, '', transform=ax_3d.transAxes)
    text_bin = ax_3d.text2D(0.0, 0.65, '', transform=ax_3d.transAxes)
    
    def update_with_text(val):
        update(val)
        actual_t = t_slider.val * 700 / 16
        frame = int(actual_t)
        text_t.set_text(f'Tangent = {e_1[frame,0]:.3f}, {e_1[frame,1]:.3f}, {e_1[frame,2]:.3f}')
        text_n.set_text(f'Normal = {e_2[frame,0]:.3f}, {e_2[frame,1]:.3f}, {e_2[frame,2]:.3f}')
        text_bin.set_text(f'Binormal = {e_3[frame,0]:.3f}, {e_3[frame,1]:.3f}, {e_3[frame,2]:.3f}')

    t_slider.on_changed(update_with_text)
    
    # Set up the 3D plot
    max_val = max(x.max(), y.max(), z.max())
    min_val = min(x.min(), y.min(), z.min())

    ax_3d.set_xlim(min_val, max_val)
    ax_3d.set_ylim(min_val, max_val)
    ax_3d.set_zlim(min_val, max_val)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Helix')
    
    # Initial plot
    update(0)
    plt.show()
    
def visualize_cubic_sections(q,a,b,c, t, helix_x, helix_y, helix_z):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the original helix points
    ax.scatter(helix_x, helix_y, helix_z, c='g', s=50, label='Original points')
    
    # Plot each cubic segment
    colors = ['r', 'b', 'c', 'm', 'y', 'k']
    
    for i in range(len(q)):
        t_percent = t[i] - np.min(t[i]) 
        t_percent = t_percent / np.max(t_percent)
        print("Length of t_percent", len(t_percent))
        print("Q len", len(q))
        
        q_l = q[i, :]
        a_l = a[i, :]
        b_l = b[i, :]
        c_l = c[i, :]
        q_r = q[i, :]
        a_r = a[i, :]
        b_r = b[i, :]
        c_r = c[i, :]
        if i < len(q) - 1:
            q_r = q[i+1, :]
            a_r = a[i+1, :]
            b_r = b[i+1, :]
            c_r = c[i+1, :]
        if i > 0:
            q_l = q[i-1, :]
            a_l = a[i-1, :]
            b_l = b[i-1, :]
            c_l = c[i-1, :]
        
        # Q, A, B, C are all 3D vectors - interpolate along t_percent for each segment
        q_interp = (q[i, :] + (q_l * (1 - t_percent[:, np.newaxis])) + (q_r * t_percent[:, np.newaxis])) / 2
        a_interp = (a[i, :] + (a_l * (1 - t_percent[:, np.newaxis])) + (a_r * t_percent[:, np.newaxis])) / 2
        b_interp = (b[i, :] + (b_l * (1 - t_percent[:, np.newaxis])) + (b_r * t_percent[:, np.newaxis])) / 2
        c_interp = (c[i, :] + (c_l * (1 - t_percent[:, np.newaxis])) + (c_r * t_percent[:, np.newaxis])) / 2

        print("q_interp", q_interp.shape)
        print("q", q[i, :].shape)
        
        # Get points along this cubic segment using interpolated coefficients
        x = q_interp[:, 0] + a_interp[:, 0]*t[i] + b_interp[:, 0]*t[i]**2 + c_interp[:, 0]*t[i]**3
        y = q_interp[:, 1] + a_interp[:, 1]*t[i] + b_interp[:, 1]*t[i]**2 + c_interp[:, 1]*t[i]**3 
        z = q_interp[:, 2] + a_interp[:, 2]*t[i] + b_interp[:, 2]*t[i]**2 + c_interp[:, 2]*t[i]**3
        
        # Plot the cubic segment
        ax.plot(x, y, z, c=colors[i%len(colors)], linewidth=2)
        
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title('3D Cubic Interpolation')
    
    # Make axes equal to preserve aspect ratio
    max_range = np.array([helix_x.max()-helix_x.min(), 
                         helix_y.max()-helix_y.min(),
                         helix_z.max()-helix_z.min()]).max()
    mid_x = (helix_x.max()+helix_x.min()) * 0.5
    mid_y = (helix_y.max()+helix_y.min()) * 0.5
    mid_z = (helix_z.max()+helix_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    ax.legend()
    ax.mouse_init()  # Enable mouse rotation
    plt.show()
    
def view_cubic_derivatives(t, q, a, b, c, axial_dist=30):
    x, y, z = get_cubic_points(t, q, a, b, c)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(-axial_dist, axial_dist)
    ax.set_ylim(-axial_dist, axial_dist)
    ax.set_zlim(-axial_dist, axial_dist)
    
    # Set view to look from above (top-down view)
    ax.view_init(elev=90, azim=-90)
    # Create three subplots vertically stacked
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})
    
    # Set the same view angle for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=90, azim=-90)
        ax.set_xlim(-axial_dist, axial_dist)
        ax.set_ylim(-axial_dist, axial_dist)
        ax.set_zlim(-axial_dist, axial_dist)
    
    # Plot position
    ax1.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    ax1.scatter(x, y, z, c='k', s=3)
    ax1.set_title('Position')
    
    # Plot first derivative
    dx, dy, dz = get_cubic_prime(a, b, c, t)
    
    dx = dx.flatten()
    dy = dy.flatten()
    dz = dz.flatten()
    ax2.plot(dx, dy, dz, c='b', linestyle='-', alpha=0.5)
    ax2.scatter(dx, dy, dz, c='k', s=3)
    ax2.set_title('First Derivative')
    
    # Plot second derivative
    ddx, ddy, ddz = get_cubic_second_derivative(b, c, t)
    ddx = ddx.flatten()
    ddy = ddy.flatten()
    ddz = ddz.flatten()
    ax3.plot(ddx, ddy, ddz, c='g', linestyle='-', alpha=0.5)
    ax3.scatter(ddx, ddy, ddz, c='k', s=3)
    ax3.set_title('Second Derivative')
    
    plt.tight_layout()
    plt.show()

def visualize_cubic_sections_global_param(q, a, b, c, s, helix_x, helix_y, helix_z, N=20):
    """
    Visualize cubic spline sections using global parameter s for each segment.
    q, a, b, c: arrays of shape (n_segments, 3)
    s: array of arc-length parameters (len = n_segments+1)
    helix_x, helix_y, helix_z: original points
    N: number of points per segment
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(helix_x, helix_y, helix_z, c='g', s=50, label='Original points')
    colors = ['r', 'b', 'c', 'm', 'y', 'k']

    for i in range(len(q)):
        s_segment = np.linspace(s[i], s[i+1], N)
        x = q[i,0] + a[i,0]*s_segment + b[i,0]*s_segment**2 + c[i,0]*s_segment**3
        y = q[i,1] + a[i,1]*s_segment + b[i,1]*s_segment**2 + c[i,1]*s_segment**3
        z = q[i,2] + a[i,2]*s_segment + b[i,2]*s_segment**2 + c[i,2]*s_segment**3
        ax.plot(x, y, z, c=colors[i%len(colors)], linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cubic Interpolation (global parameter)')
    max_range = np.array([helix_x.max()-helix_x.min(), 
                         helix_y.max()-helix_y.min(),
                         helix_z.max()-helix_z.min()]).max()
    mid_x = (helix_x.max()+helix_x.min()) * 0.5
    mid_y = (helix_y.max()+helix_y.min()) * 0.5
    mid_z = (helix_z.max()+helix_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    ax.legend()
    ax.mouse_init()
    plt.show()

def visualize_piecewise_hermite_global_param(P0s, P1s, m0s, m1s, s, helix_x, helix_y, helix_z, N=20):
    """
    Visualize piecewise Hermite cubics using global parameterization.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(helix_x, helix_y, helix_z, c='g', s=50, label='Original points')
    colors = ['r', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(P0s)):
        t_vals = np.linspace(s[i], s[i+1], N)
        tau = (t_vals - s[i]) / (s[i+1] - s[i])
        h00 = 2*tau**3 - 3*tau**2 + 1
        h10 = tau**3 - 2*tau**2 + tau
        h01 = -2*tau**3 + 3*tau**2
        h11 = tau**3 - tau**2
        seg = (
            h00[:,None]*P0s[i]
            + h10[:,None]*(s[i+1]-s[i])*m0s[i]
            + h01[:,None]*P1s[i]
            + h11[:,None]*(s[i+1]-s[i])*m1s[i]
        )
        ax.plot(seg[:,0], seg[:,1], seg[:,2], c=colors[i%len(colors)], linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Piecewise Hermite Cubics (global parameter)')
    max_range = np.array([helix_x.max()-helix_x.min(), helix_y.max()-helix_y.min(), helix_z.max()-helix_z.min()]).max()
    mid_x = (helix_x.max()+helix_x.min()) * 0.5
    mid_y = (helix_y.max()+helix_y.min()) * 0.5
    mid_z = (helix_z.max()+helix_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    ax.legend()
    ax.mouse_init()
    plt.show()
    
def show_derivatives(q, a, b, c, t):
    # Get initial values
    x, y, z = get_cubic_points(t[0], q, a, b, c)
    e_1 = get_e1_cubic(a, b, c, t[0])
    e_2 = get_e2_cubic(b, c, t[0], e_1)
    e_3 = get_e3_cubic(e_1, e_2)
    
    # Create figure and subplots
    fig = plt.figure(figsize=(18, 6))
    ax0 = fig.add_subplot(141, projection='3d')
    ax1 = fig.add_subplot(142)
    ax2 = fig.add_subplot(143) 
    ax3 = fig.add_subplot(144)
    
    # Initialize plots
    curve_plot, = ax0.plot(x, y, z, 'b-', label='Curve')
    points_plot = ax0.scatter(x, y, z, c='r', s=50, label='Points')
    
    e1_x, = ax1.plot(t[0], e_1[0,0], 'r-', label='e1_x')
    e1_y, = ax1.plot(t[0], e_1[0,1], 'g-', label='e1_y')
    e1_z, = ax1.plot(t[0], e_1[0,2], 'b-', label='e1_z')
    
    e2_x, = ax2.plot(t[0], e_2[0,0], 'r-', label='e2_x')
    e2_y, = ax2.plot(t[0], e_2[0,1], 'g-', label='e2_y')
    e2_z, = ax2.plot(t[0], e_2[0,2], 'b-', label='e2_z')
    
    e3_x, = ax3.plot(t[0], e_3[0,0], 'r-', label='e3_x')
    e3_y, = ax3.plot(t[0], e_3[0,1], 'g-', label='e3_y')
    e3_z, = ax3.plot(t[0], e_3[0,2], 'b-', label='e3_z')
    
    # Setup axes
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    ax0.set_title('3D Curve')
    ax0.legend()
    ax0.grid(True)
    
    ax1.set_xlabel('t')
    ax1.set_ylabel('e1 components')
    ax1.set_title('e1 vs t')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('e2 components')
    ax2.set_title('e2 vs t')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_xlabel('t')
    ax3.set_ylabel('e3 components')
    ax3.set_title('e3 vs t')
    ax3.legend()
    ax3.grid(True)
    
    # Create slider
    slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
    t_slider = plt.Slider(slider_ax, 't', 0, len(t)-1, valinit=0, valstep=1)
    
    def update(val):
        i = int(val)
        x, y, z = get_cubic_points(t[i], q, a, b, c)
        e_1 = get_e1_cubic(a, b, c, t[i])
        e_2 = get_e2_cubic(b, c, t[i], e_1)
        e_3 = get_e3_cubic(e_1, e_2)
        
        # Update plots
        curve_plot.set_data(x, y)
        curve_plot.set_3d_properties(z)
        points_plot._offsets3d = (x, y, z)
        
        e1_x.set_data(t[:i+1], e_1[:i+1,0])
        e1_y.set_data(t[:i+1], e_1[:i+1,1])
        e1_z.set_data(t[:i+1], e_1[:i+1,2])
        
        e2_x.set_data(t[:i+1], e_2[:i+1,0])
        e2_y.set_data(t[:i+1], e_2[:i+1,1])
        e2_z.set_data(t[:i+1], e_2[:i+1,2])
        
        e3_x.set_data(t[:i+1], e_3[:i+1,0])
        e3_y.set_data(t[:i+1], e_3[:i+1,1])
        e3_z.set_data(t[:i+1], e_3[:i+1,2])
        
        fig.canvas.draw_idle()
    
    t_slider.on_changed(update)
    plt.tight_layout()
    plt.show()
