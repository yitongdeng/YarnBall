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
    
def interactive_cubic(orig_x, orig_y, orig_z, x, y, z, e_1, e_2, e_3, q, a, b, c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    ax.scatter(orig_x, orig_y, orig_z, c='b', s=10)
    # Initialize quivers
    quivers = [None, None, None]
    
    # Create slider axis
    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])
    t_slider = plt.Slider(slider_ax, 't', 0, (len(x)-1)/700*16, valinit=0, valstep=.01)
    
    def update(val):
        actual_t = t_slider.val * 700 / 16
        frame = int(actual_t)
        
        # Remove old quivers
        for quiver in quivers:
            if quiver is not None:
                quiver.remove()
        
        # Draw new quivers
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
        fig.canvas.draw_idle()

    # Create text annotations for q, a, b, c values
    text_q = ax.text2D(0.0, 0.95, '', transform=ax.transAxes)
    text_a = ax.text2D(0.0, 0.90, '', transform=ax.transAxes)
    text_b = ax.text2D(0.0, 0.85, '', transform=ax.transAxes)
    text_c = ax.text2D(0.0, 0.80, '', transform=ax.transAxes)
    text_t = ax.text2D(0.0, 0.75, '', transform=ax.transAxes)
    text_n = ax.text2D(0.0, 0.70, '', transform=ax.transAxes)
    text_bin = ax.text2D(0.0, 0.65, '', transform=ax.transAxes)
    
    def update_with_text(val):
        update(val)
        actual_t = t_slider.val * 700 / 16
        frame = int(actual_t)
        text_t.set_text(f'Tangent = {e_1[frame,0]:.3f}, {e_1[frame,1]:.3f}, {e_1[frame,2]:.3f}')
        text_n.set_text(f'Normal = {e_2[frame,0]:.3f}, {e_2[frame,1]:.3f}, {e_2[frame,2]:.3f}')
        text_bin.set_text(f'Binormal = {e_3[frame,0]:.3f}, {e_3[frame,1]:.3f}, {e_3[frame,2]:.3f}')
        text_q.set_text(f'q = {q[frame,0]:.3f}, {q[frame,1]:.3f}, {q[frame,2]:.3f}')
        text_a.set_text(f'a = {a[frame,0]:.3f}, {a[frame,1]:.3f}, {a[frame,2]:.3f}')
        text_b.set_text(f'b = {b[frame,0]:.3f}, {b[frame,1]:.3f}, {b[frame,2]:.3f}')
        text_c.set_text(f'c = {c[frame,0]:.3f}, {c[frame,1]:.3f}, {c[frame,2]:.3f}')
        
    t_slider.on_changed(update_with_text)
    
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
        # Get points along this cubic segment
        x = q[i,0] + a[i,0]*t[i] + b[i,0]*t[i]**2 + c[i,0]*t[i]**3
        y = q[i,1] + a[i,1]*t[i] + b[i,1]*t[i]**2 + c[i,1]*t[i]**3 
        z = q[i,2] + a[i,2]*t[i] + b[i,2]*t[i]**2 + c[i,2]*t[i]**3
        
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
    
def view_cubic_derivatives(q, a, b, c, axial_dist=30):
    t = np.linspace(0, 1, 5).reshape(1, -1)
    t = t.repeat(len(q), axis=0)
    t = np.concatenate([t[0].reshape(1, -1) / 3, ((t + 1) / 3), (t[-1].reshape(1, -1) + 2) / 3], axis=0)
    q = np.concatenate([q[0].reshape(1, -1), q, q[-1].reshape(1, -1)], axis=0)
    a = np.concatenate([a[0].reshape(1, -1), a, a[-1].reshape(1, -1)], axis=0)
    b = np.concatenate([b[0].reshape(1, -1), b, b[-1].reshape(1, -1)], axis=0)
    c = np.concatenate([c[0].reshape(1, -1), c, c[-1].reshape(1, -1)], axis=0)      

    x, y, z = get_cubic_points(q, a, b, c, t)
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
