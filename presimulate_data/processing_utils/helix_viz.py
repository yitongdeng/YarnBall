import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from utils.function_derivatives import get_x, get_y, get_z
from utils.helix_functions import get_e1_at_t, get_e2_at_t, get_e3_at_t

def animate_vectors(x, y, z, e_1, e_2, e_3, points, radius, wavelength, gif_filename="helix_vectors.gif"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    
    # Get values at random points
    x_points = get_x(points, radius, wavelength)
    y_points = get_y(points, radius, wavelength)
    z_points = get_z(points)
    e_1_points = get_e1_at_t(points, radius, wavelength)
    e_2_points = get_e2_at_t(points, e_1_points, radius, wavelength)
    e_3_points = get_e3_at_t(e_1_points, e_2_points)
    
    e_1_color = 'r'
    e_2_color = 'g'
    e_3_color = 'b'
    
    # Add every frame animation
    quivers = [None, None, None]
    def update(frame):
        nonlocal quivers
        for quiver in quivers:
            if quiver is not None:
                quiver.remove()
        
        quivers[0] = ax.quiver(
            x[frame], y[frame], z[frame],
            e_1[frame, 0], e_1[frame, 1], e_1[frame, 2],
            length=0.2, normalize=True, color=e_1_color
        )
        quivers[1] = ax.quiver(
            x[frame], y[frame], z[frame], 
            e_2[frame, 0], e_2[frame, 1], e_2[frame, 2],
            length=0.2, normalize=True, color=e_2_color
        )
        quivers[2] = ax.quiver(
            x[frame], y[frame], z[frame],
            e_3[frame, 0], e_3[frame, 1], e_3[frame, 2], 
            length=0.2, normalize=True, color=e_3_color
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
    plt.show()
    ani.save(gif_filename, writer='pillow', fps=5)
    print(f"Saved {gif_filename}")
    plt.close(fig)

def show_random_points(x, y, z, points, radius, wavelength, filename="random_points.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    
    # Get values at random points
    x_points = get_x(points, radius, wavelength)
    y_points = get_y(points, radius, wavelength)
    z_points = get_z(points)
    e_1_points = get_e1_at_t(points, radius, wavelength)
    e_2_points = get_e2_at_t(points, e_1_points, radius, wavelength)
    e_3_points = get_e3_at_t(e_1_points, e_2_points)
    
    e_1_color = 'r'
    e_2_color = 'g'
    e_3_color = 'b'
    
        # Add random frames to the image
    rand_points_sz = .2
    ax.quiver(x_points, y_points, z_points,
              e_1_points[:, 0], e_1_points[:, 1], e_1_points[:, 2],
              length=rand_points_sz, normalize=True, color=e_1_color)
    ax.quiver(x_points, y_points, z_points,
              e_2_points[:, 0], e_2_points[:, 1], e_2_points[:, 2],
              length=rand_points_sz, normalize=True, color=e_2_color)
    ax.quiver(x_points, y_points, z_points,
              e_3_points[:, 0], e_3_points[:, 1], e_3_points[:, 2],
              length=rand_points_sz, normalize=True, color=e_3_color)

    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close(fig)