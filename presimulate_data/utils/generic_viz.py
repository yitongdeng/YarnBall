import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def visualize_points(x, y, z, original_x=None, original_y=None, original_z=None, title="3D Points"):
    """
    Visualize points in 3D space with equal aspect ratio axes.
    
    Args:
        x: x-coordinates of points
        y: y-coordinates of points 
        z: z-coordinates of points
    """
    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, c='r', linestyle='-', alpha=0.5)
    if original_x is not None and original_y is not None and original_z is not None:
        ax.scatter(original_x, original_y, original_z, c='b', s=50, marker='x')
    ax.scatter(x, y, z, c='k', s=3)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    
    # Auto-scale axes to data
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range*0.5, mid_x + max_range*0.5)
    ax.set_ylim(mid_y - max_range*0.5, mid_y + max_range*0.5)
    ax.set_zlim(mid_z - max_range*0.5, mid_z + max_range*0.5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.show()

def visualize_component_derivatives(x_fine, y_fine, z_fine, base_label, x_prime_fine, y_prime_fine, z_prime_fine, first_label, x_second_fine, y_second_fine, z_second_fine, second_label, x_third_fine, y_third_fine, z_third_fine, third_label, title=None):
    # Create a 4x3 grid of subplots with a moderately large figure size
    fig, axs = plt.subplots(4, 3, figsize=(18, 12))
    fig.suptitle('Component Derivatives', fontsize=28)
    t = np.linspace(0, 1, len(x_fine))

    # Plot x components
    axs[0,0].plot(t, x_fine, 'b-', label="x " + base_label)
    axs[0,0].scatter(t, x_fine, c='b', s=3)
    axs[0,0].set_title('X ' + base_label, fontsize=18)
    axs[0,0].grid(True)
    axs[0,0].legend(fontsize=14)

    axs[1,0].plot(t, x_prime_fine, 'r-', label="x " + first_label) 
    axs[1,0].scatter(t, x_prime_fine, c='r', s=3)
    axs[1,0].set_title('X ' + first_label, fontsize=18)
    axs[1,0].grid(True)
    axs[1,0].legend(fontsize=14)

    axs[2,0].plot(t, x_second_fine, 'g-', label="x " + second_label)
    axs[2,0].scatter(t, x_second_fine, c='g', s=3)
    axs[2,0].set_title('X ' + second_label, fontsize=18)
    axs[2,0].grid(True)
    axs[2,0].legend(fontsize=14)

    axs[3,0].plot(t, x_third_fine, 'k-', label="x " + third_label)
    axs[3,0].scatter(t, x_third_fine, c='k', s=3)
    axs[3,0].set_title('X ' + third_label, fontsize=18)
    axs[3,0].grid(True)
    axs[3,0].legend(fontsize=14)

    # Plot y components
    axs[0,1].plot(t, y_fine, 'b-', label="y " + base_label)
    axs[0,1].scatter(t, y_fine, c='b', s=3)
    axs[0,1].set_title('Y ' + base_label, fontsize=18)
    axs[0,1].grid(True)
    axs[0,1].legend(fontsize=14)

    axs[1,1].plot(t, y_prime_fine, 'r-', label="y " + first_label)
    axs[1,1].scatter(t, y_prime_fine, c='r', s=3)
    axs[1,1].set_title('Y ' + first_label, fontsize=18)
    axs[1,1].grid(True)
    axs[1,1].legend(fontsize=14)

    axs[2,1].plot(t, y_second_fine, 'g-', label="y " + second_label)
    axs[2,1].scatter(t, y_second_fine, c='g', s=3)
    axs[2,1].set_title('Y ' + second_label, fontsize=18)
    axs[2,1].grid(True)
    axs[2,1].legend(fontsize=14)

    axs[3,1].plot(t, y_third_fine, 'k-', label="y " + third_label)
    axs[3,1].scatter(t, y_third_fine, c='k', s=3)
    axs[3,1].set_title('Y ' + third_label, fontsize=18)
    axs[3,1].grid(True)
    axs[3,1].legend(fontsize=14)

    # Plot z components
    axs[0,2].plot(t, z_fine, 'b-', label="z " + base_label)
    axs[0,2].scatter(t, z_fine, c='b', s=3)
    axs[0,2].set_title('Z ' + base_label, fontsize=18)
    axs[0,2].grid(True)
    axs[0,2].legend(fontsize=14)

    axs[1,2].plot(t, z_prime_fine, 'r-', label="z " + first_label)
    axs[1,2].scatter(t, z_prime_fine, c='r', s=3)
    axs[1,2].set_title('Z ' + first_label, fontsize=18)
    axs[1,2].grid(True)
    axs[1,2].legend(fontsize=14)

    axs[2,2].plot(t, z_second_fine, 'g-', label="z " + second_label)
    axs[2,2].scatter(t, z_second_fine, c='g', s=3)
    axs[2,2].set_title('Z ' + second_label, fontsize=18)
    axs[2,2].grid(True)
    axs[2,2].legend(fontsize=14)

    axs[3,2].plot(t, z_third_fine, 'k-', label="z " + third_label)
    axs[3,2].scatter(t, z_third_fine, c='k', s=3)
    axs[3,2].set_title('Z ' + third_label, fontsize=18)
    axs[3,2].grid(True)
    axs[3,2].legend(fontsize=14)

    # Set axis labels font size
    for i in range(4):
        for j in range(3):
            axs[i, j].tick_params(axis='both', labelsize=14)
            axs[i, j].set_xlabel('t', fontsize=16)
            axs[i, j].set_ylabel('Value', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if title is not None:
        fig.savefig(title + "_components.png", dpi=150)
    plt.show()
        
def visualize_spline_derivatives(x_fine, y_fine, z_fine, base_label, x_prime_fine, y_prime_fine, z_prime_fine, first_label, x_second_fine, y_second_fine, z_second_fine, second_label, x_third_fine, y_third_fine, z_third_fine, third_label, title=None):
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 5))
    # Create color gradient from blue to red based on position in curve
    colors = np.zeros((len(x_fine), 3))
    colors[:, 0] = np.linspace(0, 1, len(x_fine))  # Red component
    colors[:, 2] = np.linspace(1, 0, len(x_fine))  # Blue component
    # Create slider
    from matplotlib.widgets import Slider
    
    # First subplot - 3D view
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    min_val = min(min(x_fine), min(y_fine), min(z_fine))
    max_val = max(max(x_fine), max(y_fine), max(z_fine))
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_zlim(min_val, max_val)
    
    # Create initial plots with full data
    line1, = ax1.plot(x_fine, y_fine, z_fine, label='Degree-5 Spline', color='blue')
    scatter1 = ax1.scatter(x_fine, y_fine, z_fine, color=colors, label='Data Points')
    ax1.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax1.set_title(base_label)
    ax1.mouse_init()
    ax1.view_init(elev=20, azim=45)
    ax1.legend()

    # Second subplot - XY projection
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.set_box_aspect([1, 1, 1])
    min_val = min(min(x_prime_fine), min(y_prime_fine), min(z_prime_fine))
    max_val = max(max(x_prime_fine), max(y_prime_fine), max(z_prime_fine))
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.set_zlim(min_val, max_val)
    
    line2, = ax2.plot(x_prime_fine, y_prime_fine, z_prime_fine, label='Degree-5 Spline', color='blue')
    scatter2 = ax2.scatter(x_prime_fine, y_prime_fine, z_prime_fine, color=colors, label='Data Points')
    ax2.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax2.set_title(first_label)
    ax2.mouse_init()
    ax2.view_init(elev=20, azim=45)
    ax2.legend()

    # Third subplot - Second Derivative View
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.set_box_aspect([1, 1, 1])
    min_val = min(min(x_second_fine), min(y_second_fine), min(z_second_fine))
    max_val = max(max(x_second_fine), max(y_second_fine), max(z_second_fine))
    ax3.set_xlim(min_val, max_val)
    ax3.set_ylim(min_val, max_val)
    ax3.set_zlim(min_val, max_val)
    
    line3, = ax3.plot(x_second_fine, y_second_fine, z_second_fine, label='Degree-5 Spline', color='blue')
    scatter3 = ax3.scatter(x_second_fine, y_second_fine, z_second_fine, color=colors, label='Data Points')
    ax3.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax3.set_title(second_label)
    ax3.mouse_init()
    ax3.view_init(elev=20, azim=45)
    ax3.legend()
    
    # Fourth subplot - Third Derivative View
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.set_box_aspect([1, 1, 1])
    min_val = min(min(x_third_fine), min(y_third_fine), min(z_third_fine))
    max_val = max(max(x_third_fine), max(y_third_fine), max(z_third_fine))
    ax4.set_xlim(min_val, max_val)
    ax4.set_ylim(min_val, max_val)
    ax4.set_zlim(min_val, max_val)
    
    line4, = ax4.plot(x_third_fine, y_third_fine, z_third_fine, label='Degree-5 Spline', color='blue')
    scatter4 = ax4.scatter(x_third_fine, y_third_fine, z_third_fine, color=colors, label='Data Points')
    ax4.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax4.mouse_init()
    ax4.view_init(elev=20, azim=45)
    ax4.set_title(third_label)
    ax4.legend()

    # Add slider
    plt.subplots_adjust(bottom=0.2)  # Make room for slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Points', 1, len(x_fine), valinit=len(x_fine), valstep=1)

    def update(val):
        idx = int(slider.val)
        
        # Update first plot
        line1.set_data_3d(x_fine[:idx], y_fine[:idx], z_fine[:idx])
        scatter1._offsets3d = (x_fine[:idx], y_fine[:idx], z_fine[:idx])
        scatter1.set_color(colors[:idx])
        
        # Update second plot
        line2.set_data_3d(x_prime_fine[:idx], y_prime_fine[:idx], z_prime_fine[:idx])
        scatter2._offsets3d = (x_prime_fine[:idx], y_prime_fine[:idx], z_prime_fine[:idx])
        scatter2.set_color(colors[:idx])
        
        # Update third plot
        line3.set_data_3d(x_second_fine[:idx], y_second_fine[:idx], z_second_fine[:idx])
        scatter3._offsets3d = (x_second_fine[:idx], y_second_fine[:idx], z_second_fine[:idx])
        scatter3.set_color(colors[:idx])
        
        # Update fourth plot
        line4.set_data_3d(x_third_fine[:idx], y_third_fine[:idx], z_third_fine[:idx])
        scatter4._offsets3d = (x_third_fine[:idx], y_third_fine[:idx], z_third_fine[:idx])
        scatter4.set_color(colors[:idx])
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.tight_layout()
    plt.show()
    if title is not None:
        plt.savefig(title + ".png")

def compare_spline_derivatives(x_fine, y_fine, z_fine, spline_title, x_prime_fine, y_prime_fine, z_prime_fine, prime_title, x_second_fine, y_second_fine, z_second_fine, second_title, x_third_fine, y_third_fine, z_third_fine, third_title, \
                               x_fine_2, y_fine_2, z_fine_2, spline_title_2, x_prime_fine_2, y_prime_fine_2, z_prime_fine_2, prime_title_2, x_second_fine_2, y_second_fine_2, z_second_fine_2, second_title_2, x_third_fine_2, y_third_fine_2, z_third_fine_2, third_title_2, \
                               title=None):
    # Debug: Check input data
    print(f"compare_spline_derivatives called with title: {title}")
    print(f"Data shapes - x_fine: {x_fine.shape}, y_fine: {y_fine.shape}, z_fine: {z_fine.shape}")
    print(f"Data shapes - x_fine_2: {x_fine_2.shape}, y_fine_2: {y_fine_2.shape}, z_fine_2: {z_fine_2.shape}")
    
    # Check for empty or invalid data
    if len(x_fine) == 0 or len(y_fine) == 0 or len(z_fine) == 0:
        print("ERROR: Empty data arrays in compare_spline_derivatives")
        return
    if len(x_fine_2) == 0 or len(y_fine_2) == 0 or len(z_fine_2) == 0:
        print("ERROR: Empty data arrays in compare_spline_derivatives")
        return
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 10))
    # Create color gradient from blue to red based on position in curve
    colors = np.zeros((len(x_fine), 3))
    colors[:, 0] = np.linspace(0, 1, len(x_fine))  # Red component
    colors[:, 2] = np.linspace(1, 0, len(x_fine))  # Blue component
    # Create slider
    from matplotlib.widgets import Slider
    
    # First row - original spline
    # First subplot - 3D view
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    min_val = min(min(x_fine), min(y_fine), min(z_fine))
    max_val = max(max(x_fine), max(y_fine), max(z_fine))
    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_zlim(min_val, max_val)
    
    line1, = ax1.plot(x_fine, y_fine, z_fine, label='Degree-5 Spline', color='blue')
    scatter1 = ax1.scatter(x_fine, y_fine, z_fine, color=colors, label='Data Points')
    ax1.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax1.set_title(spline_title)
    ax1.mouse_init()
    ax1.view_init(elev=20, azim=45)
    ax1.legend()

    ax2 = fig.add_subplot(242, projection='3d')
    ax2.set_box_aspect([1, 1, 1])
    min_val = min(min(x_prime_fine), min(y_prime_fine), min(z_prime_fine))
    max_val = max(max(x_prime_fine), max(y_prime_fine), max(z_prime_fine))
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.set_zlim(min_val, max_val)
    
    line2, = ax2.plot(x_prime_fine, y_prime_fine, z_prime_fine, label='Degree-5 Spline', color='blue')
    scatter2 = ax2.scatter(x_prime_fine, y_prime_fine, z_prime_fine, color=colors, label='Data Points')
    ax2.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax2.set_title(prime_title)
    ax2.mouse_init()
    ax2.view_init(elev=20, azim=45)
    ax2.legend()

    ax3 = fig.add_subplot(243, projection='3d')
    ax3.set_box_aspect([1, 1, 1])
    min_val = min(min(x_second_fine), min(y_second_fine), min(z_second_fine))
    max_val = max(max(x_second_fine), max(y_second_fine), max(z_second_fine))
    ax3.set_xlim(min_val, max_val)
    ax3.set_ylim(min_val, max_val)
    ax3.set_zlim(min_val, max_val)
    
    line3, = ax3.plot(x_second_fine, y_second_fine, z_second_fine, label='Degree-5 Spline', color='blue')
    scatter3 = ax3.scatter(x_second_fine, y_second_fine, z_second_fine, color=colors, label='Data Points')
    ax3.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax3.set_title(second_title)
    ax3.mouse_init()
    ax3.view_init(elev=20, azim=45)
    ax3.legend()
    
    ax4 = fig.add_subplot(244, projection='3d')
    ax4.set_box_aspect([1, 1, 1])
    min_val = min(min(x_third_fine), min(y_third_fine), min(z_third_fine))
    max_val = max(max(x_third_fine), max(y_third_fine), max(z_third_fine))
    ax4.set_xlim(min_val, max_val)
    ax4.set_ylim(min_val, max_val)
    ax4.set_zlim(min_val, max_val)
    
    line4, = ax4.plot(x_third_fine, y_third_fine, z_third_fine, label='Degree-5 Spline', color='blue')
    scatter4 = ax4.scatter(x_third_fine, y_third_fine, z_third_fine, color=colors, label='Data Points')
    ax4.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax4.mouse_init()
    ax4.view_init(elev=20, azim=45)
    ax4.set_title(third_title)
    ax4.legend()

    # Second row - second spline
    ax5 = fig.add_subplot(245, projection='3d')
    ax5.set_box_aspect([1, 1, 1])
    min_val = min(min(x_fine_2), min(y_fine_2), min(z_fine_2))
    max_val = max(max(x_fine_2), max(y_fine_2), max(z_fine_2))
    ax5.set_xlim(min_val, max_val)
    ax5.set_ylim(min_val, max_val)
    ax5.set_zlim(min_val, max_val)
    
    line5, = ax5.plot(x_fine_2, y_fine_2, z_fine_2, label='Degree-5 Spline', color='red')
    scatter5 = ax5.scatter(x_fine_2, y_fine_2, z_fine_2, color=colors, label='Data Points')
    ax5.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax5.set_title(spline_title_2)
    ax5.mouse_init()
    ax5.view_init(elev=20, azim=45)
    ax5.legend()

    ax6 = fig.add_subplot(246, projection='3d')
    ax6.set_box_aspect([1, 1, 1])
    min_val = min(min(x_prime_fine_2), min(y_prime_fine_2), min(z_prime_fine_2))
    max_val = max(max(x_prime_fine_2), max(y_prime_fine_2), max(z_prime_fine_2))
    ax6.set_xlim(min_val, max_val)
    ax6.set_ylim(min_val, max_val)
    ax6.set_zlim(min_val, max_val)
    
    line6, = ax6.plot(x_prime_fine_2, y_prime_fine_2, z_prime_fine_2, label='Degree-5 Spline', color='red')
    scatter6 = ax6.scatter(x_prime_fine_2, y_prime_fine_2, z_prime_fine_2, color=colors, label='Data Points')
    ax6.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax6.set_title(prime_title_2)
    ax6.mouse_init()
    ax6.view_init(elev=20, azim=45)
    ax6.legend()

    ax7 = fig.add_subplot(247, projection='3d')
    ax7.set_box_aspect([1, 1, 1])
    min_val = min(min(x_second_fine_2), min(y_second_fine_2), min(z_second_fine_2))
    max_val = max(max(x_second_fine_2), max(y_second_fine_2), max(z_second_fine_2))
    ax7.set_xlim(min_val, max_val)
    ax7.set_ylim(min_val, max_val)
    ax7.set_zlim(min_val, max_val)
    
    line7, = ax7.plot(x_second_fine_2, y_second_fine_2, z_second_fine_2, label='Degree-5 Spline', color='red')
    scatter7 = ax7.scatter(x_second_fine_2, y_second_fine_2, z_second_fine_2, color=colors, label='Data Points')
    ax7.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax7.set_title(second_title_2)
    ax7.mouse_init()
    ax7.view_init(elev=20, azim=45)
    ax7.legend()

    ax8 = fig.add_subplot(248, projection='3d')
    ax8.set_box_aspect([1, 1, 1])
    min_val = min(min(x_third_fine_2), min(y_third_fine_2), min(z_third_fine_2))
    max_val = max(max(x_third_fine_2), max(y_third_fine_2), max(z_third_fine_2))
    ax8.set_xlim(min_val, max_val)
    ax8.set_ylim(min_val, max_val)
    ax8.set_zlim(min_val, max_val)
    
    line8, = ax8.plot(x_third_fine_2, y_third_fine_2, z_third_fine_2, label='Degree-5 Spline', color='red')
    scatter8 = ax8.scatter(x_third_fine_2, y_third_fine_2, z_third_fine_2, color=colors, label='Data Points')
    ax8.scatter([0], [0], [0], color='green', label='Origin', marker='*', s=50)
    ax8.mouse_init()
    ax8.view_init(elev=20, azim=45)
    ax8.set_title(third_title_2)
    ax8.legend()

    # Add slider
    plt.subplots_adjust(bottom=0.2)  # Make room for slider
    ax_slider = plt.axes([0.1, 0.05, 0.65, 0.03])  # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Points', 1, len(x_fine), valinit=len(x_fine), valstep=1)

    def update(val):
        idx = int(slider.val)
        
        # Update first row plots
        line1.set_data_3d(x_fine[:idx], y_fine[:idx], z_fine[:idx])
        scatter1._offsets3d = (x_fine[:idx], y_fine[:idx], z_fine[:idx])
        scatter1.set_color(colors[:idx])
        
        line2.set_data_3d(x_prime_fine[:idx], y_prime_fine[:idx], z_prime_fine[:idx])
        scatter2._offsets3d = (x_prime_fine[:idx], y_prime_fine[:idx], z_prime_fine[:idx])
        scatter2.set_color(colors[:idx])
        
        line3.set_data_3d(x_second_fine[:idx], y_second_fine[:idx], z_second_fine[:idx])
        scatter3._offsets3d = (x_second_fine[:idx], y_second_fine[:idx], z_second_fine[:idx])
        scatter3.set_color(colors[:idx])
        
        line4.set_data_3d(x_third_fine[:idx], y_third_fine[:idx], z_third_fine[:idx])
        scatter4._offsets3d = (x_third_fine[:idx], y_third_fine[:idx], z_third_fine[:idx])
        scatter4.set_color(colors[:idx])

        # Update second row plots
        line5.set_data_3d(x_fine_2[:idx], y_fine_2[:idx], z_fine_2[:idx])
        scatter5._offsets3d = (x_fine_2[:idx], y_fine_2[:idx], z_fine_2[:idx])
        scatter5.set_color(colors[:idx])
        
        line6.set_data_3d(x_prime_fine_2[:idx], y_prime_fine_2[:idx], z_prime_fine_2[:idx])
        scatter6._offsets3d = (x_prime_fine_2[:idx], y_prime_fine_2[:idx], z_prime_fine_2[:idx])
        scatter6.set_color(colors[:idx])
        
        line7.set_data_3d(x_second_fine_2[:idx], y_second_fine_2[:idx], z_second_fine_2[:idx])
        scatter7._offsets3d = (x_second_fine_2[:idx], y_second_fine_2[:idx], z_second_fine_2[:idx])
        scatter7.set_color(colors[:idx])
        
        line8.set_data_3d(x_third_fine_2[:idx], y_third_fine_2[:idx], z_third_fine_2[:idx])
        scatter8._offsets3d = (x_third_fine_2[:idx], y_third_fine_2[:idx], z_third_fine_2[:idx])
        scatter8.set_color(colors[:idx])
        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.tight_layout()
    plt.show()
    if title is not None:
        plt.savefig(title + ".png")

def visualize_comb(x_fine, y_fine, z_fine, magnitude, vector, label= "Curvature",title=None):
    # Curvature comb visualization
    fig_comb = plt.figure(figsize=(10, 8))
    ax_3d = fig_comb.add_subplot(111, projection='3d')
    comb_scale = 50.0
    points_3d = np.column_stack([x_fine, y_fine, z_fine])
    # Create color map from blue to red based on curvature values
    plt.cm.RdBu_r(plt.Normalize()(magnitude))
    ax_3d.scatter(x_fine, y_fine, z_fine, c=magnitude, cmap='RdBu_r', s=10)
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max()), cmap='RdBu_r'), 
                 ax=ax_3d, label=label)
    for i in range(0, len(points_3d)):
        point = points_3d[i]
        normal = vector[i]
        tooth_length = magnitude[i] * comb_scale
        tooth_end = point + normal * tooth_length
        ax_3d.plot([point[0], tooth_end[0]], [point[1], tooth_end[1]], [point[2], tooth_end[2]], 'r-', linewidth=1, alpha=0.7)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    # Set equal aspect ratio for all axes
    max_range = np.array([
        x_fine.max() - x_fine.min(),
        y_fine.max() - y_fine.min(), 
        z_fine.max() - z_fine.min()
    ]).max()
    
    max_val = max(x_fine.max(), y_fine.max(), z_fine.max())
    min_val = min(x_fine.min(), y_fine.min(), z_fine.min())

    ax_3d.set_xlim(min_val, max_val)
    ax_3d.set_ylim(min_val, max_val)
    ax_3d.set_zlim(min_val, max_val)
    ax_3d.set_box_aspect([1,1,1])
    ax_3d.set_title(f'3D {label} Comb')
    ax_3d.legend()
    plt.tight_layout()
    plt.show()
    
def visualize_curvature_and_torsion(t, curvature, torsion, title=None):
    torsion_zero_indices = np.where(torsion < 1e-3)[0]
    curvature_zero_indices = np.where(curvature < 1e-3)[0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(t[torsion_zero_indices], torsion[torsion_zero_indices], color='blue', marker='o', s=10)
    ax.scatter(t[curvature_zero_indices], curvature[curvature_zero_indices], color='red', marker='o', s=8)
    ax.plot(t, curvature, 'r-')
    ax.plot(t, torsion, 'b-')
    ax.set_xlabel('Curvature')
    ax.set_ylabel('Torsion')
    ax.set_title(title)
    plt.show()
    
def visualize_zero_crossings(x_cs, y_cs, z_cs, x_prime_cs, y_prime_cs, z_prime_cs, x_second_cs, y_second_cs, z_second_cs, x_third_cs, y_third_cs, z_third_cs, arc_len, title=None):
    fine_t = np.linspace(0, arc_len[-1], 1000)
    x_vals = x_cs(arc_len)
    y_vals = y_cs(arc_len)
    z_vals = z_cs(arc_len)
    x_vals_fine = x_cs(fine_t)
    y_vals_fine = y_cs(fine_t)
    z_vals_fine = z_cs(fine_t)
    x_deriv = x_prime_cs(arc_len)
    y_deriv = y_prime_cs(arc_len)
    z_deriv = z_prime_cs(arc_len)
    x_deriv_fine = x_prime_cs(fine_t)
    y_deriv_fine = y_prime_cs(fine_t)
    z_deriv_fine = z_prime_cs(fine_t)
    x_second_deriv = x_second_cs(arc_len)
    y_second_deriv = y_second_cs(arc_len)
    z_second_deriv = z_second_cs(arc_len)
    x_second_deriv_fine = x_second_cs(fine_t)
    y_second_deriv_fine = y_second_cs(fine_t)
    z_second_deriv_fine = z_second_cs(fine_t)
    x_third_deriv = x_third_cs(arc_len)
    y_third_deriv = y_third_cs(arc_len)
    z_third_deriv = z_third_cs(arc_len)
    x_third_deriv_fine = x_third_cs(fine_t)
    y_third_deriv_fine = y_third_cs(fine_t)
    z_third_deriv_fine = z_third_cs(fine_t)
    x_first_zeros = []
    y_first_zeros = []
    z_first_zeros = []
    x_second_zeros = []
    y_second_zeros = []
    z_second_zeros = []
    x_third_zeros = []
    y_third_zeros = []
    z_third_zeros = []
    for i in range(len(arc_len) - 1):
        if x_deriv[i] * x_deriv[i+1] <= 0:
            x_first_zeros.append(i)
        if y_deriv[i] * y_deriv[i+1] <= 0:
            y_first_zeros.append(i)
        if z_deriv[i] * z_deriv[i+1] <= 0:
            z_first_zeros.append(i)
        if x_second_deriv[i] * x_second_deriv[i+1] <= 0:
            x_second_zeros.append(i)
        if y_second_deriv[i] * y_second_deriv[i+1] <= 0:
            y_second_zeros.append(i)
        if z_second_deriv[i] * z_second_deriv[i+1] <= 0:
            z_second_zeros.append(i)
        if x_third_deriv[i] * x_third_deriv[i+1] <= 0:
            x_third_zeros.append(i)
        if y_third_deriv[i] * y_third_deriv[i+1] <= 0:
            y_third_zeros.append(i)
        if z_third_deriv[i] * z_third_deriv[i+1] <= 0:
            z_third_zeros.append(i)
  
    # Create a figure with 3 subplots for derivatives
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(15, 5))
    
    # Plot the original curve
    ax0.scatter(arc_len, x_vals, c='r', label='x Original')
    ax0.scatter(arc_len, y_vals, c='g', label='y Original') 
    ax0.scatter(arc_len, z_vals, c='b', label='z Original')
    ax0.plot(fine_t, x_vals_fine, 'r--', label='x Spline')
    ax0.plot(fine_t, y_vals_fine, 'g--', label='y Spline')
    ax0.plot(fine_t, z_vals_fine, 'b--', label='z Spline')
    # Add vertical lines at arc length values
    #for i in range(len(arc_len)):
    #    ax0.axvline(x=arc_len[i], color='gray', linestyle='--', alpha=0.01)
    ax0.grid(True)
    ax0.legend()
    ax0.set_title('Original Curve')
    ax0.set_xlabel('Arc Length')
    ax0.set_ylabel('Original Curve Values')
    
    # First derivatives
    ax1.scatter(arc_len, x_deriv, c='r', label='x\'')
    ax1.scatter(arc_len, y_deriv, c='g', label='y\'') 
    ax1.scatter(arc_len, z_deriv, c='b', label='z\'')
    ax1.plot(fine_t, x_deriv_fine, 'r--', label='x\' Spline')
    ax1.plot(fine_t, y_deriv_fine, 'g--', label='y\' Spline')
    ax1.plot(fine_t, z_deriv_fine, 'b--', label='z\' Spline')
    # Add vertical lines at arc length values
    #for i in range(len(arc_len)):
    #    ax1.axvline(x=arc_len[i], color='gray', linestyle='--', alpha=0.01)
    for i in x_first_zeros:
        if i in y_first_zeros and i in z_first_zeros:
            ax1.axvspan(arc_len[i], arc_len[i+1], color='yellow', alpha=0.2)
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('First Derivatives')
    ax1.set_xlabel('Arc Length')
    ax1.set_ylabel('First Derivative Values')

    # Second derivatives
    ax2.scatter(arc_len, x_second_deriv, c='r', label='x\'\'')
    ax2.scatter(arc_len, y_second_deriv, c='g', label='y\'\'')
    ax2.scatter(arc_len, z_second_deriv, c='b', label='z\'\'')
    ax2.plot(fine_t, x_second_deriv_fine, 'r--', label='x\'\' Spline')
    ax2.plot(fine_t, y_second_deriv_fine, 'g--', label='y\'\' Spline')
    ax2.plot(fine_t, z_second_deriv_fine, 'b--', label='z\'\' Spline')
    # Add vertical lines at arc length values
    #for i in range(len(arc_len)):
    #    ax2.axvline(x=arc_len[i], color='gray', linestyle='--', alpha=0.01)
    for i in x_second_zeros:
        if i in y_second_zeros and i in z_second_zeros:
            ax2.axvspan(arc_len[i], arc_len[i+1], color='yellow', alpha=0.2)
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Second Derivatives')
    ax2.set_xlabel('Arc Length')
    ax2.set_ylabel('Second Derivative Values')
    

    # Third derivatives
    ax3.scatter(arc_len, x_third_deriv, c='r', label='x\'\'\'')
    ax3.scatter(arc_len, y_third_deriv, c='g', label='y\'\'\'')
    ax3.scatter(arc_len, z_third_deriv, c='b', label='z\'\'\'')
    ax3.plot(fine_t, x_third_deriv_fine, 'r--', label='x\'\'\' Spline')
    ax3.plot(fine_t, y_third_deriv_fine, 'g--', label='y\'\'\' Spline')
    ax3.plot(fine_t, z_third_deriv_fine, 'b--', label='z\'\'\' Spline')
    # Add vertical lines at arc length values
    #for i in range(len(arc_len)):
    #    ax3.axvline(x=arc_len[i], color='gray', linestyle='--', alpha=0.3)
    
    for i in x_third_zeros:
        if i in y_third_zeros and i in z_third_zeros:
            ax3.axvspan(arc_len[i], arc_len[i+1], color='yellow', alpha=0.2)
        
    ax3.grid(True)
    ax3.legend()
    ax3.set_title('Third Derivatives')
    ax3.set_xlabel('Arc Length')
    ax3.set_ylabel('Third Derivative Values')

    plt.tight_layout()
    plt.show()
    
