import numpy as np
import matplotlib.pyplot as plt
from arm_utils import forward_kinematics, inverse_kinematics, within_joint_limits, JOINT_LIMITS


def draw_arm(theta1, theta2):
    points = forward_kinematics(theta1, theta2)
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    plt.plot(x_coords, y_coords, marker='o', linewidth=3, markersize=8)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(f"θ1 = {np.degrees(theta1):.1f}°, θ2 = {np.degrees(theta2):.1f}°")
    plt.show()

def move_to_target(x,y):
    ik_result = inverse_kinematics(x,y)
    if ik_result is None:
        print(f"Target ({x}, {y}) is unreachable.")
        return
    theta1, theta2 = ik_result
    print(f"Target ({x:.2f}, {y:.2f}) → θ1: {np.degrees(theta1):.2f}°, θ2: {np.degrees(theta2):.2f}°")
    draw_arm(theta1, theta2)

def animate_arm(theta1_start, theta2_start, theta1_end, theta2_end, arm_line, fig, steps = 30, delay = 0.001):
    for i in range(steps + 1):
        t = i / steps
        theta1 = ((1-t) * theta1_start) + (t * theta1_end)
        theta2 = ((1-t) * theta2_start) + (t * theta2_end)
        points = forward_kinematics(theta1, theta2)
        arm_line.set_xdata([p[0] for p in points])
        arm_line.set_ydata([p[1] for p in points])
        fig.canvas.draw()
        plt.pause(delay)  # lets matplotlib process GUI events

def plot_shoulder_limits(ax, L1=1.0):
    t1_min, t1_max = JOINT_LIMITS["theta1"]
    arc_thetas = np.linspace(t1_min, t1_max, 300)
    arc_x = L1 * np.cos(arc_thetas)
    arc_y = L1 * np.sin(arc_thetas)

    ax.plot(arc_x, arc_y, color='#FFBF00', linewidth=1.5, alpha=0.5)


def interactive_mode():
    fig, ax = plt.subplots()
    theta1, theta2 = np.radians(45), np.radians(45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
    ax.set_title("2DOF Robotic Arm Simulator", fontsize=14, fontweight='bold', pad=15)
    ax.text(0, 2.6, 
        "Click to move the arm | Red dot = target | Green = reachable",
        ha='center', fontsize=10, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    plot_reachability(ax)
    plot_shoulder_limits(ax)

    # Initial plot
    theta1, theta2 = np.radians(45), np.radians(45)
    points = forward_kinematics(theta1, theta2)
    arm_line, = ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color="#2274A5",
        linewidth=4,
        marker='o',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=10,
        solid_capstyle="round"
    )
    target_dot, = ax.plot([], [], 'o', color='crimson', markersize=8, zorder=3)
    # Target label (updates with each click)
    target_label = ax.text(0.05, -2.3, "", fontsize=10, color='black', ha='left')
    feedback_label = ax.text(-0.05, -2.3, "", fontsize=10, color='crimson', ha='right')


    def onclick(event):
        nonlocal theta1, theta2
        feedback_label.set_text("")  # Clear old messages
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        print(f"Clicked: ({x:.2f}, {y:.2f})")
        # Set the title and update the target dot
        target_label.set_text(f"Target: ({x:.2f}, {y:.2f})")
        target_dot.set_data([x], [y])

        ik_result = inverse_kinematics(x, y)
        if ik_result is None:
            feedback_label.set_text("Target unreachable")
            fig.canvas.draw()
            return

        
        theta1_target, theta2_target = ik_result

        if not within_joint_limits(theta1_target, theta2_target):
            feedback_label.set_text("Target outside joint limits")
            fig.canvas.draw()
            return
        
        animate_arm(theta1, theta2, theta1_target, theta2_target, arm_line, fig)

        # Update current angles after animation
        theta1, theta2 = theta1_target, theta2_target
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def plot_reachability(ax, resolution=100):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', '#a8e6a3'])  # white for unreachable, soft green for reachable
    """
    Plots reachability shading on the given axes.
    Green = reachable, gray = unreachable.
    """
    x_vals = np.linspace(-2.5, 2.5, resolution)
    y_vals = np.linspace(-2.5, 2.5, resolution)
    reach_map = np.zeros((resolution, resolution))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            ik_result = inverse_kinematics(x, y)
            if ik_result is not None:
                theta1, theta2 = ik_result
                if within_joint_limits(theta1, theta2):
                    reach_map[j, i] = 1  # reachable

    # plot reachability as background image
    ax.imshow(
        reach_map,
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        origin='lower',
        cmap=cmap,
        alpha=0.25,
        zorder=0
    )


if __name__ == "__main__":
    interactive_mode()



