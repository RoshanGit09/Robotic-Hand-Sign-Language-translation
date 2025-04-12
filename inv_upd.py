# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, TextBox
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize import minimize

# # Rotation around X-axis (for finger curl)
# def rot_x(theta):
#     c, s = np.cos(theta), np.sin(theta)
#     return np.array([
#         [1, 0,  0],
#         [0, c, -s],
#         [0, s,  c]
#     ])

# # Convert tendon pull to joint angles
# def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
#     pull_ratio = min(pull_amount / max_pull, 1.0)
#     return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# # Compute joint positions in 3D
# def compute_positions(base_pos, lengths, joint_angles):
#     points = [base_pos]
#     transform = np.eye(3)
#     direction = np.array([0.0, 1.0, 0.0])  # Initial direction (along Y-axis)
#     for i in range(3):
#         transform = transform @ rot_x(joint_angles[i])
#         new_point = points[-1] + transform @ (lengths[i] * direction)
#         points.append(new_point)
#     return points

# # Advanced Inverse Kinematics using optimization
# def inverse_kinematics(target_pos, base_pos, lengths, max_pull, max_angles_rad, ratios):
#     """
#     Calculate both tendon pull and joint angles to reach target position
#     Returns: (pull_amount, joint_angles)
#     """
#     def forward_kinematics(pull):
#         angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
#         points = compute_positions(base_pos, lengths, angles)
#         return points[-1], angles
    
#     def objective(pull):
#         tip, _ = forward_kinematics(pull[0])
#         return np.linalg.norm(tip - target_pos)
    
#     # First try to find a solution within bounds
#     result = minimize(objective, x0=[0.5 * max_pull], bounds=[(0.0, max_pull)], method='L-BFGS-B')
    
#     if result.success:
#         best_pull = result.x[0]
#         _, best_angles = forward_kinematics(best_pull)
#         return best_pull, best_angles
#     else:
#         # If constrained optimization fails, try unconstrained
#         result = minimize(objective, x0=[0.5 * max_pull], method='Nelder-Mead')
#         if result.success:
#             best_pull = np.clip(result.x[0], 0.0, max_pull)
#             _, best_angles = forward_kinematics(best_pull)
#             return best_pull, best_angles
    
#     print("IK failed to converge")
#     return None, None

# # Configuration
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],  # Thumb
#     [3, 2.5, 2.0],    # Index
#     [3.5, 3.0, 2.5],  # Middle
#     [3.0, 2.5, 2.4],  # Ring
#     [2.2, 1.8, 1.5],  # Pinky
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers  # How much each joint responds to pull
# max_pull_cm = 2.5
# palm_spacing = 2.5
# base_positions = [np.array([i * palm_spacing, 0.0, 0.0]) for i in range(num_fingers)]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Plot setup
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.35)
# finger_lines = []
# tip_dots = []
# target_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3, markersize=5)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro', markersize=8)
#     tip_dots.append(dot)
#     target, = ax.plot([], [], [], 'go', markersize=8, alpha=0.7)
#     target_dots.append(target)
#     label = ax.text(0, 0, 0, '', fontsize=8)
#     text_labels.append(label)

# # Axis limits
# ax.set_xlim(-2, num_fingers * palm_spacing + 1)
# ax.set_ylim(0, 15)
# ax.set_zlim(-8, 1)
# ax.set_xlabel('X (Palm width)')
# ax.set_ylabel('Y (Finger length)')
# ax.set_zlabel('Z (Bend)')
# ax.set_title('Tendon-Driven Hand with Target Position IK')

# # Sliders
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Target position input boxes
# axbox_x = plt.axes([0.15, 0.05, 0.2, 0.03])
# axbox_y = plt.axes([0.15, 0.10, 0.2, 0.03])
# axbox_z = plt.axes([0.15, 0.15, 0.2, 0.03])
# axbox_finger = plt.axes([0.15, 0.20, 0.2, 0.03])

# target_x = TextBox(axbox_x, 'Target X', initial="0.0")
# target_y = TextBox(axbox_y, 'Target Y', initial="10.0")
# target_z = TextBox(axbox_z, 'Target Z', initial="-3.0")
# finger_select = TextBox(axbox_finger, 'Finger (0-4)', initial="1")

# # Current target positions
# current_target = None
# current_finger = 1

# # Update function for forward kinematics
# def update(val):
#     global current_target
    
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles)
#         xs, ys, zs = zip(*points)
        
#         # Update finger display
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
        
#         # Update text label
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\nPull: {pull:.2f}cm")
        
#         # Update target display if it exists
#         if current_target is not None and i == current_finger:
#             target_dots[i].set_data([current_target[0]], [current_target[1]])
#             target_dots[i].set_3d_properties([current_target[2]])
#         else:
#             target_dots[i].set_data([], [])
#             target_dots[i].set_3d_properties([])
    
#     fig.canvas.draw_idle()

# # Function to set target and solve IK
# def set_target_and_solve(event):
#     global current_target, current_finger
    
#     try:
#         # Get target position from text boxes
#         tx = float(target_x.text)
#         ty = float(target_y.text)
#         tz = float(target_z.text)
#         current_finger = int(finger_select.text)
#         current_finger = np.clip(current_finger, 0, num_fingers-1)
        
#         target_pos = np.array([tx, ty, tz])
#         current_target = target_pos
        
#         # Solve IK for selected finger
#         pull, angles = inverse_kinematics(
#             target_pos, 
#             base_positions[current_finger], 
#             lengths[current_finger], 
#             max_pull_cm, 
#             max_angles_rad[current_finger], 
#             ratios[current_finger]
#         )
        
#         if pull is not None:
#             # Update the slider
#             sliders[current_finger].set_val(pull)
            
#             # Print results
#             print(f"\nIK Solution for {finger_names[current_finger]} finger:")
#             print(f"Target Position: {target_pos}")
#             print(f"Required Pull: {pull:.2f} cm")
#             print(f"Joint Angles (deg): {np.degrees(angles).round(2)}")
            
#             # Update the display
#             points = compute_positions(base_positions[current_finger], lengths[current_finger], angles)
#             actual_pos = points[-1]
#             error = np.linalg.norm(actual_pos - target_pos)
#             print(f"Actual Position: {actual_pos.round(2)}")
#             print(f"Position Error: {error:.4f} cm")
            
#             # Update the text label with IK info
#             text_labels[current_finger].set_text(
#                 f"{finger_names[current_finger]}\n"
#                 f"Pull: {pull:.2f}cm\n"
#                 f"Target: {target_pos.round(1)}\n"
#                 f"Actual: {actual_pos.round(1)}\n"
#                 f"Error: {error:.2f}cm"
#             )
#         else:
#             print("IK failed to find a solution")
        
#         update(None)
        
#     except ValueError:
#         print("Invalid input values")

# # Button to execute IK
# ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
# button = Button(ax_button, 'Solve IK', color='lightgoldenrodyellow')
# button.on_clicked(set_target_and_solve)

# # Button to reset all fingers
# ax_reset = plt.axes([0.4, 0.15, 0.2, 0.05])
# reset_button = Button(ax_reset, 'Reset All', color='lightblue')
# def reset_all(event):
#     for slider in sliders:
#         slider.set_val(0)
#     update(None)
# reset_button.on_clicked(reset_all)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update)

# # Initial update
# update(None)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Rotation around X-axis
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

# Tendon pull to joint angles
def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
    pull_ratio = min(pull_amount / max_pull, 1.0)
    return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# Compute joint positions
def compute_positions(base_pos, lengths, joint_angles):
    points = [base_pos]
    transform = np.eye(3)
    direction = np.array([0.0, 1.0, 0.0])
    for i in range(3):
        transform = transform @ rot_x(joint_angles[i])
        new_point = points[-1] + transform @ (lengths[i] * direction)
        points.append(new_point)
    return points

# Inverse Kinematics solver
def inverse_kinematics(target_pos, base_pos, lengths, max_pull, max_angles_rad, ratios):
    def forward_kinematics(pull):
        angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
        points = compute_positions(base_pos, lengths, angles)
        return points[-1], angles

    def objective(pull):
        tip, _ = forward_kinematics(pull[0])
        return np.linalg.norm(tip - target_pos)

    result = minimize(objective, x0=[0.5 * max_pull], bounds=[(0.0, max_pull)], method='L-BFGS-B')

    if result.success:
        best_pull = result.x[0]
        _, best_angles = forward_kinematics(best_pull)
        return best_pull, best_angles
    else:
        print("IK failed to converge")
        return None, None

# Setup
finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
num_fingers = 5
lengths = [
    [3.5, 2.5, 1.5], [3, 2.5, 2.0], [3.5, 3.0, 2.5],
    [3.0, 2.5, 2.4], [2.2, 1.8, 1.5]
]
ratios = [[1.0, 0.7, 0.5]] * num_fingers
max_angles_deg = [[55, 90, 90]] * num_fingers
max_pull_cm = 2.5
base_positions = [np.array([i * 2.5, 0.0, 0.0]) for i in range(num_fingers)]
max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# Plot setup
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35)
finger_lines = []
tip_dots = []
target_dots = []
text_labels = []

for _ in range(num_fingers):
    line, = ax.plot([], [], [], 'o-', lw=3, markersize=5)
    finger_lines.append(line)
    dot, = ax.plot([], [], [], 'ro', markersize=8)
    tip_dots.append(dot)
    target, = ax.plot([], [], [], 'go', markersize=8, alpha=0.7)
    target_dots.append(target)
    label = ax.text(0, 0, 0, '', fontsize=8)
    text_labels.append(label)

ax.set_xlim(-2, num_fingers * 2.5 + 1)
ax.set_ylim(0, 15)
ax.set_zlim(-8, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Tendon-Driven Hand (Target Position IK)')

sliders = []
for i in range(num_fingers):
    ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
    slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
    sliders.append(slider)

# Text Boxes
axbox_x = plt.axes([0.15, 0.05, 0.2, 0.03])
axbox_y = plt.axes([0.15, 0.10, 0.2, 0.03])
axbox_z = plt.axes([0.15, 0.15, 0.2, 0.03])
axbox_finger = plt.axes([0.15, 0.20, 0.2, 0.03])

target_x = TextBox(axbox_x, 'Target X', initial="0.0")
target_y = TextBox(axbox_y, 'Target Y', initial="10.0")
target_z = TextBox(axbox_z, 'Target Z', initial="-3.0")
finger_select = TextBox(axbox_finger, 'Finger (0-4)', initial="1")

current_target = None
current_finger = 1

def update(val):
    for i in range(num_fingers):
        pull = sliders[i].val
        joint_angles = get_joint_angles_from_pull(pull, max_pull_cm, max_angles_rad[i], ratios[i])
        points = compute_positions(base_positions[i], lengths[i], joint_angles)
        xs, ys, zs = zip(*points)
        finger_lines[i].set_data(xs, ys)
        finger_lines[i].set_3d_properties(zs)
        tip_dots[i].set_data([xs[-1]], [ys[-1]])
        tip_dots[i].set_3d_properties([zs[-1]])
        text_labels[i].set_position((xs[-1], ys[-1]))
        text_labels[i].set_3d_properties(zs[-1], zdir='z')
        text_labels[i].set_text(f"{finger_names[i]}\nPull: {pull:.2f}cm")
        if current_target is not None and i == current_finger:
            target_dots[i].set_data([current_target[0]], [current_target[1]])
            target_dots[i].set_3d_properties([current_target[2]])
        else:
            target_dots[i].set_data([], [])
            target_dots[i].set_3d_properties([])
    fig.canvas.draw_idle()

def set_target_and_solve(event):
    global current_target, current_finger
    try:
        tx = float(target_x.text)
        ty = float(target_y.text)
        tz = float(target_z.text)
        current_finger = int(finger_select.text)
        current_finger = np.clip(current_finger, 0, num_fingers-1)
        target_pos = np.array([tx, ty, tz])
        current_target = target_pos
        pull, angles = inverse_kinematics(
            target_pos,
            base_positions[current_finger],
            lengths[current_finger],
            max_pull_cm,
            max_angles_rad[current_finger],
            ratios[current_finger]
        )
        if pull is not None:
            sliders[current_finger].set_val(pull)
            print(f"\nFinger: {finger_names[current_finger]}")
            print(f"Target: {target_pos}")
            print(f"Required Pull: {pull:.2f} cm")
            print(f"Joint Angles: {np.degrees(angles).round(2)} deg")
        else:
            print("IK failed to find a solution.")
        update(None)
    except ValueError:
        print("Invalid input values.")

ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
button = Button(ax_button, 'Solve IK', color='lightgoldenrodyellow')
button.on_clicked(set_target_and_solve)

for slider in sliders:
    slider.on_changed(update)

update(None)
plt.show()
