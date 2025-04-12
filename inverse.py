# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
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

# # Convert tendon pull to joint angles (Forward mapping)
# def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
#     pull_ratio = min(pull_amount / max_pull, 1.0)
#     return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# # Convert joint angles to tendon pull (Inverse mapping)
# def get_pull_from_joint_angles(joint_angles, max_pull, max_angles_rad, ratios):
#     # Find the limiting joint (the one requiring the most pull)
#     pull_ratios = [abs(angle) / (r * max_angle) 
#                   for angle, r, max_angle in zip(joint_angles, ratios, max_angles_rad)]
#     max_ratio = max(pull_ratios)
#     pull_amount = max_ratio * max_pull
#     return min(pull_amount, max_pull)

# # Compute joint positions in 3D (Forward kinematics)
# def compute_positions(base_pos, lengths, joint_angles):
#     points = [base_pos]
#     transform = np.eye(3)
#     direction = np.array([0.0, 1.0, 0.0])  # finger grows along Y-axis
#     for i in range(3):
#         transform = transform @ rot_x(joint_angles[i])
#         new_point = points[-1] + transform @ (lengths[i] * direction)
#         points.append(new_point)
#     return points

# # Inverse Kinematics function for a 3-joint finger
# def inverse_kinematics(target_pos, base_pos, lengths):
#     """
#     Calculate joint angles to reach target position
#     Using analytical IK solution for 3-joint manipulator
#     """
#     # Convert target to local coordinates
#     target_local = target_pos - base_pos
    
#     x, y, z = target_local
#     l1, l2, l3 = lengths
    
#     # Check if target is reachable
#     max_reach = sum(lengths)
#     if np.linalg.norm(target_local) > max_reach:
#         print("Target position out of reach")
#         return None
    
#     # Try analytical solution first
#     try:
#         # Calculate θ₁
#         theta1 = np.arctan2(x, z)
        
#         # Helper calculations
#         x_projected = np.sqrt(x**2 + z**2)
        
#         # Calculate θ₃
#         c3 = (x_projected**2 + y**2 - l1**2 - l2**2 - l3**2) / (2 * l2 * l3)
#         # Clamp to valid range to handle numerical errors
#         c3 = np.clip(c3, -1.0, 1.0)
#         s3 = np.sqrt(1 - c3**2)  # This assumes elbow up configuration
#         theta3 = np.arctan2(s3, c3)
        
#         # Calculate θ₂
#         k1 = l2 + l3 * c3
#         k2 = l3 * s3
#         theta2 = np.arctan2(y, x_projected) - np.arctan2(k2, k1)
        
#         # Return angles ensuring they're in valid ranges
#         return [
#             np.clip(theta1, -np.pi/2, np.pi/2),
#             np.clip(theta2, 0, np.pi),
#             np.clip(theta3, 0, np.pi)
#         ]
    
#     except Exception as e:
#         print(f"Analytical IK failed: {e}")
        
#         # Fall back to numerical optimization
#         def objective(angles):
#             # Calculate forward kinematics
#             points = compute_positions(base_pos, lengths, angles)
#             tip_pos = points[-1]
#             # Return squared distance to target
#             return np.sum((tip_pos - target_pos)**2)
        
#         # Initial guess and constraints
#         initial_guess = [0, 0, 0]
#         bounds = [(-np.pi/2, np.pi/2), (0, np.pi), (0, np.pi)]
        
#         # Run optimization
#         result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
#         if result.success:
#             return result.x
#         else:
#             print("Failed to find IK solution")
#             return None

# # Finger + Palm Config
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],  # Thumb
#     [3, 2.5, 2.0],  # Index
#     [3.5, 3.0, 2.5],  # Middle
#     [3.0, 2.5, 2.4],  # Ring
#     [2.2, 1.8, 1.5],  # Pinky
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers
# max_pull_cm = 2.5
# palm_spacing = 2.5  # space between finger bases on X-axis
# base_positions = [np.array([i * palm_spacing, 0.0, 0.0]) for i in range(num_fingers)]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Tendon routing parameters (pulley radii in cm)
# pulley_radii = [[0.9, 0.7, 0.5]] * num_fingers

# # Plot setup
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.25 + 0.05 * num_fingers)
# finger_lines = []
# tip_dots = []
# target_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro')
#     tip_dots.append(dot)
#     target, = ax.plot([], [], [], 'go', alpha=0.5)
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
# ax.set_title('Tendon-Driven Hand with Inverse Kinematics')

# # Sliders for forward kinematics
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.2 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Current target positions
# target_positions = [None] * num_fingers
# current_mode = "forward"  # "forward" or "inverse"

# # Update function for forward kinematics
# def update_forward(val):
#     global current_mode
#     if current_mode != "forward":
#         return
    
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
        
#         # Update targets
#         if target_positions[i] is not None:
#             target_dots[i].set_data([target_positions[i][0]], [target_positions[i][1]])
#             target_dots[i].set_3d_properties([target_positions[i][2]])
#         else:
#             target_dots[i].set_data([], [])
#             target_dots[i].set_3d_properties([])
            
#     fig.canvas.draw_idle()

# # Create individual button click handlers to avoid closure issues
# def create_button_handler(idx):
#     def handler(event):
#         set_target_and_solve(idx)
#     return handler

# # Function to set target and run inverse kinematics
# def set_target_and_solve(finger_idx):
#     global current_mode, target_positions
#     current_mode = "inverse"
    
#     # Get current tip position as target
#     pull = sliders[finger_idx].val
#     joint_angles = get_joint_angles_from_pull(
#         pull, max_pull_cm, max_angles_rad[finger_idx], ratios[finger_idx]
#     )
#     points = compute_positions(base_positions[finger_idx], lengths[finger_idx], joint_angles)
#     target_pos = points[-1]
#     target_positions[finger_idx] = target_pos
    
#     # Print initial state
#     print(f"\n{'='*60}")
#     print(f"  INVERSE KINEMATICS TEST FOR {finger_names[finger_idx]} FINGER")
#     print(f"{'='*60}")
#     print(f"\n1. INITIAL STATE:")
#     print(f"   - Current pull: {pull:.2f} cm")
#     print(f"   - Current joint angles (degrees): {np.degrees(joint_angles).round(2).tolist()}")
#     print(f"   - Current tip position: {target_pos.round(2).tolist()}")
    
#     # Show target
#     target_dots[finger_idx].set_data([target_pos[0]], [target_pos[1]])
#     target_dots[finger_idx].set_3d_properties([target_pos[2]])
    
#     # Move target slightly to demonstrate IK
#     adjusted_target = target_pos + np.array([0, 0.5, -0.5])
#     target_positions[finger_idx] = adjusted_target
#     target_dots[finger_idx].set_data([adjusted_target[0]], [adjusted_target[1]])
#     target_dots[finger_idx].set_3d_properties([adjusted_target[2]])
    
#     # Print target information
#     print(f"\n2. TARGET INFORMATION:")
#     print(f"   - Adjusted target position: {adjusted_target.round(2).tolist()}")
#     print(f"   - Target offset: [0.00, 0.50, -0.50]")
    
#     # Solve inverse kinematics
#     ik_angles = inverse_kinematics(adjusted_target, base_positions[finger_idx], lengths[finger_idx])
    
#     print(f"\n3. INVERSE KINEMATICS SOLUTION:")
    
#     if ik_angles is not None:
#         # Calculate required tendon pull
#         pull_required = get_pull_from_joint_angles(
#             ik_angles, max_pull_cm, max_angles_rad[finger_idx], ratios[finger_idx]
#         )
        
#         # Update slider
#         sliders[finger_idx].set_val(pull_required)
        
#         # Update display
#         points = compute_positions(base_positions[finger_idx], lengths[finger_idx], ik_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[finger_idx].set_data(xs, ys)
#         finger_lines[finger_idx].set_3d_properties(zs)
#         tip_dots[finger_idx].set_data([xs[-1]], [ys[-1]])
#         tip_dots[finger_idx].set_3d_properties([zs[-1]])
#         text_labels[finger_idx].set_position((xs[-1], ys[-1]))
#         text_labels[finger_idx].set_3d_properties(zs[-1], zdir='z')
#         text_labels[finger_idx].set_text(
#             f"{finger_names[finger_idx]}\nTarget: ({adjusted_target[0]:.1f}, {adjusted_target[1]:.1f}, {adjusted_target[2]:.1f})\n"
#             f"Actual: ({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})\n"
#             f"Pull: {pull_required:.2f}cm"
#         )
        
#         # Print IK solution details
#         print(f"   - Solution found!")
#         print(f"   - IK joint angles (degrees): {np.degrees(ik_angles).round(2).tolist()}")
#         print(f"   - Required tendon pull: {pull_required:.2f} cm")
#         print(f"   - Final tip position: {np.array([xs[-1], ys[-1], zs[-1]]).round(2).tolist()}")
#         print(f"   - Position error: {np.linalg.norm(np.array([xs[-1], ys[-1], zs[-1]]) - adjusted_target):.4f} cm")
        
#         print(f"\n4. JOINT POSITIONS:")
#         for i, pos in enumerate(points):
#             joint_type = "Base" if i == 0 else f"Joint {i-1}"
#             print(f"   - {joint_type}: {pos.round(2).tolist()}")
            
#         print(f"\n5. TENDON-SPACE MAPPING:")
#         print(f"   - Joint-to-tendon ratio: {ratios[finger_idx]}")
#         print(f"   - Max angles (degrees): {np.degrees(max_angles_rad[finger_idx]).tolist()}")
#         print(f"   - Pulley radii (cm): {pulley_radii[finger_idx]}")
#     else:
#         print("   - No IK solution found!")
    
#     print(f"\n{'='*60}\n")
    
#     current_mode = "forward"
#     fig.canvas.draw_idle()

# # Button setup for IK demonstration
# buttons = []
# for i in range(num_fingers):
#     ax_button = plt.axes([0.85, 0.2 - i*0.045, 0.1, 0.03])
#     button = Button(ax_button, 'IK Test')
#     button.on_clicked(create_button_handler(i))
#     buttons.append(button)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update_forward)

# update_forward(None)
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
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

# # Simplified relationship between tendon pull and joint angles for direct tendon routing
# def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
#     pull_ratio = min(pull_amount / max_pull, 1.0)
#     return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# # Inverse function: Convert joint angles to tendon pull (for direct tendon routing)
# def get_pull_from_joint_angles(joint_angles, max_pull, max_angles_rad, ratios):
#     # Find the limiting joint (the one requiring the most pull)
#     pull_ratios = [abs(angle) / (r * max_angle) 
#                   for angle, r, max_angle in zip(joint_angles, ratios, max_angles_rad)]
#     max_ratio = max(pull_ratios)
#     return min(max_ratio * max_pull, max_pull)

# # Compute joint positions in 3D (Forward kinematics)
# def compute_positions(base_pos, lengths, joint_angles):
#     points = [base_pos]
#     transform = np.eye(3)
#     direction = np.array([0.0, 1.0, 0.0])  # finger grows along Y-axis
#     for i in range(3):
#         transform = transform @ rot_x(joint_angles[i])
#         new_point = points[-1] + transform @ (lengths[i] * direction)
#         points.append(new_point)
#     return points

# # Inverse Kinematics function for a 3-joint finger
# def inverse_kinematics(target_pos, base_pos, lengths):
#     """
#     Calculate joint angles to reach target position
#     Using analytical IK solution for 3-joint manipulator
#     """
#     # Convert target to local coordinates
#     target_local = target_pos - base_pos
    
#     x, y, z = target_local
#     l1, l2, l3 = lengths
    
#     # Check if target is reachable
#     max_reach = sum(lengths)
#     if np.linalg.norm(target_local) > max_reach:
#         print("Target position out of reach")
#         return None
    
#     # Try analytical solution first
#     try:
#         # Calculate θ₁
#         theta1 = np.arctan2(x, z)
        
#         # Helper calculations
#         x_projected = np.sqrt(x**2 + z**2)
        
#         # Calculate θ₃
#         c3 = (x_projected**2 + y**2 - l1**2 - l2**2 - l3**2) / (2 * l2 * l3)
#         # Clamp to valid range to handle numerical errors
#         c3 = np.clip(c3, -1.0, 1.0)
#         s3 = np.sqrt(1 - c3**2)  # This assumes elbow up configuration
#         theta3 = np.arctan2(s3, c3)
        
#         # Calculate θ₂
#         k1 = l2 + l3 * c3
#         k2 = l3 * s3
#         theta2 = np.arctan2(y, x_projected) - np.arctan2(k2, k1)
        
#         # Return angles ensuring they're in valid ranges
#         return [
#             np.clip(theta1, -np.pi/2, np.pi/2),
#             np.clip(theta2, 0, np.pi),
#             np.clip(theta3, 0, np.pi)
#         ]
    
#     except Exception as e:
#         print(f"Analytical IK failed: {e}")
        
#         # Fall back to numerical optimization
#         def objective(angles):
#             # Calculate forward kinematics
#             points = compute_positions(base_pos, lengths, angles)
#             tip_pos = points[-1]
#             # Return squared distance to target
#             return np.sum((tip_pos - target_pos)**2)
        
#         # Initial guess and constraints
#         initial_guess = [0, 0, 0]
#         bounds = [(-np.pi/2, np.pi/2), (0, np.pi), (0, np.pi)]
        
#         # Run optimization
#         result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
#         if result.success:
#             return result.x
#         else:
#             print("Failed to find IK solution")
#             return None

# # Finger + Palm Config
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],  # Thumb
#     [3, 2.5, 2.0],  # Index
#     [3.5, 3.0, 2.5],  # Middle
#     [3.0, 2.5, 2.4],  # Ring
#     [2.2, 1.8, 1.5],  # Pinky
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers  # Ratio of joint angle to tendon pull
# max_pull_cm = 2.5  # Maximum tendon pull distance
# palm_spacing = 2.5  # space between finger bases on X-axis
# base_positions = [np.array([i * palm_spacing, 0.0, 0.0]) for i in range(num_fingers)]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Plot setup
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.25 + 0.05 * num_fingers)
# finger_lines = []
# tip_dots = []
# target_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro')
#     tip_dots.append(dot)
#     target, = ax.plot([], [], [], 'go', alpha=0.5)
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
# ax.set_title('Direct Tendon-Driven Hand with Inverse Kinematics')

# # Sliders for forward kinematics
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.2 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Current target positions
# target_positions = [None] * num_fingers
# current_mode = "forward"  # "forward" or "inverse"

# # Update function for forward kinematics
# def update_forward(val):
#     global current_mode
#     if current_mode != "forward":
#         return
    
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
        
#         # Update targets
#         if target_positions[i] is not None:
#             target_dots[i].set_data([target_positions[i][0]], [target_positions[i][1]])
#             target_dots[i].set_3d_properties([target_positions[i][2]])
#         else:
#             target_dots[i].set_data([], [])
#             target_dots[i].set_3d_properties([])
            
#     fig.canvas.draw_idle()

# # Create individual button click handlers
# def create_button_handler(idx):
#     def handler(event):
#         set_target_and_solve(idx)
#     return handler

# # Function to set target and run inverse kinematics
# def set_target_and_solve(finger_idx):
#     global current_mode, target_positions
#     current_mode = "inverse"
    
#     # Get current tip position as target
#     pull = sliders[finger_idx].val
#     joint_angles = get_joint_angles_from_pull(
#         pull, max_pull_cm, max_angles_rad[finger_idx], ratios[finger_idx]
#     )
#     points = compute_positions(base_positions[finger_idx], lengths[finger_idx], joint_angles)
#     target_pos = points[-1]
#     target_positions[finger_idx] = target_pos
    
#     # Print initial state
#     print(f"\n{'='*60}")
#     print(f"  INVERSE KINEMATICS TEST FOR {finger_names[finger_idx]} FINGER")
#     print(f"{'='*60}")
#     print(f"\n1. INITIAL STATE:")
#     print(f"   - Current pull: {pull:.2f} cm")
#     print(f"   - Current joint angles (degrees): {np.degrees(joint_angles).round(2).tolist()}")
#     print(f"   - Current tip position: {target_pos.round(2).tolist()}")
    
#     # Show target
#     target_dots[finger_idx].set_data([target_pos[0]], [target_pos[1]])
#     target_dots[finger_idx].set_3d_properties([target_pos[2]])
    
#     # Move target slightly to demonstrate IK
#     adjusted_target = target_pos + np.array([0, 0.5, -0.5])
#     target_positions[finger_idx] = adjusted_target
#     target_dots[finger_idx].set_data([adjusted_target[0]], [adjusted_target[1]])
#     target_dots[finger_idx].set_3d_properties([adjusted_target[2]])
    
#     # Print target information
#     print(f"\n2. TARGET INFORMATION:")
#     print(f"   - Adjusted target position: {adjusted_target.round(2).tolist()}")
#     print(f"   - Target offset: [0.00, 0.50, -0.50]")
    
#     # Solve inverse kinematics
#     ik_angles = inverse_kinematics(adjusted_target, base_positions[finger_idx], lengths[finger_idx])
    
#     print(f"\n3. INVERSE KINEMATICS SOLUTION:")
    
#     if ik_angles is not None:
#         # Calculate required tendon pull - direct relationship for direct tendon routing
#         pull_required = get_pull_from_joint_angles(
#             ik_angles, max_pull_cm, max_angles_rad[finger_idx], ratios[finger_idx]
#         )
        
#         # Update slider
#         sliders[finger_idx].set_val(pull_required)
        
#         # Update display
#         points = compute_positions(base_positions[finger_idx], lengths[finger_idx], ik_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[finger_idx].set_data(xs, ys)
#         finger_lines[finger_idx].set_3d_properties(zs)
#         tip_dots[finger_idx].set_data([xs[-1]], [ys[-1]])
#         tip_dots[finger_idx].set_3d_properties([zs[-1]])
#         text_labels[finger_idx].set_position((xs[-1], ys[-1]))
#         text_labels[finger_idx].set_3d_properties(zs[-1], zdir='z')
#         text_labels[finger_idx].set_text(
#             f"{finger_names[finger_idx]}\nTarget: ({adjusted_target[0]:.1f}, {adjusted_target[1]:.1f}, {adjusted_target[2]:.1f})\n"
#             f"Actual: ({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})\n"
#             f"Pull: {pull_required:.2f}cm"
#         )
        
#         # Print IK solution details
#         print(f"   - Solution found!")
#         print(f"   - IK joint angles (degrees): {np.degrees(ik_angles).round(2).tolist()}")
#         print(f"   - Required tendon pull: {pull_required:.2f} cm")
#         print(f"   - Final tip position: {np.array([xs[-1], ys[-1], zs[-1]]).round(2).tolist()}")
#         print(f"   - Position error: {np.linalg.norm(np.array([xs[-1], ys[-1], zs[-1]]) - adjusted_target):.4f} cm")
        
#         print(f"\n4. JOINT POSITIONS:")
#         for i, pos in enumerate(points):
#             joint_type = "Base" if i == 0 else f"Joint {i-1}"
#             print(f"   - {joint_type}: {pos.round(2).tolist()}")
            
#         print(f"\n5. DIRECT TENDON MAPPING:")
#         print(f"   - Joint-to-tendon ratio: {ratios[finger_idx]}")
#         print(f"   - Max angles (degrees): {np.degrees(max_angles_rad[finger_idx]).tolist()}")
#         print(f"   - Direct tendon pull-to-angle relationship used (no pulleys)")
#     else:
#         print("   - No IK solution found!")
    
#     print(f"\n{'='*60}\n")
    
#     current_mode = "forward"
#     fig.canvas.draw_idle()

# # Button setup for IK demonstration
# buttons = []
# for i in range(num_fingers):
#     ax_button = plt.axes([0.85, 0.2 - i*0.045, 0.1, 0.03])
#     button = Button(ax_button, 'IK Test')
#     button.on_clicked(create_button_handler(i))
#     buttons.append(button)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update_forward)

# update_forward(None)
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
# from mpl_toolkits.mplot3d import Axes3D

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
#     direction = np.array([0.0, 1.0, 0.0])
#     for i in range(3):
#         transform = transform @ rot_x(joint_angles[i])
#         new_point = points[-1] + transform @ (lengths[i] * direction)
#         points.append(new_point)
#     return points

# # Inverse Kinematics using Grid Search (returns pull and joint angles)
# def inverse_kinematics(fingertip_target, base_pos, lengths, max_pull, max_angles_rad, ratios):
#     best_error = float('inf')
#     best_pull = 0.0
#     best_angles = [0.0, 0.0, 0.0]
#     for pull in np.linspace(0.0, max_pull, 100):
#         angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
#         points = compute_positions(base_pos, lengths, angles)
#         tip = points[-1]
#         error = np.linalg.norm(tip - fingertip_target)
#         if error < best_error:
#             best_error = error
#             best_pull = pull
#             best_angles = angles
#     return best_pull, best_angles

# # Configuration
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],
#     [3, 2.5, 2.0],
#     [3.5, 3.0, 2.5],
#     [3.0, 2.5, 2.4],
#     [2.2, 1.8, 1.5],
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers
# max_pull_cm = 2.5
# palm_spacing = 2.5
# base_positions = [np.array([i * palm_spacing, 0.0, 0.0]) for i in range(num_fingers)]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Plot setup
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.3)
# finger_lines = []
# tip_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro')
#     tip_dots.append(dot)
#     label = ax.text(0, 0, 0, '', fontsize=8)
#     text_labels.append(label)

# # Axis limits
# ax.set_xlim(-2, num_fingers * palm_spacing + 1)
# ax.set_ylim(0, 15)
# ax.set_zlim(-8, 1)
# ax.set_xlabel('X (Palm width)')
# ax.set_ylabel('Y (Finger length)')
# ax.set_zlabel('Z (Bend)')
# ax.set_title('Tendon-Driven Hand (3D Simulation)')

# # Sliders
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Update function
# def update(val):
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
#     fig.canvas.draw_idle()

# # Button for inverse kinematics
# ax_button = plt.axes([0.42, 0.01, 0.2, 0.05])
# button = Button(ax_button, 'Apply Inverse Kinematics')

# def apply_inverse_kinematics(event):
#     print("\nRunning Inverse Kinematics:")
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         angles = get_joint_angles_from_pull(pull, max_pull_cm, max_angles_rad[i], ratios[i])
#         points = compute_positions(base_positions[i], lengths[i], angles)
#         tip = points[-1]
#         inv_pull, joint_angles = inverse_kinematics(tip, base_positions[i], lengths[i], max_pull_cm, max_angles_rad[i], ratios[i])
#         sliders[i].set_val(inv_pull)
#         print(f"{finger_names[i]} ▶ Pull: {inv_pull:.2f} cm | Tip: {tip} | Angles (deg): {[f'{np.degrees(a):.1f}' for a in joint_angles]}")
#     update(None)

# button.on_clicked(apply_inverse_kinematics)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update)

# update(None)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
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
#     direction = np.array([0.0, 1.0, 0.0])
#     for i in range(3):
#         transform = transform @ rot_x(joint_angles[i])
#         new_point = points[-1] + transform @ (lengths[i] * direction)
#         points.append(new_point)
#     return points

# # Numerical Jacobian (optional for advanced IK)
# def numerical_jacobian(base_pos, lengths, pull, max_pull, max_angles_rad, ratios, delta=1e-4):
#     f0 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios))[-1]
#     f1 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull + delta, max_pull, max_angles_rad, ratios))[-1]
#     return (f1 - f0) / delta

# # Jacobian-based Inverse Kinematics using optimization
# def inverse_kinematics(fingertip_target, base_pos, lengths, max_pull, max_angles_rad, ratios):
#     def forward_kinematics(pull):
#         angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
#         points = compute_positions(base_pos, lengths, angles)
#         return points[-1]

#     def objective(pull):
#         tip = forward_kinematics(pull[0])
#         return np.linalg.norm(tip - fingertip_target)

#     result = minimize(objective, x0=[0.5 * max_pull], bounds=[(0.0, max_pull)], method='L-BFGS-B')
#     best_pull = result.x[0]
#     best_angles = get_joint_angles_from_pull(best_pull, max_pull, max_angles_rad, ratios)
#     return best_pull, best_angles

# # Configuration
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],
#     [3, 2.5, 2.0],
#     [3.5, 3.0, 2.5],
#     [3.0, 2.5, 2.4],
#     [2.2, 1.8, 1.5],
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers
# max_pull_cm = 2.5
# palm_spacing = 2.5
# base_positions = [np.array([i * palm_spacing, 0.0, 0.0]) for i in range(num_fingers)]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Plot setup
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.3)
# finger_lines = []
# tip_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro')
#     tip_dots.append(dot)
#     label = ax.text(0, 0, 0, '', fontsize=8)
#     text_labels.append(label)

# # Axis limits
# ax.set_xlim(-2, num_fingers * palm_spacing + 1)
# ax.set_ylim(0, 15)
# ax.set_zlim(-8, 1)
# ax.set_xlabel('X (Palm width)')
# ax.set_ylabel('Y (Finger length)')
# ax.set_zlabel('Z (Bend)')
# ax.set_title('Tendon-Driven Hand (3D Simulation)')

# # Sliders
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Update function
# def update(val):
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles)
#         xs, ys, zs = zip(*points)
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
#     fig.canvas.draw_idle()

# # Button for inverse kinematics
# ax_button = plt.axes([0.42, 0.01, 0.2, 0.05])
# button = Button(ax_button, 'Apply Inverse Kinematics')

# def apply_inverse_kinematics(event):
#     print("\nRunning Inverse Kinematics:")
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         angles = get_joint_angles_from_pull(pull, max_pull_cm, max_angles_rad[i], ratios[i])
#         points = compute_positions(base_positions[i], lengths[i], angles)
#         tip = points[-1]
#         inv_pull, joint_angles = inverse_kinematics(tip, base_positions[i], lengths[i], max_pull_cm, max_angles_rad[i], ratios[i])
#         sliders[i].set_val(inv_pull)
#         joint_deg = [np.degrees(a) for a in joint_angles]
#         print(f"{finger_names[i]} ▶ Pull: {inv_pull:.2f} cm | Tip: {tip} | Angles (deg): {[f'{d:.1f}' for d in joint_deg]}")
#         # Optional Jacobian print
#         jac = numerical_jacobian(base_positions[i], lengths[i], inv_pull, max_pull_cm, max_angles_rad[i], ratios[i])
#         print(f"    Numerical Jacobian: dx={jac[0]:.4f}, dy={jac[1]:.4f}, dz={jac[2]:.4f}")
#     update(None)

# button.on_clicked(apply_inverse_kinematics)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update)

# update(None)
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.optimize import minimize

# # Rotation matrices
# def rot_x(theta):
#     c, s = np.cos(theta), np.sin(theta)
#     return np.array([
#         [1, 0,  0],
#         [0, c, -s],
#         [0, s,  c]
#     ])

# def rot_y(theta):
#     c, s = np.cos(theta), np.sin(theta)
#     return np.array([
#         [ c, 0, s],
#         [ 0, 1, 0],
#         [-s, 0, c]
#     ])

# def rot_z(theta):
#     c, s = np.cos(theta), np.sin(theta)
#     return np.array([
#         [c, -s, 0],
#         [s,  c, 0],
#         [0,  0, 1]
#     ])

# # Convert tendon pull to joint angles
# def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
#     pull_ratio = min(pull_amount / max_pull, 1.0)
#     return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# # Compute joint positions in 3D
# def compute_positions(base_pos, lengths, joint_angles, is_thumb=False):
#     points = [base_pos]
#     transform = np.eye(3)

#     if is_thumb:
#         # Thumb starts with segment pointing along +Y (same as other fingers),
#         # but we rotate each joint around Z to curl upward (instead of X like other fingers)
#         direction = np.array([0.0, 1.0, 0.0])  # Segment extends along Y
#         for i in range(3):
#             transform = transform @ rot_z(joint_angles[i])  # Curl upward about Z
#             new_point = points[-1] + transform @ (lengths[i] * direction)
#             points.append(new_point)
#     else:
#         # Other fingers bend around X-axis
#         direction = np.array([0.0, 1.0, 0.0])
#         for i in range(3):
#             transform = transform @ rot_x(joint_angles[i])
#             new_point = points[-1] + transform @ (lengths[i] * direction)
#             points.append(new_point)

#     return points

# # Numerical Jacobian
# def numerical_jacobian(base_pos, lengths, pull, max_pull, max_angles_rad, ratios, is_thumb=False, delta=1e-4):
#     f0 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios), is_thumb)[-1]
#     f1 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull + delta, max_pull, max_angles_rad, ratios), is_thumb)[-1]
#     return (f1 - f0) / delta

# # Inverse Kinematics
# def inverse_kinematics(fingertip_target, base_pos, lengths, max_pull, max_angles_rad, ratios, is_thumb=False):
#     def forward_kinematics(pull):
#         angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
#         points = compute_positions(base_pos, lengths, angles, is_thumb)
#         return points[-1]

#     def objective(pull):
#         tip = forward_kinematics(pull[0])
#         return np.linalg.norm(tip - fingertip_target)

#     result = minimize(objective, x0=[0.5 * max_pull], bounds=[(0.0, max_pull)], method='L-BFGS-B')
#     best_pull = result.x[0]
#     best_angles = get_joint_angles_from_pull(best_pull, max_pull, max_angles_rad, ratios)
#     return best_pull, best_angles

# # Configuration
# finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
# num_fingers = 5
# lengths = [
#     [3.5, 2.5, 1.5],
#     [3, 2.5, 2.0],
#     [3.5, 3.0, 2.5],
#     [3.0, 2.5, 2.4],
#     [2.2, 1.8, 1.5],
# ]
# max_angles_deg = [[55, 90, 90]] * num_fingers
# ratios = [[1.0, 0.7, 0.5]] * num_fingers
# max_pull_cm = 2.5
# palm_spacing = 2.5
# base_positions = [
#     np.array([1.0, -7.0, 0.0]),  # Thumb base
#     *[np.array([i * palm_spacing, 0.0, 0.0]) for i in range(1, num_fingers)]
# ]
# max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# # Plot setup
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# plt.subplots_adjust(bottom=0.3)
# finger_lines = []
# tip_dots = []
# text_labels = []

# # Create visuals for fingers
# for _ in range(num_fingers):
#     line, = ax.plot([], [], [], 'o-', lw=3)
#     finger_lines.append(line)
#     dot, = ax.plot([], [], [], 'ro')
#     tip_dots.append(dot)
#     label = ax.text(0, 0, 0, '', fontsize=8)
#     text_labels.append(label)

# # Axis limits
# ax.set_xlim(-10, num_fingers * palm_spacing + 1)
# ax.set_ylim(-10, 15)
# ax.set_zlim(-8, 1)
# ax.set_xlabel('X (Palm width)')
# ax.set_ylabel('Y (Finger length)')
# ax.set_zlabel('Z (Bend)')
# ax.set_title('Tendon-Driven Hand (3D Simulation)')

# # Sliders
# sliders = []
# for i in range(num_fingers):
#     ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
#     slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
#     sliders.append(slider)

# # Function to calculate current finger positions
# def get_current_positions():
#     positions = []
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles, is_thumb=(i == 0))
#         positions.append(points[-1])  # Store the endpoint position
#     return positions

# # Update function
# def update(val):
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         joint_angles = get_joint_angles_from_pull(
#             pull, max_pull_cm, max_angles_rad[i], ratios[i]
#         )
#         points = compute_positions(base_positions[i], lengths[i], joint_angles, is_thumb=(i == 0))
#         xs, ys, zs = zip(*points)
#         finger_lines[i].set_data(xs, ys)
#         finger_lines[i].set_3d_properties(zs)
#         tip_dots[i].set_data([xs[-1]], [ys[-1]])
#         tip_dots[i].set_3d_properties([zs[-1]])
#         text_labels[i].set_position((xs[-1], ys[-1]))
#         text_labels[i].set_3d_properties(zs[-1], zdir='z')
#         text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
#     fig.canvas.draw_idle()

# # Button for inverse kinematics
# ax_button = plt.axes([0.25, 0.01, 0.2, 0.05])
# button = Button(ax_button, 'Apply Inverse Kinematics')

# def apply_inverse_kinematics(event):
#     print("\nRunning Inverse Kinematics:")
#     for i in range(num_fingers):
#         pull = sliders[i].val
#         angles = get_joint_angles_from_pull(pull, max_pull_cm, max_angles_rad[i], ratios[i])
#         points = compute_positions(base_positions[i], lengths[i], angles, is_thumb=(i == 0))
#         tip = points[-1]
#         inv_pull, joint_angles = inverse_kinematics(tip, base_positions[i], lengths[i], max_pull_cm, max_angles_rad[i], ratios[i], is_thumb=(i == 0))
#         sliders[i].set_val(inv_pull)
#         joint_deg = [np.degrees(a) for a in joint_angles]
#         print(f"{finger_names[i]} ▶ Pull: {inv_pull:.2f} cm | Tip: {tip} | Angles (deg): {[f'{d:.1f}' for d in joint_deg]}")
#         jac = numerical_jacobian(base_positions[i], lengths[i], inv_pull, max_pull_cm, max_angles_rad[i], ratios[i], is_thumb=(i == 0))
#         print(f"    Numerical Jacobian: dx={jac[0]:.4f}, dy={jac[1]:.4f}, dz={jac[2]:.4f}")
#     update(None)

# # New print button and function
# ax_print_button = plt.axes([0.55, 0.01, 0.2, 0.05])
# print_button = Button(ax_print_button, 'Print Finger Positions')

# def print_finger_positions(event):
#     positions = get_current_positions()
#     print("\nCurrent Finger Endpoint Positions:")
#     print("----------------------------------")
#     for i, pos in enumerate(positions):
#         print(f"{finger_names[i]}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
#     print("----------------------------------")

# print_button.on_clicked(print_finger_positions)
# button.on_clicked(apply_inverse_kinematics)

# # Connect sliders
# for slider in sliders:
#     slider.on_changed(update)

# update(None)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import csv
import os
from datetime import datetime

# Rotation matrices
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# Convert tendon pull to joint angles
def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
    pull_ratio = min(pull_amount / max_pull, 1.0)
    return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

# Compute joint positions in 3D
def compute_positions(base_pos, lengths, joint_angles, is_thumb=False):
    points = [base_pos]
    transform = np.eye(3)

    if is_thumb:
        # Thumb starts with segment pointing along +Y (same as other fingers),
        # but we rotate each joint around Z to curl upward (instead of X like other fingers)
        direction = np.array([0.0, 1.0, 0.0])  # Segment extends along Y
        for i in range(3):
            transform = transform @ rot_z(joint_angles[i])  # Curl upward about Z
            new_point = points[-1] + transform @ (lengths[i] * direction)
            points.append(new_point)
    else:
        # Other fingers bend around X-axis
        direction = np.array([0.0, 1.0, 0.0])
        for i in range(3):
            transform = transform @ rot_x(joint_angles[i])
            new_point = points[-1] + transform @ (lengths[i] * direction)
            points.append(new_point)

    return points

# Numerical Jacobian
def numerical_jacobian(base_pos, lengths, pull, max_pull, max_angles_rad, ratios, is_thumb=False, delta=1e-4):
    f0 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios), is_thumb)[-1]
    f1 = compute_positions(base_pos, lengths, get_joint_angles_from_pull(pull + delta, max_pull, max_angles_rad, ratios), is_thumb)[-1]
    return (f1 - f0) / delta

# Inverse Kinematics
def inverse_kinematics(fingertip_target, base_pos, lengths, max_pull, max_angles_rad, ratios, is_thumb=False):
    def forward_kinematics(pull):
        angles = get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
        points = compute_positions(base_pos, lengths, angles, is_thumb)
        return points[-1]

    def objective(pull):
        tip = forward_kinematics(pull[0])
        return np.linalg.norm(tip - fingertip_target)

    result = minimize(objective, x0=[0.5 * max_pull], bounds=[(0.0, max_pull)], method='L-BFGS-B')
    best_pull = result.x[0]
    best_angles = get_joint_angles_from_pull(best_pull, max_pull, max_angles_rad, ratios)
    return best_pull, best_angles

# Configuration
finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
num_fingers = 5
lengths = [
    [3.5, 2.5, 1.5],
    [3, 2.5, 2.0],
    [3.5, 3.0, 2.5],
    [3.0, 2.5, 2.4],
    [2.2, 1.8, 1.5],
]
max_angles_deg = [[55, 90, 90]] * num_fingers
ratios = [[1.0, 0.7, 0.5]] * num_fingers
max_pull_cm = 2.5
palm_spacing = 2.5
base_positions = [
    np.array([1.0, -7.0, 0.0]),  # Thumb base
    *[np.array([i * palm_spacing, 0.0, 0.0]) for i in range(1, num_fingers)]
]
max_angles_rad = [np.radians(angles) for angles in max_angles_deg]

# CSV file setup
csv_filename = "finger_positions.csv"
# Check if file exists and create with headers if it doesn't
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Timestamp'] + [f"{name}_{coord}" for name in finger_names for coord in ['X', 'Y', 'Z']]
        writer.writerow(headers)
    print(f"Created new CSV file: {csv_filename}")
else:
    print(f"Will append to existing CSV file: {csv_filename}")

# Plot setup
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)
finger_lines = []
tip_dots = []
text_labels = []

# Create visuals for fingers
for _ in range(num_fingers):
    line, = ax.plot([], [], [], 'o-', lw=3)
    finger_lines.append(line)
    dot, = ax.plot([], [], [], 'ro')
    tip_dots.append(dot)
    label = ax.text(0, 0, 0, '', fontsize=8)
    text_labels.append(label)

# Axis limits
ax.set_xlim(-10, num_fingers * palm_spacing + 1)
ax.set_ylim(-10, 15)
ax.set_zlim(-8, 1)
ax.set_xlabel('X (Palm width)')
ax.set_ylabel('Y (Finger length)')
ax.set_zlabel('Z (Bend)')
ax.set_title('Tendon-Driven Hand (3D Simulation)')

# Sliders
sliders = []
for i in range(num_fingers):
    ax_slider = plt.axes([0.15, 0.25 - i*0.045, 0.65, 0.03])
    slider = Slider(ax_slider, f'{finger_names[i]} Pull (cm)', 0.0, max_pull_cm, valinit=0.0)
    sliders.append(slider)

# Function to calculate current finger positions
def get_current_positions():
    positions = []
    for i in range(num_fingers):
        pull = sliders[i].val
        joint_angles = get_joint_angles_from_pull(
            pull, max_pull_cm, max_angles_rad[i], ratios[i]
        )
        points = compute_positions(base_positions[i], lengths[i], joint_angles, is_thumb=(i == 0))
        positions.append(points[-1])  # Store the endpoint position
    return positions

# Function to save positions to CSV
def save_positions_to_csv(positions):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Flatten the positions data
    row_data = [timestamp]
    for pos in positions:
        row_data.extend([pos[0], pos[1], pos[2]])
    
    # Append to CSV
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)
    
    print(f"Saved positions to {csv_filename}")

# Update function
def update(val):
    for i in range(num_fingers):
        pull = sliders[i].val
        joint_angles = get_joint_angles_from_pull(
            pull, max_pull_cm, max_angles_rad[i], ratios[i]
        )
        points = compute_positions(base_positions[i], lengths[i], joint_angles, is_thumb=(i == 0))
        xs, ys, zs = zip(*points)
        finger_lines[i].set_data(xs, ys)
        finger_lines[i].set_3d_properties(zs)
        tip_dots[i].set_data([xs[-1]], [ys[-1]])
        tip_dots[i].set_3d_properties([zs[-1]])
        text_labels[i].set_position((xs[-1], ys[-1]))
        text_labels[i].set_3d_properties(zs[-1], zdir='z')
        text_labels[i].set_text(f"{finger_names[i]}\n({xs[-1]:.1f}, {ys[-1]:.1f}, {zs[-1]:.1f})")
    fig.canvas.draw_idle()

# Button for inverse kinematics
ax_button = plt.axes([0.15, 0.01, 0.2, 0.05])
button = Button(ax_button, 'Apply Inverse Kinematics')

def apply_inverse_kinematics(event):
    print("\nRunning Inverse Kinematics:")
    for i in range(num_fingers):
        pull = sliders[i].val
        angles = get_joint_angles_from_pull(pull, max_pull_cm, max_angles_rad[i], ratios[i])
        points = compute_positions(base_positions[i], lengths[i], angles, is_thumb=(i == 0))
        tip = points[-1]
        inv_pull, joint_angles = inverse_kinematics(tip, base_positions[i], lengths[i], max_pull_cm, max_angles_rad[i], ratios[i], is_thumb=(i == 0))
        sliders[i].set_val(inv_pull)
        joint_deg = [np.degrees(a) for a in joint_angles]
        print(f"{finger_names[i]} ▶ Pull: {inv_pull:.2f} cm | Tip: {tip} | Angles (deg): {[f'{d:.1f}' for d in joint_deg]}")
        jac = numerical_jacobian(base_positions[i], lengths[i], inv_pull, max_pull_cm, max_angles_rad[i], ratios[i], is_thumb=(i == 0))
        print(f"    Numerical Jacobian: dx={jac[0]:.4f}, dy={jac[1]:.4f}, dz={jac[2]:.4f}")
    update(None)

# New print and save button and function
ax_print_button = plt.axes([0.4, 0.01, 0.2, 0.05])
print_button = Button(ax_print_button, 'Print & Save Positions')

def print_and_save_finger_positions(event):
    positions = get_current_positions()
    
    # Print to console
    print("\nCurrent Finger Endpoint Positions:")
    print("----------------------------------")
    for i, pos in enumerate(positions):
        print(f"{finger_names[i]}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print("----------------------------------")
    
    # Save to CSV
    save_positions_to_csv(positions)

print_button.on_clicked(print_and_save_finger_positions)
button.on_clicked(apply_inverse_kinematics)

# Add a clear CSV button
ax_clear_button = plt.axes([0.65, 0.01, 0.2, 0.05])
clear_button = Button(ax_clear_button, 'Clear CSV File')

def clear_csv_file(event):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Timestamp'] + [f"{name}_{coord}" for name in finger_names for coord in ['X', 'Y', 'Z']]
        writer.writerow(headers)
    print(f"Cleared CSV file: {csv_filename}")

clear_button.on_clicked(clear_csv_file)

# Connect sliders
for slider in sliders:
    slider.on_changed(update)

update(None)
plt.show()