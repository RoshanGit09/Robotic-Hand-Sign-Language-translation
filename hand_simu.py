# import pybullet as p
# import pybullet_data
# import time

# # Connect to the PyBullet physics server
# physicsClient = p.connect(p.GUI)

# # Set the additional search path for PyBullet data
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load the plane for reference
# planeId = p.loadURDF("plane.urdf")

# # Load the URDF file of the hand
# urdf_path = "ability_hand_right_large.urdf"
# robotId = p.loadURDF(urdf_path, useFixedBase=True)

# # Set gravity for realistic physics simulation
# p.setGravity(0, 0, -9.81)

# # Get the number of joints in the robot
# num_joints = p.getNumJoints(robotId)

# # Create a dictionary to store joint indices by name
# joint_indices = {}
# for i in range(num_joints):
#     joint_info = p.getJointInfo(robotId, i)
#     joint_name = joint_info[1].decode('utf-8')  # Decode byte string to normal string
#     joint_indices[joint_name] = i
#     print(f"Joint {i}: {joint_name}")  # Print joint name for reference

# # Create sliders for controlling joint positions dynamically
# sliders = {}
# for joint_name, joint_index in joint_indices.items():
#     sliders[joint_name] = p.addUserDebugParameter(joint_name, -2.0, 2.0, 0.0)

# # Set the control parameters
# control_mode = p.POSITION_CONTROL  # Use position control
# max_force = 100  # Maximum force to apply (in N)

# # Run the simulation loop
# while True:
#     # Update the target positions from sliders
#     for joint_name, joint_index in joint_indices.items():
#         target_pos = p.readUserDebugParameter(sliders[joint_name])
#         p.setJointMotorControl2(robotId, joint_index, control_mode, targetPosition=target_pos, force=max_force)

#     # Step the simulation
#     p.stepSimulation()
#     time.sleep(1 / 240)  # PyBullet runs at 240Hz by default

# import pybullet as p
# import pybullet_data
# import time
# import tkinter as tk

# # Connect to PyBullet physics server
# physicsClient = p.connect(p.GUI)

# # Load environment and URDF
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = p.loadURDF("plane.urdf")
# robotId = p.loadURDF("shadow_hand.urdf", useFixedBase=True)

# # Disable gravity as requested
# p.setGravity(0, 0, 0)

# # Gather joint information
# num_joints = p.getNumJoints(robotId)
# print(num_joints)
# joint_indices = {}
# for i in range(num_joints):
#     joint_info = p.getJointInfo(robotId, i)
#     joint_name = joint_info[1].decode('utf-8')
#     joint_indices[joint_name] = i

# # Servo-like angle distribution for realistic motion
# finger_joints = {
#     "thumb": ["thumb_joint2", "thumb_joint3", "thumb_joint4"],
#     "index_finger": ["index_finger_join2", "index_finger_joint3", "index_finger_joint4"],
#     "middle_finger": ["middle_finger_joint2", "middle_finger_joint3", "middle_finger_joint4"],
#     "ring_finger": ["ring_finger_joint2", "ring_finger_joint3", "ring_finger_joint4"],
#     "little_finger": ["little_finger_joint2", "little_finger_joint3", "little_finger_joint4"]
# }

# def get_fingertip_position(finger_name):
#     """Returns the world position of the fingertip (last joint) of the specified finger."""
#     joints = finger_joints.get(finger_name, [])
#     if not joints:
#         print(f"No joints found for finger: {finger_name}")
#         return None

#     # Get the last joint name and index
#     last_joint_name = joints[-1]
#     joint_index = joint_indices.get(last_joint_name)

#     if joint_index is not None:
#         link_state = p.getLinkState(robotId, joint_index)
#         position = link_state[4]  # World position of the link frame
#         print(f"{finger_name.capitalize()} fingertip position: {position}")
#         return position
#     else:
#         print(f"Invalid joint name: {last_joint_name}")
#         return None


# def set_finger_angle(finger, angle):
#     for finger, angle_var in angle_vars.items():
#             target_angle = float(angle_var.get())
#             target_angle=target_angle/37.5
#             distributed_angle = target_angle / 3 

#             for joint_name in finger_joints[finger]:
#                 if joint_name in joint_indices:
#                     p.setJointMotorControl2(
#                         robotId, joint_indices[joint_name],
#                         p.POSITION_CONTROL, targetPosition=distributed_angle
#                     )

# root = tk.Tk()
# root.title("Control Shadow Hand")

# angle_vars = {}
# for finger in finger_joints.keys():
#     frame = tk.Frame(root)
#     frame.pack(pady=5)

#     label = tk.Label(frame, text=f"{finger.capitalize()} Rotation:")
#     label.pack(side=tk.LEFT)

#     angle_var = tk.StringVar(value="0")
#     entry = tk.Entry(frame, textvariable=angle_var)
#     entry.pack(side=tk.LEFT)

#     angle_vars[finger] = angle_var


# def apply_angles():
#     for finger, angle_var in angle_vars.items():
#         try:
#             target_angle = float(angle_var.get())
#             set_finger_angle(finger, target_angle)

#             # Display fingertip position
#             get_fingertip_position(finger)
#         except ValueError:
#             print(f"Invalid input for {finger}")


# apply_button = tk.Button(root, text="Apply", command=apply_angles)
# apply_button.pack(pady=10)


# def simulation_loop():
#     p.stepSimulation()
#     root.after(10, simulation_loop)

# simulation_loop()
# root.mainloop()


import pybullet as p
import pybullet_data
import time
import tkinter as tk

# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("shadow_hand.urdf", useFixedBase=True)
p.setGravity(0, 0, 0)

# Get joint info
num_joints = p.getNumJoints(robotId)
print("Number of joints:", num_joints)

joint_indices = {}
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_indices[joint_name] = i

# Finger joint map
finger_joints = {
    "thumb": ["thumb_joint2", "thumb_joint3", "thumb_joint4"],
    "index_finger": ["index_finger_join2", "index_finger_joint3", "index_finger_joint4"],
    "middle_finger": ["middle_finger_joint2", "middle_finger_joint3", "middle_finger_joint4"],
    "ring_finger": ["ring_finger_joint2", "ring_finger_joint3", "ring_finger_joint4"],
    "little_finger": ["little_finger_joint2", "little_finger_joint3", "little_finger_joint4"]
}

# GUI setup
root = tk.Tk()
root.title("Control Shadow Hand")

angle_vars = {}
for finger in finger_joints.keys():
    frame = tk.Frame(root)
    frame.pack(pady=5)

    label = tk.Label(frame, text=f"{finger.capitalize()} Rotation:")
    label.pack(side=tk.LEFT)

    angle_var = tk.StringVar(value="0")
    entry = tk.Entry(frame, textvariable=angle_var)
    entry.pack(side=tk.LEFT)

    angle_vars[finger] = angle_var

# Set finger angles
def set_finger_angle(finger, angle):
    target_angle = angle / 37.5  # Scale
    distributed_angle = target_angle / 3  # Even distribution

    for joint_name in finger_joints[finger]:
        if joint_name in joint_indices:
            p.setJointMotorControl2(
                robotId, joint_indices[joint_name],
                p.POSITION_CONTROL, targetPosition=distributed_angle
            )

# Fingertip position
def get_fingertip_position(finger_name):
    joints = finger_joints.get(finger_name, [])
    if not joints:
        return

    last_joint_name = joints[-1]
    joint_index = joint_indices.get(last_joint_name)
    if joint_index is None:
        return

    link_state = p.getLinkState(robotId, joint_index, computeForwardKinematics=True)
    pos = link_state[4]
    print(f"[{finger_name.capitalize()} Tip] Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
    return pos

# Draw axes at fingertip
def draw_fingertip_axes(finger_name):
    joints = finger_joints.get(finger_name, [])
    if not joints:
        return

    last_joint_name = joints[-1]
    joint_index = joint_indices.get(last_joint_name)
    if joint_index is None:
        return

    link_state = p.getLinkState(robotId, joint_index, computeForwardKinematics=True)
    pos = link_state[4]
    orn = link_state[5]

    # Quaternion → rotation matrix
    rot_matrix = p.getMatrixFromQuaternion(orn)
    
    x_dir = [rot_matrix[0], rot_matrix[1], rot_matrix[2]]
    y_dir = [rot_matrix[3], rot_matrix[4], rot_matrix[5]]
    z_dir = [rot_matrix[6], rot_matrix[7], rot_matrix[8]]

    axis_length = 0.03

    # Draw X (red), Y (green), Z (blue)
    p.addUserDebugLine(pos, [pos[0]+x_dir[0]*axis_length, pos[1]+x_dir[1]*axis_length, pos[2]+x_dir[2]*axis_length], [1, 0, 0], 2, 0.1)
    p.addUserDebugLine(pos, [pos[0]+y_dir[0]*axis_length, pos[1]+y_dir[1]*axis_length, pos[2]+y_dir[2]*axis_length], [0, 1, 0], 2, 0.1)
    p.addUserDebugLine(pos, [pos[0]+z_dir[0]*axis_length, pos[1]+z_dir[1]*axis_length, pos[2]+z_dir[2]*axis_length], [0, 0, 1], 2, 0.1)

    # Draw fingertip marker
    p.addUserDebugText("●", pos, textColorRGB=[1, 1, 0], textSize=1.5, lifeTime=0.2)

# Apply button logic
def apply_angles():
    for finger, angle_var in angle_vars.items():
        try:
            target_angle = float(angle_var.get())
            set_finger_angle(finger, target_angle)
            get_fingertip_position(finger)
            draw_fingertip_axes(finger)
        except ValueError:
            print(f"Invalid input for {finger}")

apply_button = tk.Button(root, text="Apply", command=apply_angles)
apply_button.pack(pady=10)

# Sim loop
def simulation_loop():
    p.stepSimulation()
    root.after(10, simulation_loop)

simulation_loop()
root.mainloop()
