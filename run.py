import numpy as np
import serial
import time
from scipy.optimize import minimize

# === Kinematics ===
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def get_joint_angles_from_pull(pull_amount, max_pull, max_angles_rad, ratios):
    pull_ratio = min(pull_amount / max_pull, 1.0)
    return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

def compute_positions(base_pos, lengths, joint_angles):
    points = [base_pos]
    transform = np.eye(3)
    direction = np.array([0.0, 1.0, 0.0])
    for i in range(3):
        transform = transform @ rot_x(joint_angles[i])
        new_point = points[-1] + transform @ (lengths[i] * direction)
        points.append(new_point)
    return points

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

# === Hardware Interface ===
SERIAL_PORT = 'COM15' 
BAUDRATE = 9600
SERVO_HORN_RADIUS_CM = 1.4

def pull_to_servo_angle(pull_cm):
    angle = (pull_cm / (2 * np.pi * SERVO_HORN_RADIUS_CM)) * 360
    return np.clip(angle, 0, 180)

# def send_to_arduino(servo_angles):
#     with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2) as ser:
#         time.sleep(2)  # wait for Arduino to reboot
#         command = ';'.join([f"M{i}:{int(a)}" for i, a in enumerate(servo_angles)]) + '\n'
#         print("Sending:", command.strip())
#         ser.write(command.encode())
#         response = ser.readline().decode().strip()
#         print("Arduino response:", response)

def send_to_arduino(servo_angles):
    
    adjusted_angles = [angle + 78 if angle != 0 and i != len(servo_angles) - 1 else angle for i, angle in enumerate(servo_angles)]
    
    with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2) as ser:
        time.sleep(2)  # wait for Arduino to reboot
        command = ';'.join([f"M{i}:{int(a)}" for i, a in enumerate(adjusted_angles)]) + '\n'
        print("Sending:", command.strip())
        ser.write(command.encode())
        response = ser.readline().decode().strip()
        print("Arduino response:", response)


# === Main Control ===
finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
lengths = [
    [3.5, 2.5, 1.5], [3, 2.5, 2.0], [3.5, 3.0, 2.5],
    [3.0, 2.5, 2.4], [2.2, 1.8, 1.5]
]
ratios = [[1.0, 0.7, 0.5]] * 5
max_angles_deg = [[55, 90, 90]] * 5
max_angles_rad = [np.radians(angles) for angles in max_angles_deg]
base_positions = [np.array([i * 2.5, 0.0, 0.0]) for i in range(5)]
max_pull_cm = 2.5

def move_hand_to_targets(target_positions, thumb_angle_override=None):
    servo_angles = []
    for i in range(6):
        if i == 5 and thumb_angle_override is not None:
           
            servo_angles.append(thumb_angle_override)
        else:
            print(f"Finger {i}:")
            target = target_positions[i]
            pull, _ = inverse_kinematics(
                target, base_positions[i], lengths[i], max_pull_cm,
                max_angles_rad[i], ratios[i]
            )
            if pull is None:
                servo_angles.append(0)
            else:
                servo_angles.append(pull_to_servo_angle(pull))

    
    
    

    send_to_arduino(servo_angles)
    
    
# === Example usage ===
if __name__ == "__main__":
    target_positions = {
 'A': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'B': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([7.5, 7.9, 0. ]),
       np.array([10. ,  5.5,  0. ])],
 'C': [np.array([1. , 0.5, 0. ]),
       np.array([2.5       , 4.24515946, 5.50571017]),
       np.array([5.        , 4.9756452 , 6.68259205]),
       np.array([7.5       , 4.26418692, 5.9211207 ]),
       np.array([10.        ,  3.04694316,  4.06987409])],
 'D': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'E': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'F': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'G': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'H': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'I': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10. ,  5.5,  0. ])],
 'J': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10. ,  5.5,  0. ])],
 'K': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'L': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'M': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'N': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'O': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'P': [np.array([1. , 0.5, 0. ]),
       np.array([2.5       , 5.34763808, 4.74914903]),
       np.array([5.        , 0.28905226, 7.24154611]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'Q': [np.array([1. , 0.5, 0. ]),
       np.array([2.5       , 5.34763808, 4.74914903]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'R': [np.array([1. , 0.5, 0. ]),
       np.array([2.5       , 5.34763808, 4.74914903]),
       np.array([5.        , 7.69739494, 4.26482907]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'S': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'T': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'U': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'V': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'W': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([5., 9., 0.]),
       np.array([7.5, 7.9, 0. ]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'X': [np.array([1. , 0.5, 0. ]),
       np.array([2.5       , 5.92299342, 4.18090606]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 'Y': [np.array([1. , 0.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10. ,  5.5,  0. ])],
 'Z': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])]}
    
    Thumb_angle = {
 'A': [180],
 'B': [0],
 'C': [0],
 'D': [90],
 'E': [0],
 'F': [60],
 'G': [90],
 'H': [90],
 'I': [0],
 'J': [0],
 'K': [90],
 'L': [180],
 'M': [0],
 'N': [0],
 'O': [90],
 'P': [90],
 'Q': [90],
 'R': [80],
 'S': [180],
 'T': [90],
 'U': [90],
 'V': [90],
 'W': [0],
 'X': [180],
 'Y': [180],
 'Z': [180]}
    
    letter = 'A'
    thumb_direct_angle=Thumb_angle[letter][0]
    print(thumb_direct_angle)
    move_hand_to_targets(target_positions[letter], thumb_angle_override=thumb_direct_angle)