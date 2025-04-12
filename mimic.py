import cv2
import mediapipe as mp
import numpy as np
import time
import serial
from scipy.optimize import minimize

# ===== Hand Tracking with MediaPipe =====
class HandTracker:
    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """Initialize the hand tracking system."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define finger segment lengths (can be adjusted)
        self.finger_lengths = {
            'thumb': [3.5, 2.5, 1.5],    # CMC to MCP, MCP to IP, IP to TIP
            'index': [3, 2.5, 2.0],      # MCP to PIP, PIP to DIP, DIP to TIP
            'middle': [3.5, 3.0, 2.5],   # MCP to PIP, PIP to DIP, DIP to TIP
            'ring': [3.0, 2.5, 2.4],     # MCP to PIP, PIP to DIP, DIP to TIP
            'pinky': [2.2, 1.8, 1.5]     # MCP to PIP, PIP to DIP, DIP to TIP
        }
        
        # Define base positions for each finger
        self.base_positions = [
            np.array([0.0, 0.0, 0.0]),    # Thumb
            np.array([2.5, 0.0, 0.0]),    # Index
            np.array([5.0, 0.0, 0.0]),    # Middle
            np.array([7.5, 0.0, 0.0]),    # Ring
            np.array([10.0, 0.0, 0.0])    # Pinky
        ]
        
    def find_hands(self, frame, draw=True):
        """Detect hands in a frame and optionally draw landmarks."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get hand landmarks
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks if requested
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, results
    
    def get_landmark_positions(self, frame, results):
        """Extract landmark positions from detection results."""
        height, width, _ = frame.shape
        hand_landmarks_dict = {}
        hand_visible = False
        
        if results.multi_hand_landmarks:
            hand_visible = True
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract all landmark positions
            for id, landmark in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                x, y, z = landmark.x * width, landmark.y * height, landmark.z * width
                hand_landmarks_dict[id] = (x, y, z)
        
        return hand_landmarks_dict, hand_visible
    
    def calculate_finger_angles(self, landmarks_dict):
        """Calculate 3 joint angles for each finger based on landmark positions."""
        if not landmarks_dict:
            return None

        angles_dict = {
            'thumb': {
                'CMC': self._calculate_bend_angle(landmarks_dict, 0, 1, 2),
                'MCP': self._calculate_bend_angle(landmarks_dict, 1, 2, 3),
                'IP': self._calculate_bend_angle(landmarks_dict, 2, 3, 4)
            },
            'index': {
                'MCP': self._calculate_bend_angle(landmarks_dict, 0, 5, 6),
                'PIP': self._calculate_bend_angle(landmarks_dict, 5, 6, 7),
                'DIP': self._calculate_bend_angle(landmarks_dict, 6, 7, 8)
            },
            'middle': {
                'MCP': self._calculate_bend_angle(landmarks_dict, 0, 9, 10),
                'PIP': self._calculate_bend_angle(landmarks_dict, 9, 10, 11),
                'DIP': self._calculate_bend_angle(landmarks_dict, 10, 11, 12)
            },
            'ring': {
                'MCP': self._calculate_bend_angle(landmarks_dict, 0, 13, 14),
                'PIP': self._calculate_bend_angle(landmarks_dict, 13, 14, 15),
                'DIP': self._calculate_bend_angle(landmarks_dict, 14, 15, 16)
            },
            'pinky': {
                'MCP': self._calculate_bend_angle(landmarks_dict, 0, 17, 18),
                'PIP': self._calculate_bend_angle(landmarks_dict, 17, 18, 19),
                'DIP': self._calculate_bend_angle(landmarks_dict, 18, 19, 20)
            }
        }

        return angles_dict
    
    def _calculate_bend_angle(self, landmarks, p1, p2, p3):
        """Calculate the bend angle between three points."""
        if p1 not in landmarks or p2 not in landmarks or p3 not in landmarks:
            return 0
        
        # Get coordinates
        a = np.array(landmarks[p1])
        b = np.array(landmarks[p2])
        c = np.array(landmarks[p3])
        
        # Calculate vectors
        vector1 = a - b  # Vector from joint2 to joint1
        vector2 = c - b  # Vector from joint2 to joint3
        
        # Normalize vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        
        # Calculate dot product
        dot_product = np.dot(vector1_norm, vector2_norm)
        
        # Calculate angle
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Invert the angle so that 0 = straight, 180 = fully bent
        bend_angle = 180 - angle_deg
        
        return bend_angle
    
    def compute_finger_positions(self, finger_angles):
        """Compute 3D positions of finger joints based on joint angles."""
        if not finger_angles:
            return None
        
        tip_positions = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, finger in enumerate(finger_names):
            if finger not in finger_angles:
                continue
                
            # Get joint angles for this finger
            angles = []
            for joint in ['CMC', 'MCP', 'IP'] if finger == 'thumb' else ['MCP', 'PIP', 'DIP']:
                if joint in finger_angles[finger]:
                    # Convert to radians for computation
                    angles.append(np.radians(finger_angles[finger][joint]))
                else:
                    angles.append(0.0)  # Default to 0 if missing
                    
            # Compute joint positions using forward kinematics
            positions = self.compute_positions(
                self.base_positions[i], 
                self.finger_lengths[finger], 
                angles
            )
            
            # Store only the tip position (last element)
            tip_positions[finger] = positions[-1]
            
        return tip_positions
    
    def compute_positions(self, base_pos, lengths, joint_angles):
        """Compute positions of finger joints based on base position, segment lengths, and joint angles."""
        points = [base_pos]
        transform = np.eye(3)
        direction = np.array([0.0, 1.0, 0.0])
        
        for i in range(len(joint_angles)):
            transform = transform @ self.rot_x(joint_angles[i])
            new_point = points[-1] + transform @ (lengths[i] * direction)
            points.append(new_point)
            
        return points
    
    def rot_x(self, theta):
        """Create a rotation matrix around the X axis."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t]
        ])

    def get_tip_positions_as_list(self, landmarks, frame):
        """Get all finger tip positions from landmarks as a list."""
        if not landmarks:
            return None
            
        # Calculate finger angles
        finger_angles = self.calculate_finger_angles(landmarks)
        
        # Compute finger tip positions
        tip_positions_dict = self.compute_finger_positions(finger_angles)
        
        # Convert to list format that matches the format used in the servo control code
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        tip_positions_list = []
        
        for finger in finger_order:
            if finger in tip_positions_dict:
                tip_positions_list.append(tip_positions_dict[finger])
            else:
                # Default position if finger not detected
                tip_positions_list.append(np.array([0.0, 0.0, 0.0]))
        
        # Calculate thumb angle (based on wrist, thumb base, and index base)
        thumb_angle = self._calculate_thumb_angle(landmarks)
                
        return tip_positions_list, thumb_angle
        
    def _calculate_thumb_angle(self, landmarks):
        """Calculate a thumb rotation angle from landmarks."""
        if 0 not in landmarks or 1 not in landmarks or 5 not in landmarks:
            return 90  # Default if landmarks not found
            
        # Get wrist, thumb CMC, and index base positions
        wrist = np.array(landmarks[0])
        thumb_cmc = np.array(landmarks[1])
        index_base = np.array(landmarks[5])
        
        # Calculate vectors
        wrist_to_thumb = thumb_cmc - wrist
        wrist_to_index = index_base - wrist
        
        # Project onto the x-z plane (removing y component for a top-down view)
        wrist_to_thumb_xz = np.array([wrist_to_thumb[0], wrist_to_thumb[2]])
        wrist_to_index_xz = np.array([wrist_to_index[0], wrist_to_index[2]])
        
        # Normalize vectors
        if np.linalg.norm(wrist_to_thumb_xz) > 0 and np.linalg.norm(wrist_to_index_xz) > 0:
            wrist_to_thumb_xz = wrist_to_thumb_xz / np.linalg.norm(wrist_to_thumb_xz)
            wrist_to_index_xz = wrist_to_index_xz / np.linalg.norm(wrist_to_index_xz)
            
            # Calculate angle
            dot_product = np.dot(wrist_to_thumb_xz, wrist_to_index_xz)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            # Map to servo angle range (0-180)
            # Adjust this mapping based on your specific needs
            servo_angle = np.clip(angle_deg, 0, 180)
            return servo_angle
        else:
            return 90  # Default angle

# ===== Servo Control System =====
class ServoController:
    def __init__(self, serial_port='COM15', baudrate=9600):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.servo_horn_radius_cm = 1.4
        
        # Finger parameters
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self.lengths = [
            [3.5, 2.5, 1.5], [3, 2.5, 2.0], [3.5, 3.0, 2.5],
            [3.0, 2.5, 2.4], [2.2, 1.8, 1.5]
        ]
        self.ratios = [[1.0, 0.7, 0.5]] * 5
        self.max_angles_deg = [[55, 90, 90]] * 5
        self.max_angles_rad = [np.radians(angles) for angles in self.max_angles_deg]
        self.base_positions = [np.array([i * 2.5, 0.0, 0.0]) for i in range(5)]
        self.max_pull_cm = 2.5
        
    def rot_x(self, theta):
        """Create a rotation matrix around the X axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])

    def get_joint_angles_from_pull(self, pull_amount, max_pull, max_angles_rad, ratios):
        """Convert pull amount to joint angles."""
        pull_ratio = min(pull_amount / max_pull, 1.0)
        return [pull_ratio * r * max_angle for r, max_angle in zip(ratios, max_angles_rad)]

    def compute_positions(self, base_pos, lengths, joint_angles):
        """Compute positions of finger joints based on base position, segment lengths, and joint angles."""
        points = [base_pos]
        transform = np.eye(3)
        direction = np.array([0.0, 1.0, 0.0])
        
        for i in range(3):
            transform = transform @ self.rot_x(joint_angles[i])
            new_point = points[-1] + transform @ (lengths[i] * direction)
            points.append(new_point)
            
        return points

    def inverse_kinematics(self, target_pos, base_pos, lengths, max_pull, max_angles_rad, ratios):
        """Find the pull amount needed to reach a target position."""
        def forward_kinematics(pull):
            angles = self.get_joint_angles_from_pull(pull, max_pull, max_angles_rad, ratios)
            points = self.compute_positions(base_pos, lengths, angles)
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

    def pull_to_servo_angle(self, pull_cm):
        """Convert pull amount to servo angle."""
        angle = (pull_cm / (2 * np.pi * self.servo_horn_radius_cm)) * 360
        return np.clip(angle, 0, 180)

    def send_to_arduino(self, servo_angles):
        """Send servo angles to Arduino."""
        adjusted_angles = [angle + 78 if int(angle) == 102 and angle != 0 and i != len(servo_angles) - 1 else angle for i, angle in enumerate(servo_angles)]
        print(adjusted_angles)
        adjusted_angles[len(servo_angles) - 1] = abs(180-adjusted_angles[len(servo_angles) - 1])
        try:
            with serial.Serial(self.serial_port, self.baudrate, timeout=2) as ser:
                time.sleep(2)  # wait for Arduino to reboot
                command = ';'.join([f"M{i}:{int(a)}" for i, a in enumerate(adjusted_angles)]) + '\n'
                print("Sending:", command.strip())
                ser.write(command.encode())
                response = ser.readline().decode().strip()
                print("Arduino response:", response)
        except Exception as e:
            print(f"Serial communication error: {e}")

    def move_hand_to_targets(self, target_positions, thumb_angle_override=None):
        """Calculate servo angles from target positions and send to Arduino."""
        servo_angles = []
        
        for i in range(6):  # 5 fingers + 1 thumb rotation
            if i == 5 and thumb_angle_override is not None:
                # Special case for thumb rotation servo
                servo_angles.append(thumb_angle_override)
            else:
                if i < len(target_positions):
                    print(f"Finger {i}:")
                    target = target_positions[i]
                    pull, _ = self.inverse_kinematics(
                        target, self.base_positions[i], self.lengths[i], self.max_pull_cm,
                        self.max_angles_rad[i], self.ratios[i]
                    )
                    if pull is None:
                        servo_angles.append(0)
                    else:
                        servo_angles.append(self.pull_to_servo_angle(pull))
                else:
                    servo_angles.append(0)  # Default angle if position not provided

        self.send_to_arduino(servo_angles)

# ===== Main Application =====
def run_hand_mimicry_system(camera_id=0, serial_port='COM15', baudrate=9600, update_rate=0.1):
    """Run the hand tracking and servo control system."""
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Initialize servo controller
    servo_controller = ServoController(serial_port, baudrate)
    
    # Initialize timer
    last_update_time = time.time()
    
    print("Hand mimicry system started. Press 'q' to quit.")
    
    while cap.isOpened():
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Failed to capture video")
            break
        
        # Find hands and draw landmarks
        frame, results = tracker.find_hands(frame, draw=True)
        
        # Get landmark positions
        landmarks, hand_visible = tracker.get_landmark_positions(frame, results)
        
        # Current time
        current_time = time.time()
        
        # Update servo positions at the specified rate
        if current_time - last_update_time >= update_rate and hand_visible:
            # Get finger tip positions and thumb angle
            tip_positions, thumb_angle = tracker.get_tip_positions_as_list(landmarks, frame)
            
            if tip_positions:
                # Send positions to servo controller
                servo_controller.move_hand_to_targets(tip_positions, thumb_angle_override=thumb_angle)
                
                # Display status on frame
                cv2.putText(frame, "Sending to servos", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update last update time
                last_update_time = current_time
        
        # Display FPS
        fps = 1 / (time.time() - current_time + 0.001)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                  
        # Display hand visibility status
        status = "Hand Detected" if hand_visible else "No Hand"
        cv2.putText(frame, status, (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display the camera feed with hand landmarks
        cv2.imshow('Hand Mimicry System', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    # Configuration - update these values as needed
    CAMERA_ID = 0       # Usually 0 for built-in webcam, try 1, 2, etc. for external cameras
    SERIAL_PORT = 'COM15'  # Update to your Arduino's serial port
    BAUDRATE = 9600
    UPDATE_RATE = 10   # Update servos every 0.2 seconds (5 times per second)
    
    run_hand_mimicry_system(
        camera_id=CAMERA_ID,
        serial_port=SERIAL_PORT,
        baudrate=BAUDRATE,
        update_rate=UPDATE_RATE
    )