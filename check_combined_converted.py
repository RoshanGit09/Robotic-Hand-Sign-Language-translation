import whisper
import pyaudio
import numpy as np
import time
import threading
from scipy.fft import fft
import serial
from scipy.optimize import minimize

# Speech Recognition Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio_buffer = np.array([], dtype=np.float32)
is_recording = True

# Voice activity detection parameters
ENERGY_THRESHOLD = 0.005  # Adjust based on your environment
MIN_SILENCE_DURATION = 0.5  # seconds
voice_active = False
silence_start_time = None

# State control flags
currently_displaying = False  # Flag to indicate if hand is currently displaying letters
display_complete_event = threading.Event()  # Event to signal completion of displaying

# Serial Configuration
SERIAL_PORT = 'COM15'  # Change to your port
BAUDRATE = 9600
SERVO_HORN_RADIUS_CM = 1.4

# === Kinematics and Hand Configuration ===
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

# Pre-defined target positions for each letter
target_positions = {
 'A': [np.array([0 , 7.5, 0. ]),
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
 'G': [np.array([0. , 7.5, 0. ]),
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
 'L': [np.array([0. , 7.5, 0. ]),
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
 'Y': [np.array([0. , 7.5, 0. ]),
       np.array([ 2.5       , -1.36555911,  5.24956852]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10. ,  5.5,  0. ])],
 'Z': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 7.5, 0. ]),
       np.array([ 5.        , -1.79165905,  6.2468042 ]),
       np.array([ 7.5       , -1.74808101,  5.36651721]),
       np.array([10.        , -1.01763779,  3.82999772])],
 ' ': [np.array([1. , 0.5, 0. ]),
       np.array([2.5, 0.5, 0. ]),
       np.array([5. , 0.5, 0. ]),
       np.array([7.5, 0.5, 0. ]),
       np.array([10. , 0.5,  0. ])]
}


# Special thumb angles for each letter
thumb_angles = {
 'A': 180,
 'B': 0,
 'C': 0,
 'D': 90,
 'E': 0,
 'F': 60,
 'G': 90,
 'H': 90,
 'I': 0,
 'J': 0,
 'K': 90,
 'L': 180,
 'M': 0,
 'N': 0,
 'O': 90,
 'P': 90,
 'Q': 90,
 'R': 80,
 'S': 180,
 'T': 90,
 'U': 90,
 'V': 90,
 'W': 0,
 'X': 180,
 'Y': 180,
 'Z': 180,
 ' ': 0  # Neutral position
}

# === Kinematics Functions ===
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

def pull_to_servo_angle(pull_cm):
    angle = (pull_cm / (2 * np.pi * SERVO_HORN_RADIUS_CM)) * 360
    return np.clip(angle, 0, 180)

def send_to_arduino(servo_angles):
    adjusted_angles = [angle + 78 if angle != 0 and i != len(servo_angles) - 1 else angle for i, angle in enumerate(servo_angles)]
    
    try:
        with serial.Serial(SERIAL_PORT, BAUDRATE, timeout=2) as ser:
            time.sleep(2)  # wait for Arduino to reboot
            command = ';'.join([f"M{i}:{int(a)}" for i, a in enumerate(adjusted_angles)]) + '\n'
            print("Sending:", command.strip())
            ser.write(command.encode())
            response = ser.readline().decode().strip()
            print("Arduino response:", response)
    except Exception as e:
        print(f"Serial communication error: {e}")

def move_hand_to_letter(letter):
    """Move the robotic hand to form the letter shape"""
    letter = letter.upper()
    if letter not in target_positions:
        print(f"Unknown letter: {letter}, using neutral position")
        letter = ' '
    
    servo_angles = []
    
    # Process the first 5 fingers using target positions
    for i in range(5):
        target = target_positions[letter][i]
        pull, _ = inverse_kinematics(
            target, base_positions[i], lengths[i], max_pull_cm,
            max_angles_rad[i], ratios[i]
        )
        if pull is None:
            servo_angles.append(0)
        else:
            servo_angles.append(pull_to_servo_angle(pull))
    
    # Add the special thumb angle as the 6th servo
    servo_angles.append(thumb_angles.get(letter, 0))
    
    print(f"Moving hand to form letter '{letter}'")
    send_to_arduino(servo_angles)
    return True

# === Speech Recognition Functions ===

def detect_voice_activity(audio_chunk):
    """Detect if there is active speech using FFT analysis"""
    # Calculate RMS energy
    rms_energy = np.sqrt(np.mean(np.square(audio_chunk)))
    
    # Apply FFT analysis
    n = len(audio_chunk)
    if n == 0:
        return False
        
    # Apply window function and FFT
    windowed = audio_chunk * np.hanning(n)
    spectrum = fft(windowed)
    freq_magnitudes = np.abs(spectrum[:n//2])
    
    # Get frequency bins
    freq_bins = np.fft.fftfreq(n, 1/RATE)[:n//2]
    
    # Focus on speech frequencies (85Hz-3800Hz)
    speech_range = (freq_bins >= 85) & (freq_bins <= 3800)
    speech_energy = np.sum(freq_magnitudes[speech_range]) if np.any(speech_range) else 0
    
    # Detect voice based on energy
    return rms_energy > ENERGY_THRESHOLD

def display_text_with_hand(text):
    """Display text using hand movements and signal when complete"""
    global currently_displaying
    
    currently_displaying = True
    display_complete_event.clear()  # Reset the completion event
    
    try:
        # Move servos for each character
        for char in text:
            if char.isalpha() or char == ' ':
                move_hand_to_letter(char)
                time.sleep(1.5)  # Display duration per character
        
        # Return to neutral position
        move_hand_to_letter(' ')
        
        print("Display completed! Ready for next speech input.")
    except Exception as e:
        print(f"Error during hand display: {e}")
    finally:
        currently_displaying = False
        display_complete_event.set()  # Signal that display is complete

def record_audio():
    """Capture audio from microphone"""
    global audio_buffer, voice_active, silence_start_time
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print("System ready. Start speaking...")
    
    while is_recording:
        # Check if we should be recording based on hand movement state
        if currently_displaying:
            # Skip recording while displaying hand movements
            time.sleep(0.1)
            continue
        
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Check for voice activity
        is_active = detect_voice_activity(audio_chunk)
        
        # State management
        if is_active and not voice_active:
            voice_active = True
            silence_start_time = None
            print("Voice detected - recording speech...")
        elif not is_active and voice_active:
            if silence_start_time is None:
                silence_start_time = time.time()
            elif time.time() - silence_start_time > MIN_SILENCE_DURATION:
                voice_active = False
                print("Speech segment complete - processing...")
        
        # Add to buffer if voice is active or in short silence
        if voice_active or (silence_start_time is not None and 
                           time.time() - silence_start_time <= MIN_SILENCE_DURATION):
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))

    stream.stop_stream()
    stream.close()
    p.terminate()

def transcribe_and_control():
    """Process audio and control servos"""
    global audio_buffer, voice_active
    
    # Initialize Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Model loaded and ready.")
    
    while is_recording:
        # Don't process if currently displaying
        if currently_displaying:
            time.sleep(0.1)
            continue
        
        # Process when voice activity has ended and we have enough audio
        buffer_length = len(audio_buffer) / RATE  # seconds
        
        if buffer_length >= 0.5 and not voice_active and silence_start_time is not None:
            current_buffer = audio_buffer.copy()
            audio_buffer = np.array([], dtype=np.float32)  # Clear buffer
            
            try:
                # Transcribe speech
                result = model.transcribe(current_buffer, language='en', fp16=False)
                
                # Extract clean text for servo control
                clean_text = ''.join([c.upper() for c in result['text'] if c.isalpha() or c == ' ']).strip()
                
                if clean_text:
                    print(f"Transcribed: \"{result['text']}\"")
                    print(f"Moving hand to display: {clean_text}")
                    
                    # Start displaying the text with the hand
                    display_thread = threading.Thread(target=display_text_with_hand, args=(clean_text,))
                    display_thread.start()
                    
                    # Wait for display to complete before continuing
                    display_complete_event.wait()
                    
            except Exception as e:
                print(f"Error during transcription: {e}")
                audio_buffer = np.array([], dtype=np.float32)  # Clear buffer on error
        
        time.sleep(0.1)

def main():
    global is_recording
    
    print("Starting Speech-to-Sign Language Servo Control System")
    print("Press Ctrl+C to exit")
    
    # Initialize the display complete event
    display_complete_event.set()
    
    # Start threads
    record_thread = threading.Thread(target=record_audio)
    transcribe_thread = threading.Thread(target=transcribe_and_control)
    
    try:
        record_thread.start()
        transcribe_thread.start()
        
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        is_recording = False
        display_complete_event.set()  # Release any waiting threads
        record_thread.join()
        transcribe_thread.join()
        print("System stopped.")

if __name__ == "__main__":
    main()