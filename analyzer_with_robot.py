import tensorflow as tf
import numpy as np
import cv2
import os
import socket

# ==========================================
# 0. ROBOT SETUP (TCP/IP CLIENT)
# ==========================================
ROBOT_IP = '127.0.0.1'   # Localhost for Simulator
ROBOT_PORT = 2000        # Ensure this matches Epson RC+ System Configuration

print("Connecting to Epson VT6 Simulator...")
try:
    robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot_socket.connect((ROBOT_IP, ROBOT_PORT))
    print("Connected to Robot via TCP/IP!\n")
except Exception as e:
    print(f"Could not connect to robot: {e}")
    print("Running in standalone model mode without robot integration.\n")
    robot_socket = None

def send_to_robot(command):
    """Sends a string command to the Epson RC+ SPEL+ program."""
    if robot_socket:
        try:
            # SPEL+ Input expects a CRLF (\r\n) termination by default
            robot_socket.sendall(f"{command}\r\n".encode('utf-8'))
            print(f"--> [ROBOT] Instructed manipulator: {command}")
        except Exception as e:
            print(f"--> [ROBOT] Failed to send command: {e}")

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
# Make sure this matches your final grid search model name
MODEL_PATH = 'best_breadboard_model.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Keras automatically sorts folders alphabetically, so F comes before P
CLASS_NAMES = ['FAIL', 'PASS'] 

print("Loading model... Please wait.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'best_breadboard_model.keras' is in the same folder.")
    exit()

# Helper function to process the prediction
def predict_image(img_array):
    # Expand dimensions to create a batch of 1
    img_array = tf.expand_dims(img_array, 0) 
    
    # Run the prediction
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence


# ==========================================
# 2. MODE: STATIC IMAGE
# ==========================================
def analyze_static_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    # Load image using Keras (reads as RGB)
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    
    predicted_class, confidence = predict_image(img_array)
    
    print(f"\n--- ANALYSIS RESULT ---")
    print(f"Image: {image_path}")
    print(f"Result: {predicted_class} ({confidence:.2f}% confidence)\n")

    # Send command to Epson RC+ Simulator
    send_to_robot(predicted_class)


# ==========================================
# 3. MODE: LIVE VIDEO FEED
# ==========================================
def analyze_video():
    print("\n--- LIVE VIDEO MODE ---")
    print("Opening webcam...")
    print("Press 't' to TEST the current frame.")
    print("Press 'q' to QUIT the video feed.\n")
    
    # 0 is usually the default built-in webcam. 
    # Change to 1 or 2 if you have an external USB camera plugged in.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # To store the result text to display on screen
    display_text = "Press 't' to Test"
    color = (255, 255, 255) # Default White

    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Draw the latest analysis result on the screen
        # cv2 uses BGR colors (Blue, Green, Red)
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow('Breadboard Analyzer', frame)

        # Wait for key press (1ms delay)
        key = cv2.waitKey(1) & 0xFF

        # If 't' is pressed -> Test the circuit
        if key == ord('t'):
            print("Capturing frame for analysis...")
            
            # 1. Resize the frame to match what the CNN expects (224x224)
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            
            # 2. OpenCV captures in BGR, but our model was trained on RGB images. We must convert it!
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # 3. Convert to numpy array and predict
            img_array = np.array(rgb_frame, dtype=np.float32)
            predicted_class, confidence = predict_image(img_array)
            
            # Update the text to show the result on the video feed
            display_text = f"RESULT: {predicted_class} ({confidence:.1f}%)"
            print(display_text)
            
            # Change text color based on result
            if predicted_class == 'PASS':
                color = (0, 255, 0) # Green for PASS
            else:
                color = (0, 0, 255) # Red for FAIL

            # Send result command to Epson RC+ Simulator
            send_to_robot(predicted_class)
                
        # If 'q' is pressed -> Quit
        elif key == ord('q'):
            print("Closing webcam...")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
    if robot_socket:
        robot_socket.close()


# ==========================================
# MAIN MENU
# ==========================================
if __name__ == "__main__":
    print("=======================================")
    print("    Breadboard Quality Control AI      ")
    print("=======================================")
    print("1: Analyze a single static image")
    print("2: Start live webcam analyzer")
    print("=======================================")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        img_path = input("Enter the path to your image (e.g., test.jpg): ")
        analyze_static_image(img_path)
    elif choice == '2':
        # You may need to run `pip install opencv-python` in your terminal if you don't have it
        analyze_video()
    else:
        print("Invalid choice. Exiting.")
