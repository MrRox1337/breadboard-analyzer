import serial
import time
import threading
import tensorflow as tf
import numpy as np
import cv2
import os

# ==========================================
# 0. HARDWARE SETUP
# ==========================================
SERIAL_PORT = 'COM5' # Change to your port
BAUD_RATE = 9600

# Initialize Serial Connection
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for Arduino to reset
    print("Connected to Arduino hardware.")
    # Clear all LEDs on startup
    arduino.write(b'C')
except:
    print("Arduino not found. Check SERIAL_PORT.")
    arduino = None

def hardware_handler(result):
    if not arduino:
        return
    
    if result == 'PASS':
        arduino.write(b'P')
    elif result == 'FAIL':
        arduino.write(b'F')
        
    # Wait for 5 seconds
    time.sleep(5)
    
    # Send clear signal to turn all LEDs off
    arduino.write(b'C')

def update_hardware(result):
    if arduino:
        # Run in a background thread to prevent webcam frame freezing
        t = threading.Thread(target=hardware_handler, args=(result,))
        t.start()

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
MODEL_PATH = 'best_breadboard_model.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = ['FAIL', 'PASS'] 

print("Loading model... Please wait.")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'best_breadboard_model.keras' is in the same folder.")
    exit()

def predict_image(img_array):
    img_array = tf.expand_dims(img_array, 0) 
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

    update_hardware(predicted_class)


# ==========================================
# 3. MODE: LIVE VIDEO FEED
# ==========================================
def analyze_video():
    print("\n--- LIVE VIDEO MODE ---")
    print("Opening webcam...")
    print("Press 't' to TEST the current frame.")
    print("Press 'q' to QUIT the video feed.\n")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    display_text = "Press 't' to Test"
    color = (255, 255, 255) # Default White

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)

        cv2.imshow('Hardware Notifier Mode', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            print("Capturing frame for analysis...")
            
            resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img_array = np.array(rgb_frame, dtype=np.float32)
            
            predicted_class, confidence = predict_image(img_array)
            
            display_text = f"RESULT: {predicted_class} ({confidence:.1f}%)"
            print(display_text)
            
            if predicted_class == 'PASS':
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            
            update_hardware(predicted_class)
                
        elif key == ord('q'):
            print("Closing webcam...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================================
# MAIN MENU
# ==========================================
if __name__ == "__main__":
    print("=======================================")
    print("    Breadboard Quality Control AI      ")
    print("          (Arduino Edition)            ")
    print("=======================================")
    print("1: Analyze a single static image")
    print("2: Start live webcam analyzer")
    print("=======================================")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        img_path = input("Enter the path to your image (e.g., test.jpg): ")
        analyze_static_image(img_path)
    elif choice == '2':
        analyze_video()
    else:
        print("Invalid choice. Exiting.")