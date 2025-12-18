import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'sign_language_model.h5'
IMG_SIZE = 128
PADDING_FACTOR = 0.15
# The order of these letters must match the training folder numbers (0=A, 1=B, etc.)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# ---------------------

def run_detection():
    """
    Initializes the webcam and runs the real-time sign language detection loop.
    """
    # Load the trained CNN model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # --- Preprocess the frame exactly as done in 2_preprocess_images.py ---
            points = np.array([[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(points)

            x -= int(w * PADDING_FACTOR)
            y -= int(h * PADDING_FACTOR)
            w += int(2 * w * PADDING_FACTOR)
            h += int(2 * h * PADDING_FACTOR)
            
            x = max(0, x)
            y = max(0, y)

            # Crop the hand region from the original frame
            hand_crop = frame[y:y+h, x:x+w]
            
            if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                # Resize to the input size expected by the CNN
                resized_crop = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                
                # Reshape for model prediction (add a batch dimension)
                input_data = np.expand_dims(resized_crop, axis=0)
                
                # --- Make a Prediction ---
                prediction = model.predict(input_data, verbose=0)
                predicted_class_index = np.argmax(prediction)
                predicted_label = LABELS[predicted_class_index]
                confidence = np.max(prediction) * 100
                
                # --- Display the result on the frame ---
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Create the label text with confidence
                display_text = f"{predicted_label} ({confidence:.2f}%)"
                cv2.putText(frame, display_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the final frame
        cv2.imshow('Sign Language Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    run_detection()