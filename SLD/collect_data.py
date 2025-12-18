import os
import cv2
import time

# --- Configuration ---
from config import DATA_DIR, NUM_CLASSES, SAMPLES_PER_CLASS

def collect_images():
    """
    Initializes the webcam to capture and save images for each sign language gesture.
    """
    # Create the main data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Loop through each class
    for j in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(j))
        
        print(f"Starting in 3...")
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, str(i), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)
        
        # Create a directory for the current class
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {j}')

        # --- User prompt to start collection ---
        while True:
            ret, frame = cap.read()
            if not ret or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Error: Could not read frame.")
                break
            
            # Display text to prompt the user
            cv2.putText(frame, 'Ready? Press "W" to start collecting!', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Wait for 'w' key to be pressed
            if cv2.waitKey(25) & 0xFF == ord('w'):
                break
        
        # --- Image collection loop ---
        print("Collecting images...")
        counter = 0
        while counter < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Error: Could not read frame during collection.")
                break
            
            # Show the frame and save the image
            cv2.imshow('frame', frame)
            image_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(image_path, frame)
            
            counter += 1
            # A small delay to allow for different hand positions
            cv2.waitKey(25)

    print("Data collection complete.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()