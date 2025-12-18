import os
import cv2
import mediapipe as mp
import numpy as np

# --- Configuration ---
RAW_DATA_DIR = './data'
PROCESSED_DATA_DIR = './processed_data'
IMG_SIZE = 64
PADDING_FACTOR = 0.15 # Add 15% padding around the hand
# ---------------------

def preprocess_images():
    """
    Loads raw images, finds the hand, crops a square region around it,
    resizes it, and saves it to a new directory without segmentation.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    print("Starting simplified image preprocessing...")
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    for dir_name in sorted(os.listdir(RAW_DATA_DIR), key=int):
        raw_class_dir = os.path.join(RAW_DATA_DIR, dir_name)
        processed_class_dir = os.path.join(PROCESSED_DATA_DIR, dir_name)

        if not os.path.isdir(raw_class_dir):
            continue
        
        if not os.path.exists(processed_class_dir):
            os.makedirs(processed_class_dir)
        
        print(f"Processing class: {dir_name}")

        for img_name in os.listdir(raw_class_dir):
            img_path = os.path.join(raw_class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            H, W, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                points = np.array([[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark], dtype=np.int32)
                
                # Get the bounding box of the hand
                x, y, w, h = cv2.boundingRect(points)
                
                # Add padding to the bounding box
                x -= int(w * PADDING_FACTOR)
                y -= int(h * PADDING_FACTOR)
                w += int(2 * w * PADDING_FACTOR)
                h += int(2 * h * PADDING_FACTOR)
                
                # Ensure coordinates are within image boundaries
                x = max(0, x)
                y = max(0, y)
                
                # Crop the original image (not segmented)
                cropped_img = img[y:y+h, x:x+w]
                
                if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                    continue
                
                # Resize to the final standard size
                final_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE))
                
                # Save the final image
                cv2.imwrite(os.path.join(processed_class_dir, img_name), final_img)

    print("Image preprocessing complete.")
    hands.close()

if __name__ == "__main__":
    preprocess_images()