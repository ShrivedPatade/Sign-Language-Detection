import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms
from collections import deque, Counter
from config import MODEL_PATH, IMG_SIZE, DEVICE, LABELS
# Import the model class from your training script
from train_cnn import SignLanguageCNN 

def run_detection():
    # 1. Load Model
    num_classes = len(LABELS)
    model = SignLanguageCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Preprocessing Transform
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    prediction_history = deque(maxlen=10)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        H, W, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = np.array([[lm.x * W, lm.y * H] for lm in hand_landmarks.landmark], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                
                # Crop and Predict
                crop = frame[max(0, y-20):min(H, y+h+20), max(0, x-20):min(W, x+w+20)]
                if crop.size != 0:
                    input_tensor = preprocess(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        prediction_history.append(LABELS[predicted.item()])

                    display_label = Counter(prediction_history).most_common(1)[0][0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, display_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('PyTorch Sign Detection (GPU)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()