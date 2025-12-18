import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_PATH = os.path.join(BASE_DIR, 'sign_language_model.pth')

# Image Parameters
IMG_SIZE = 128
PADDING_FACTOR = 0.15

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 20

# Device Configuration: Automatically use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Labels (Mapped to folder names 0, 1, 2...)
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
NUM_CLASSES = len(LABELS)
SAMPLES_PER_CLASS = 100