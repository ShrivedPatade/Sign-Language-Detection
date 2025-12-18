import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling

# --- Configuration ---
PROCESSED_DATA_DIR = './processed_data'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20 # Number of times to train on the entire dataset
# ---------------------

def train_model():
    """
    Loads the preprocessed image data, builds a CNN model, trains it,
    and saves the trained model to a file.
    """
    # --- 1. Load Data ---
    # Use Keras utility to load images from directories.
    # It automatically infers labels from the folder names.
    # We'll use 80% of the data for training and 20% for validation.
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    # Get the number of classes from the directory names
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # --- 2. Build the CNN Model ---
    model = Sequential([
        # Normalize pixel values from [0, 255] to [0, 1]
        Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),

        # First convolutional block
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Flatten the feature maps to a 1D vector
        Flatten(),

        # Dense layer for classification
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting

        # Output layer with a neuron for each class
        Dense(num_classes, activation='softmax')
    ])

    # --- 3. Compile the Model ---
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print a summary of the model's architecture
    model.summary()

    # --- 4. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )
    print("Model training complete.")

    # --- 5. Save the Model ---
    model.save('sign_language_model.h5')
    print("Model saved successfully as 'sign_language_model.h5'")


if __name__ == "__main__":
    train_model()