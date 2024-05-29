import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model('face-predict.keras')

def predict_emotion(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    if prediction >= 0.5:
        return "happy"
    else:
        return "sad"

# Использование:
image_path_dir = 'test_images'
for image_file in os.listdir(image_path_dir):
    image_path = os.path.join(image_path_dir, image_file)
    if os.path.isfile(image_path):
        emotion = predict_emotion(image_path)
        print(f"Эмоция на изображении {image_file}: {emotion}")