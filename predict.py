import cv2
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
image_path = 'test_images/image.jpg'
emotion = predict_emotion(image_path)
print(f"Эмоция на этом изображении: {emotion}")