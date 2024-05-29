import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

CATEGORIES = ['Cat', 'Dog']

def image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Wrong path:', path)
    else:
        new_arr = cv2.resize(img, (60, 60))
        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(-1, 60, 60, 1)
        return new_arr


model = keras.models.load_model('cat-vs-dog.keras')
image_path_dir = 'test_animals'
for image_file in os.listdir(image_path_dir):
    image_path = os.path.join(image_path_dir, image_file)
    if os.path.isfile(image_path):
        prediction = model.predict([image(image_path)])
        print(f"Это животное {image_file}: {CATEGORIES[prediction.argmax()]}")