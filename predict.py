import os
import cv2
import numpy as np
from tensorflow import keras


CATEGORIES = ['daytime', 'nighttime', 'sunrise']
IMG_SIZE = 224

def image(path):
    img = cv2.imread(path)
    if img is None:
        print('Неправильный путь:', path)
    else:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img

# Загрузка модели
model = keras.models.load_model('time_of_day_classifier.keras')

image_path_dir = 'time_test'
for image_file in os.listdir(image_path_dir):
    image_path = os.path.join(image_path_dir, image_file)
    if os.path.isfile(image_path):
        img = image(image_path)
        prediction = model.predict(img)
        print(f"Изображение {image_file}: {CATEGORIES[np.argmax(prediction)]}")
