import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Rescaling
import time

# Определение констант
DIR = 'time'  # Путь к директории с изображениями
CATEGORIES = ['daytime', 'nighttime', 'sunrise']  # Категории изображений (день, ночь, рассвет)
IMG_SIZE = 224  # Размер изображений (224x224 пикселя)

# Функция для загрузки и предобработки изображений
def load_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DIR, category)  # Путь к категории
        label = CATEGORIES.index(category)  # Метка категории
        for img in os.listdir(path):  # Перебор всех изображений в категории
            try:
                img_path = os.path.join(path, img)  # Путь к изображению
                arr = cv2.imread(img_path)  # Загрузка изображения
                arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))  # Изменение размера изображения
                data.append([arr, label])  # Добавление изображения и его метки в данные
                print(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")  # Обработка ошибок при загрузке изображения
    return data

# Загрузка и перемешивание данных
data = load_data()
random.shuffle(data)

# Разделение данных на признаки (X) и метки (y)
X, y = zip(*data)
X = np.array(X)
y = np.array(y)

# Нормализация изображений
X = X / 255.0

# Сохранение данных с использованием pickle
pickle.dump(X, open('X_time2.pkl', 'wb'))
pickle.dump(y, open('y_time2.pkl', 'wb'))

# Загрузка данных из pickle
X = pickle.load(open('X_time2.pkl', 'rb'))
y = pickle.load(open('y_time2.pkl', 'rb'))

# Функция для создания модели
def create_model(dense_layers, conv_layers, neurons):
    model = Sequential()
    model.add(Rescaling(1. / 255, input_shape=(IMG_SIZE, IMG_SIZE, 3)))  # Масштабирование входных данных

    for _ in range(conv_layers):
        model.add(Conv2D(neurons, (3, 3), activation='relu'))  # Сверточный слой
        model.add(MaxPooling2D((2, 2)))  # Слой подвыборки

    model.add(Flatten())  # Преобразование данных в одномерный массив
    for _ in range(dense_layers):
        model.add(Dense(neurons, activation='relu'))  # Полносвязный слой
    model.add(Dense(len(CATEGORIES), activation='softmax'))  # Выходной слой

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])  # Компиляция модели
    return model

# Параметры для модели
dense_layers = [1]
conv_layers = [3]
neurons = [64]

# Создание и обучение модели
model = create_model(dense_layers[0], conv_layers[0], neurons[0])
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1)

# Функция для построения графиков истории обучения
def plot_history(history):
    # Построение графика точности
    fig = plt.figure()
    plt.plot(history.history['accuracy'], color='teal', label='accuracy')
    plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Построение графика потерь
    fig = plt.figure()
    plt.plot(history.history['loss'], color='teal', label='loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

plot_history(history)

# Сохранение модели
model.save('time_of_day_classifier.keras')
