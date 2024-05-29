import pickle

from matplotlib import pyplot as plt

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X/255
X = X.reshape(-1, 60, 60, 1)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from keras.callbacks import TensorBoard
import time

dense_layers = [3]
conv_layers = [3]
neurons = [64]

for dense_layer in dense_layers:
  for conv_layer in conv_layers:
    for neuron in neurons:

      NAME = '{}-denselayer-{}-convlayer-neuron-{}'.format(
          dense_layer,
          conv_layer,
          neuron,
          int(time.time())
      )

      tensorboard = TensorBoard(log_dir = 'logs2\\{}'.format(NAME))

      model = Sequential()

      for i in range(conv_layer):
        model.add(Conv2D(neuron, (3,3), activation = 'relu'))
        model.add(MaxPooling2D((2,2)))

      model.add(Flatten())

      model.add(Dense(neuron, input_shape = X.shape[1:], activation = 'relu'))

      for l in range(dense_layer - 1):
        model.add(Dense(neuron, activation = 'relu'))

      model.add(Dense(2, activation = 'softmax'))

      model.compile(optimizer = 'adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

      hist = model.fit(X, y, epochs = 5, batch_size = 32, validation_split = 0.1, callbacks = [tensorboard])

      fig = plt.figure()
      plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
      plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
      fig.suptitle('Accuracy', fontsize=20)
      plt.legend(loc="upper left")
      plt.show()

      # Plot loss
      fig = plt.figure()
      plt.plot(hist.history['loss'], color='teal', label='loss')
      plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
      fig.suptitle('Loss', fontsize=20)
      plt.legend(loc="upper left")
      plt.show()

      model.save('cat-vs-dog.keras')