import keras
from preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers import LSTM, SimpleRNN
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import numpy as np
import random
import sys
import io
import time
import argparse
import matplotlib.pyplot as plt

NAME = "Audio_Classification_CNN_{time}".format(time=int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{name}'.format(name=NAME))

max_len = 11
buckets = 20

# save_data_to_array(max_len=max_len, n_mfcc=buckets)
labels = ["bed", "happy", "cat"]

X_train, X_test, y_train, y_test = get_train_test()

channels = 1
epochs = 50
batch_size = 100

num_classes = 3

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)

# plt.imshow(X_train[1000, :, :, 0])
# plt.show()

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=(buckets, max_len, channels)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train, y_train_hot,
          epochs=epochs,
          validation_data=(X_test, y_test_hot),
          callbacks=[tensorboard])
