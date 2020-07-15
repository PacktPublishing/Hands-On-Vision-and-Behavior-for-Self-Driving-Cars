import cv2
import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, SpatialDropout2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from time import time
import numpy as np

import sys
sys.path.append('../')

from utils import show_history,save
from keras.preprocessing.image import ImageDataGenerator

use_mnist = True

# Customize the training
name = "mnist" if use_mnist else "cifar10"
batch_size = 64
num_classes = 10
epochs = 250
augment = True
patience = 20

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=[-5, 0, 5], horizontal_flip=True)
#datagen = ImageDataGenerator(rotation_range=15, width_shift_range=[-8, -4, 0, 4, 8], horizontal_flip=True, height_shift_range=[-5, 0, 5], zoom_range=[0.9, 1.1])

print("Dataset in use: ", name.upper())

# Loading test and training datasets
if use_mnist:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train = np.reshape(x_train, np.append(x_train.shape, (1)))
    print(x_train.shape)
    x_test = np.reshape(x_test, np.append(x_test.shape, (1)))
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

save(name + "_train.jpg", cv2.hconcat([x_train[0], x_train[1], x_train[2], x_train[3], x_train[4]]))
save(name + "_test.jpg", cv2.hconcat([x_test[0], x_test[1], x_test[2], x_test[3], x_test[4]]))
print('X Train', x_train.shape, ' - X Test', x_test.shape)
print('Y Train', y_train.shape, ' - Y Test', y_test.shape)
print('First 5 labels, train:', y_train[0], y_train[1], y_train[2], y_train[3], y_train[4])
print('First 5 labels, test:', y_test[0], y_test[1], y_test[2], y_test[3], y_test[4])
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model_name = name + ".h5"
checkpoint = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=patience, verbose=1)

def create_model_0():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='elu', input_shape=x_train.shape[1:]))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model


def create_model_1():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_2():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_3():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_4():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_5():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_bn_1():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=num_classes, activation = 'softmax'))

    return model

def create_model_bn_2_dropout():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def create_model_bn_3_dropout():
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def create_model_bn_4_dropout():
    # Max Accuracy: 0.73944
    # Max Validation Accuracy: 0.8144999742507935

    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='elu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.3))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(Dropout(0.2))

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def create_model_bn_5_dropout():
    #Max Accuracy: 0.78884
    #Max Validation Accuracy: 0.8482999801635742

    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.25))

    model.add(Conv2D(filters=40, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=40, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.15))

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def create_model_bn_6_dropout():
    # Max Accuracy: 0.84324
    # Max Validation Accuracy: 0.8779000043869019

    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=48, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(filters=72, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=72, kernel_size=(3, 3), activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(AveragePooling2D())
    model.add(Dropout(0.1))

    # Fully Connected layers
    model.add(Flatten())
    model.add(Dense(units=384, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=num_classes, activation='softmax'))

    return model

# Choose the model that you want to train
model = create_model_bn_6_dropout()
model.summary()
opt = Adam()

model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
start = time()
if augment:
    it_train = datagen.flow(x_train, y_train, batch_size=batch_size)
    history_object = model.fit(it_train, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpoint, early_stopping])
else:
    history_object = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpoint, early_stopping])
print("Training time:", time()-start)

show_history(history_object)
