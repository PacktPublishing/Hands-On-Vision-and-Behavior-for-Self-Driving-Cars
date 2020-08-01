import cv2
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Lambda, Conv2D, SpatialDropout2D, GaussianNoise, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import regularizers
import collections
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, Adadelta
from keras.losses import categorical_crossentropy, mse
import sys
import glob
import keras
import numpy as np
import time
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from behavioral import expand_name, double_shuffle, shuffle

sys.path.append('../')

from utils import show_history_regression


def find_files(pattern):
    files = []
    for file_name in glob.iglob(pattern, recursive=True):
        files.append(file_name)

    return files

shape = (66, 200)
num_channels = 3
files = find_files("dataset_out/**/*")
#files.extend(find_files("dataset_out/recover/*"))
labels = []
images = []
for file in files:
    if ".jpg" in file:
        (seq, camera, steer, throttle, brake, img_type) = expand_name(file)
        labels.append(steer)
        if num_channels==1:
            images.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY))
        else:
            images.append(cv2.imread(file))

(images, labels) = double_shuffle(images, labels)
labels_np = np.array(labels)
images_np = np.ndarray(shape=(len(images), shape[0], shape[1], num_channels))

for idx in range(len(images)):
    images_np[idx] = np.reshape(images[idx],  (shape[0], shape[1], num_channels))


idx_split = int(len(labels) * 0.8)
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]

print(len(files), len(labels), len(images))
print(collections.Counter(labels))

def Dave2():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

def Dave2Faster():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 6), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

def Dave2Fastest():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3), name="net_input"))

    model.add(Conv2D(24, (5, 5), strides=(2, 6), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh', name="net_output"))

    return model



def DriveNet():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

def DriveNet2():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

def DriveNet3():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model

def DriveNet4():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Activation("relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(GaussianNoise(0.15))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(GaussianNoise(0.1))
    model.add(Activation("tanh"))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))

    return model

def DriveNet5():
    r_cn = regularizers.l2(0.001)
    r_fc = regularizers.l2(0.001)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), kernel_regularizer=r_cn))
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), kernel_regularizer=r_cn))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.2))
    model.add(Activation("relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), kernel_regularizer=r_cn))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3), kernel_regularizer=r_cn))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=r_cn))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164, kernel_regularizer=r_fc))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(100, kernel_regularizer=r_fc))
    model.add(GaussianNoise(0.15))
    model.add(Activation("relu"))
    model.add(Dense(50, kernel_regularizer=r_fc))
    model.add(GaussianNoise(0.1))
    model.add(Activation("tanh"))
    model.add(Dense(10, kernel_regularizer=r_fc, activation='tanh'))
    model.add(Dense(1, kernel_regularizer=r_fc))

    return model

def DriveNet6():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), activation='elu'))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.15))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(GaussianNoise(0.15))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(GaussianNoise(0.1))
    model.add(Activation("tanh"))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))

    return model

def DriveNet7():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, num_channels)))

    model.add(Conv2D(24, (5, 5), activation='elu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(GaussianNoise(0.15))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(GaussianNoise(0.1))
    model.add(Activation("tanh"))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))

    return model

def DriveNet8():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.15))

    model.add(Conv2D(96, (3, 3), activation='relu'))
    model.add(SpatialDropout2D(0.15))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu', padding="same"))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))


    return model

def Dave2Bis():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1))

    return model

def Dave2Tris():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3), name="net_input"))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.2))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, name="net_output"))

    return model



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow allowed growth to ", len(gpus), " GPUs")
    except RuntimeError as e:
        print(e)



datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.9, 1.1], height_shift_range=[-10, -5, -2, 0, 2, 5, 10])


checkpoint = ModelCheckpoint("behave.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.00005, patience=15, verbose=1)

model = Dave2()

model.summary()

#it_train = datagen.flow(x_train, y_train, batch_size=32)

model.compile(loss=mse, optimizer=Adam(), metrics=['cosine_proximity'])
# CV part 1: easier training
start = time.time()
history_object = model.fit(x_train, y_train, batch_size=32, epochs=250, validation_data=(x_valid, y_valid), shuffle=True, callbacks=[checkpoint, early_stopping])
end = time.time()

show_history_regression(history_object)
model = keras.models.load_model("behave.h5")
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation cosine proximity:', score[1])

print('Saving validation')
print("Training time: ", (end-start))
