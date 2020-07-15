import cv2
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, Lambda, Conv2D, SpatialDropout2D, GaussianNoise, Activation, MaxPooling2D
from keras.models import Model, Sequential
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
files = find_files("dataset_out/**/*.jpg")
#files.extend(find_files("dataset_out/recover/*"))

def extract_image(file_name):
    return cv2.imread(file_name)

def extract_label(file_name):
    (seq, camera, steer, throttle, brake, img_type) = expand_name(file_name)

    return steer


labels = []
images = []
for file in files:
    labels.append(extract_label(file))
    images.append(extract_image(file))

(images, labels) = double_shuffle(images, labels)
labels_np = np.array(labels)
images_np = np.ndarray(shape=(len(images), shape[0], shape[1], num_channels))

for idx in range(len(images)):
    images_np[idx] = np.reshape(images[idx],  (shape[0], shape[1], num_channels))


idx_split = int(len(files) * 0.8)
val_size = len(files) - idx_split
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]

def generator(ids, fn_image, fn_label, batch_size=32):
    num_samples = len(ids)
    while 1: # Loop forever so the generator never terminates
        samples_ids = shuffle(ids)  # New epoch

        for offset in range(0, num_samples, batch_size):
            batch_samples_ids = samples_ids[offset:offset + batch_size]
            batch_samples = [fn_image(x) for x in batch_samples_ids]
            batch_labels = [fn_label(x) for x in batch_samples_ids]

            yield np.array(batch_samples), np.array(batch_labels)

batch_size = 32

files = shuffle(files)

train_gen = generator(files[0:idx_split], extract_image, extract_label, batch_size)
valid_gen = generator(files[idx_split:], extract_image, extract_label, batch_size)

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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow allowed growth to ", len(gpus), " GPUs")
    except RuntimeError as e:
        print(e)



checkpoint = ModelCheckpoint("behave.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.00005, patience=15, verbose=1)

model = Dave2()

model.summary()

model.compile(loss=mse, optimizer=Adam(), metrics=['cosine_proximity'])
start = time.time()
history_object = model.fit(train_gen, epochs=250, steps_per_epoch=idx_split/batch_size, validation_data=valid_gen, validation_steps=val_size/batch_size, shuffle=False, callbacks=[checkpoint, early_stopping])
end = time.time()

show_history_regression(history_object)
model = keras.models.load_model("behave.h5")

print('Saving validation')
print("Training time: ", (end-start))
