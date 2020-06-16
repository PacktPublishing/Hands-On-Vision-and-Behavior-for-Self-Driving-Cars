import cv2
import object_detection
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.utils import to_categorical
import collections
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, Adadelta
from keras.losses import categorical_crossentropy
import sys
import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('../')

from utils import show_history

print("Keras", keras.__version__)
print("Tensorflow", tf.__version__)


# Version using transfer learning from Inception V3
def Transfer(n_classes, freeze_layers=True):
    print('Loading InceptionV3')
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    print('InceptionV3 loaded.')

    print('Layers: ', len(base_model.layers))
    print("Shape:", base_model.output_shape[1:])
    print("Shape:", base_model.output_shape)
    print("Shape:", base_model.outputs)

    base_model.summary()

    top_model = Sequential()
    top_model.add(base_model)
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(0.5))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(n_classes, activation='softmax'))

    # model = Model(input=base_model.input, output=top_model)

    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    return top_model


datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             horizontal_flip=True)

shape = (299, 299)
img_0_green = object_detection.load_images_rgb("traffic_dataset/0_green/*", shape)
img_1_yellow = object_detection.load_images_rgb("traffic_dataset/1_yellow/*", shape)
img_2_red = object_detection.load_images_rgb("traffic_dataset/2_red/*", shape)
img_3_not_traffic_light = object_detection.load_images_rgb("traffic_dataset/3_not/*", shape)

labels = [0] * len(img_0_green)
labels.extend([1] * len(img_1_yellow))
labels.extend([2] * len(img_2_red))
labels.extend([3] * len(img_3_not_traffic_light))

labels_np = np.ndarray(shape=(len(labels), 4))
images_np = np.ndarray(shape=(len(labels), shape[0], shape[1], 3))

img_all = []
img_all.extend(img_0_green)
img_all.extend(img_1_yellow)
img_all.extend(img_2_red)
img_all.extend(img_3_not_traffic_light)

assert len(img_all) == len(labels)

img_all = [preprocess_input(img) for img in img_all]
(img_all, labels) = object_detection.double_shuffle(img_all, labels)

for idx in range(len(labels)):
    images_np[idx] = img_all[idx]
    labels_np[idx] = labels[idx]

print("Images: ", len(img_all))
print("Labels: ", len(labels))

for idx in range(len(labels_np)):
    labels_np[idx] = np.array(to_categorical(labels[idx], 4))

idx_split = int(len(labels_np) * 0.8)
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]

cnt = collections.Counter(labels)
print('Labels:', cnt)
n = len(labels)
print('0:', cnt[0])
print('1:', cnt[1])
print('2:', cnt[2])
print('3:', cnt[3])
class_weight = {0: n / cnt[0], 1: n / cnt[1], 2: n / cnt[2], 3: n / cnt[3]}
print('Class weight:', class_weight)

checkpoint = ModelCheckpoint("traffic.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=15, verbose=1)

model = Transfer(4)

model.summary()

it_train = datagen.flow(x_train, y_train, batch_size=32)

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
history_object = model.fit(it_train, epochs=250, validation_data=(x_valid, y_valid), shuffle=True,
                           callbacks=[checkpoint, early_stopping], class_weight=class_weight)

show_history(history_object)
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

print('Saving validation')

for idx in range(len(x_valid)):
    img_as_ar = np.array([x_valid[idx]])
    prediction = model.predict(img_as_ar)
    label = np.argmax(prediction)
    file_name = "out_valid/" + str(label) + "/" + str(idx) + "_" + str(label) + "_" + str(np.argmax(str(y_valid[idx]))) + ".jpg"
    img = img_as_ar[0]
    img = object_detection.reverse_preprocess_inception(img)
    cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print('Validation saved!')
