import sys
from time import time

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, AveragePooling2D
from keras.layers import Dropout, Activation, BatchNormalization, ReLU, Concatenate, \
    Conv2DTranspose
from keras.losses import categorical_crossentropy
from keras.models import Input, Model
from keras.optimizers import Adam

sys.path.append('../')

from utils import find_files, show_history, shuffle

from segmentation import convert_to_segmentation_label
import sys

sys.path.append('../')

size = (160, 160)
size_cv = (size[1], size[0])
batch_size = 4
num_classes = 13
epochs = 150


def generator(ids, fn_image, fn_label, augment, batch_size):
    num_samples = len(ids)
    while 1:  # Loop forever so the generator never terminates
        samples_ids = shuffle(ids)  # New epoch

        for offset in range(0, num_samples, batch_size):
            batch_samples_ids = samples_ids[offset:offset + batch_size]
            batch_samples = np.array([fn_image(x, augment, offset + idx) for idx, x in enumerate(batch_samples_ids)])
            batch_labels = np.array([fn_label(x, augment, offset + idx) for idx, x in enumerate(batch_samples_ids)])

            yield batch_samples, batch_labels


def extract_image(file_name, augment, idx):
    img = cv2.resize(cv2.imread(file_name), size_cv, interpolation=cv2.INTER_NEAREST)

    if augment and (idx % 2 == 0):
        img = cv2.flip(img, 1)

    return img


def extract_label(file_name, augment, idx):
    img = cv2.resize(cv2.imread(file_name.replace("rgb", "seg_raw", 2)), size_cv, interpolation=cv2.INTER_NEAREST)

    if augment and (idx % 2 == 0):
        img = cv2.flip(img, 1)

    return convert_to_segmentation_label(img, num_classes)


files = find_files("dataset/rgb/*.png")
idx_split = int(len(files) * 0.8)
val_size = len(files) - idx_split

print("Training: ", idx_split, "images, validation:", val_size, " - Images size: ", size)

train_gen = generator(files[0:idx_split], extract_image,
                      extract_label, True, batch_size)

valid_gen = generator(files[idx_split:], extract_image,
                      extract_label, False, batch_size)


model_name = "fc_densenet.h5"
checkpoint = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.00005, patience=15, verbose=1)


def dn_conv(layer, num_filters, kernel_size, dropout=0.0):
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(num_filters, kernel_size, padding="same", kernel_initializer='he_uniform')(layer)

    if dropout > 0.0:
        layer = Dropout(dropout)(layer)

    return layer


def dn_dense(layer, growth_rate, num_layers, add_bottleneck_layer, dropout=0.0):
    block_layers = []
    for i in range(num_layers):
        new_layer = dn_conv(layer, 4 * growth_rate, (1, 1), dropout) if add_bottleneck_layer else layer
        new_layer = dn_conv(new_layer, growth_rate, (3, 3), dropout)
        block_layers.append(new_layer)
        # Skip connection including all the previous layers
        layer = Concatenate()([layer, new_layer])

    return layer, Concatenate()(block_layers)


def dn_transition_down(layer, compression_factor=1.0, dropout=0.0):
    assert 0.0 < compression_factor <= 1.0

    num_filters_compressed = int(layer.shape[-1] * compression_factor)
    layer = dn_conv(layer, num_filters_compressed, (1, 1), dropout)

    return AveragePooling2D(2, 2, padding='same')(layer)


def dn_transition_up(skip_connection, layer):
    num_filters = int(layer.shape[-1])
    layer = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding='same',
                            kernel_initializer='he_uniform')(layer)

    return Concatenate()([layer, skip_connection])

def conv_net(input_shape, num_channels, dropout=0.0):
    input = Input(input_shape)

    # Initial convolutional layer
    layer = Conv2D(num_channels, 7, padding='same', activation="relu")(input)
    for i in range(8):
        layer = Conv2D(num_channels*(i+2), 3, padding='same', activation="relu")(layer)
        if dropout > 0.0:
            layer = Dropout(dropout)(layer)
    layer = Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(layer)

    output = Activation('softmax')(layer)

    return Model(input, output)


def fc_dense_net(input_shape, dropout=0.0):
    growth_rate = 12
    groups = 5
    transition_compression_factor = 0.6
    add_bottleneck_layer = False

    input = Input(input_shape)

    # Initial convolutional layer
    layer = Conv2D(36, 7, padding='same')(input)

    ### Down-sampling
    skip_connections = []

    for idx in range(groups):
        (layer, _) = dn_dense(layer, growth_rate, 4, add_bottleneck_layer, dropout)
        skip_connections.append(layer)
        layer = dn_transition_down(layer, transition_compression_factor, dropout)

    # We need the skip connections in reverse order when going up
    skip_connections.reverse()

    ### Bottleneck layer
    (layer, block_layers) = dn_dense(layer, growth_rate, 4, add_bottleneck_layer, dropout)

    ### Up-sampling
    for idx in range(groups):
        layer = dn_transition_up(skip_connections[idx], block_layers)

        (layer, block_layers) = dn_dense(layer, growth_rate, 4, add_bottleneck_layer, dropout)

    # Output
    layer = Conv2D(num_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(layer)
    output = Activation('softmax')(layer)

    return Model(input, output)


model = fc_dense_net((size[0], size[1], 3), 0.25)
#model = conv_net((size[0], size[1], 3), 36, 0.20)
model.summary()
opt = Adam()
#opt = RAdam(total_steps=50, warmup_proportion=0.1, min_lr=1e-5)

# model.compile(loss=mse, optimizer=opt, metrics=['accuracy', 'cosine_proximity'])
model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
start = time()
# history_object = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpoint])
history_object = model.fit(train_gen, epochs=epochs,
                           steps_per_epoch=idx_split / batch_size, validation_data=valid_gen,
                           validation_steps=val_size / batch_size, shuffle=False, callbacks=
                           [checkpoint, early_stopping])
print("Training time:", time() - start)

show_history(history_object)
