import cv2
import keras
from keras.datasets import mnist, cifar10
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

use_mnist = False

name = "mnist" if use_mnist else "cifar10"
model_name = name + ".h5"
num_classes = 10

# Loading test and training datasets
if use_mnist:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    x_train = np.reshape(x_train, np.append(x_train.shape, (1)))
    print(x_train.shape)
    x_test = np.reshape(x_test, np.append(x_test.shape, (1)))
else:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def display_conv(model, conv_name, num_predictions):
    from keras import models

    conv_layer = next(x.output for x in model.layers if x.output.name.startswith(conv_name))
    if conv_layer is not None:
        act_model = models.Model(inputs=model.input, outputs=[conv_layer])
        activations = act_model.predict(x_test[0:num_predictions, :, :, :])

        print("Activations:", activations.shape)

        col_act = []

        for pred_idx, act in enumerate(activations):
            row_act = []

            for idx in range(act.shape[2]):
                row_act.append(act[:, :, idx])

            col_act.append(cv2.hconcat(row_act))

        plt.matshow(cv2.vconcat(col_act), cmap='viridis')
        plt.show()
        plt.waitforbuttonpress()
        plt.close()


model = load_model(model_name)
model.summary()
print("H5 Output: " + str(model.output.op.name))
print("H5 Input: " + str(model.input.op.name))

# Score trained model.
print ("X Test:", x_test.shape, "Y Test:", y_test.shape)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])
x_pred = model.predict(x_test[0:1, :, :, :])
print("Expected:", np.argmax(y_test))
print("First prediction prob:", x_pred)
print("First prediction:", np.argmax(x_pred))

display_conv(model, "conv2d_1", 10)
display_conv(model, "conv2d_2", 30)
display_conv(model, "conv2d_4", 30)
display_conv(model, "conv2d_6", 30)
