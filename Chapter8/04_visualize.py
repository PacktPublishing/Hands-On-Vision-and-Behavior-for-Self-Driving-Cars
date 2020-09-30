import cv2
import glob
import numpy as np
import keras
import keras.models
from vis.visualization import visualize_saliency, visualize_activation
from keras import activations
import tempfile
import os

import matplotlib.pyplot as plt

def update_model(model):
    model_path = tempfile.gettempdir() + '/' + next(tempfile._get_candidate_names()) + '.h5'
    try:
        model.save(model_path)
        return keras.models.load_model(model_path)
    finally:
        os.remove(model_path)


def find_files(pattern):
    files = []
    for file_name in glob.iglob(pattern, recursive=True):
        files.append(file_name)

    return files


def visualize_single(model, conv_name, image, show_activations = True):
    from keras import models

    conv_layer, idx_layer = next((layer.output, idx) for idx, layer in enumerate(model.layers) if
                                 layer.output.name.startswith(conv_name))

    act_model = models.Model(inputs=model.input, outputs=[conv_layer])

    if show_activations:
        layer_activations = act_model.predict([[image]])

        print("Activations:", layer_activations.shape, "Index: ", idx_layer, len(act_model.layers))
        col_act = []

        for pred_idx, act in enumerate(layer_activations):
            row_act = []

            for act_idx in range(act.shape[2]):
                row_act.append(act[:, :, act_idx])

            col_act.append(cv2.hconcat(row_act))

        plt.matshow(cv2.vconcat(col_act), cmap='viridis')
        plt.waitforbuttonpress()
        plt.close()

    conv_layer.activation = activations.linear
    sal_model = update_model(act_model)

    grads = visualize_saliency(sal_model, idx_layer, filter_indices=None, seed_input=image)
    plt.matshow(image)
    plt.imshow(grads, alpha=.6)
    plt.waitforbuttonpress()
    plt.close()

shape = (66, 200)
files = find_files("visual/00003048_LEFT_0.25_0.050000_0.000000_ORIG.jpg")


images = []
for file in files:
    images.append(cv2.imread(file))

images_np = np.ndarray(shape=(len(images), shape[0], shape[1], 3), dtype=np.uint8)

for idx in range(len(images)):
    images_np[idx] = np.reshape(images[idx],  (shape[0], shape[1], 3))

print(images[0].shape)

cv2.waitKey(-1)

model = keras.models.load_model("behave.h5")

model.summary()

visualize_single(model, "conv2d_1", images[0])
visualize_single(model, "conv2d_2", images[0])
visualize_single(model, "conv2d_3", images[0])
visualize_single(model, "conv2d_4", images[0])
visualize_single(model, "conv2d_5", images[0])

visualize_single(model, "dense_1", images[0], False)
visualize_single(model, "dense_5", images[0], False)


