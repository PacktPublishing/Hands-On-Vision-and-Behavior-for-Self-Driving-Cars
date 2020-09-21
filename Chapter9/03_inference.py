import sys

sys.path.append('../')

from utils import find_files
from segmentation import convert_from_segmentation_label
import keras
import cv2
import numpy as np
import os

size = (160,160)
out_size = (200,160)

files=find_files("dataset_test/rgb_*")
model = keras.models.load_model("fc_densenet.h5")

add_median = True
add_alpha = True

if not (os.path.isdir("dataset_out")):
    os.mkdir("dataset_out")

for file in files:
    img = cv2.resize(cv2.imread(file), size, interpolation=cv2.INTER_NEAREST)

    result = model.predict(np.asarray(img)[None, :, :, :], batch_size=1)[0]
    (raw, color) = convert_from_segmentation_label(result)
    new_file = file.replace("dataset_test", "dataset_out")

    #ground_truth = file.replace("dataset_test", "dataset").replace("rgb", "seg\\seg")
    ground_truth = file.replace("rgb_", "seg_")
    print(file, " ==> ", new_file, "+",ground_truth)

    images = []
    images.append(img)
    if (os.path.exists(ground_truth)):
        images.append(cv2.resize(cv2.imread(ground_truth), size, interpolation=cv2.INTER_NEAREST))
    images.append(color)

    if add_median:
        images.append(cv2.medianBlur(color, 3))

    if add_alpha:
        images.append(cv2.addWeighted(img, 0.6, color, 0.4, 0.0))

    images = [cv2.resize(x, out_size) for x in images]
    img_res = cv2.hconcat(images)

    print(file, " ==> ", new_file)
    cv2.imwrite(new_file, img_res)


print("Done")