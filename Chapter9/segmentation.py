import numpy as np
from keras.utils import to_categorical

palette = [] # in rgb

palette.append([0, 0, 0])  # 0: None
palette.append([70, 70, 70])  # 1: Buildings
palette.append([190, 153, 153])  # 2: Fences
palette.append([192, 192, 192])  # 3: Other  (?)
palette.append([220, 20, 60])  # 4: Pedestrians
palette.append([153,153, 153])  # 5: Poles
palette.append([0, 255, 0])  # 6: RoadLines  ?
palette.append([128, 64, 128])  # 7: Roads
palette.append([244, 35,232])  # 8: Sidewalks
palette.append([107, 142, 35])  # 9: Vegetation
palette.append([0, 0, 142])  # 10: Vehicles
palette.append([102,102,156])  # 11: Walls
palette.append([220, 220, 0])  # 11: Traffic signs

def convert_to_segmentation_label(image, num_classes):
    img_label = np.ndarray((image.shape[0], image.shape[1], num_classes), dtype=np.uint8)

    one_hot_encoding = []

    for i in range(num_classes):
        one_hot_encoding.append(to_categorical(i, num_classes))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            img_label[i, j] = one_hot_encoding[image[i, j, 2]]

    return img_label

def convert_from_segmentation_label(label):
    raw = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            color_label = int(np.argmax(label[i,j]))
            raw[i, j][2] = color_label
            # palette from rgb to bgr
            color[i, j][0] = palette[color_label][2]
            color[i, j][1] = palette[color_label][1]
            color[i, j][2] = palette[color_label][0]

    return (raw, color)

