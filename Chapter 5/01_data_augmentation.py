import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

x = np.array([cv2.resize(cv2.imread("test.jpg"), (120, 180))])

datagen = ImageDataGenerator(horizontal_flip=True)
print(x.shape)
x_augmented = datagen.flow(x).next()
print(x_augmented.shape)


def augment(images, num_samples, augmenters):
    rows = []

    for aug in augmenters:
        row = []
        iter = aug.flow(images)
        for i in range(num_samples):
          row.append(iter.next()[0].astype('uint8'))

        rows.append(cv2.hconcat(row))

    cv2.imshow("Augmented", cv2.vconcat(rows))
    cv2.waitKey(0)

augment(x, 4, [ImageDataGenerator(brightness_range=[0.1, 1.5]), ImageDataGenerator(rotation_range=60), ImageDataGenerator(width_shift_range=[-50, -25, 25, 50]), ImageDataGenerator(height_shift_range=[-75, -35, 35, 75])])
augment(x, 4, [ImageDataGenerator(shear_range=60), ImageDataGenerator(zoom_range=[0.5, 2]), ImageDataGenerator(horizontal_flip=True), ImageDataGenerator(vertical_flip=True)])
augment(x, 4, [ImageDataGenerator(brightness_range=[0.1, 1.5], rotation_range=60, width_shift_range=[-50, -25, 25, 50], horizontal_flip=True)])
#cv2.imshow("Augmented", cv2.hconcat([x[0], x_augmented[0].astype('uint8')]))
#cv2.waitKey(0)
