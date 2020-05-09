import cv2
import numpy as np
import time

import sys
sys.path.append('../')
from utils import save_and_show

# Load an image, in BGR
image = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
original = image.copy()

# Image information
print("Image height: ", image.shape[0])
print("Image width: ", image.shape[1])
print("Channels: ", image.shape[2])
print("First pixel (B,G,R): ", image[0][0])

# Flip images
flipH = cv2.flip(image, 1)
flipV = cv2.flip(image, 0)
flip = cv2.flip(image, -1)

save_and_show("flip.jpg", [original, flipH, flipV, flip])

# Blur
start = time.time()
blurred = cv2.blur(image, (15, 15))
time_blur = time.time()
gaussian = cv2.GaussianBlur(image, (15, 15), sigmaX=15, sigmaY=15)
time_gaussian = time.time()
median = cv2.medianBlur(image, 15)
time_median = time.time()
bilateral = cv2.bilateralFilter(image, 15, 50, 50)
time_bilateral = time.time()

print("Blur", time_blur-start)
print("Gaussian", time_gaussian-time_blur)
print("Median", time_median-time_blur)
print("Bilateral", time_bilateral-time_blur)

save_and_show("blur.jpg", [original, blurred, gaussian, median, bilateral])

# Contrast, brightness, alpha
more_contrast = image.copy()
less_contrast = image.copy()
more_brightness = image.copy()
less_brightness = image.copy()

cv2.convertScaleAbs(image, more_contrast, 2, 0)
cv2.convertScaleAbs(image, less_contrast, 0.5, 0)

cv2.convertScaleAbs(image, more_brightness, 1, 64)
cv2.convertScaleAbs(image, less_brightness, 1, -64)

save_and_show("cont_bright.jpg", [original, more_contrast, less_contrast, more_brightness, less_brightness])


# Apply Gamma=1.5 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
higher_gamma = np.array(255 * (image / 255) ** (1 / 1.5), dtype='uint8')  # Gamma correction: 1.5
# Apply Gamma=0.7
lower_gamma = np.array(255 * (image / 255) ** (1 / 0.7), dtype='uint8')   # Gamma correction: 0.7
save_and_show("gamma.jpg", [original, higher_gamma, lower_gamma])
