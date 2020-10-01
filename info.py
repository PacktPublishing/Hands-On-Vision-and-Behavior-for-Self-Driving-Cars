import sys
import vis

print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

import cv2

print("OpenCV:", cv2.__version__)

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("TensorFlow Git:", tf.version.GIT_VERSION)

print("CUDA ON" if tf.test.is_built_with_cuda() else "CUDA OFF")
print("GPU ON" if tf.test.is_gpu_available() else "GPU OFF")

import keras

print("kERAS:", keras.__version__)

