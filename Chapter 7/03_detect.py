import object_detection
import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import numpy as np
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

model_inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299,299,3))
img = cv2.resize(preprocess_input(cv2.imread("test.jpg")), (299, 299))
out_inception = model_inception.predict(np.array([img]))
out_inception = imagenet_utils.decode_predictions(out_inception)
print("Prediction for test.jpg: ", out_inception[0][0][1], out_inception[0][0][2], "%")
model_inception.summary()


files = object_detection.find_files('images/*.jpg')
model_ssd = object_detection.load_ssd_coco()
model_traffic_lights = keras.models.load_model("traffic.h5")
print("Input:", model_ssd.inputs)
print("Output: ", model_ssd.output_dtypes)

for file in files:
    (img, out, file_name) = object_detection.detect_image(model_ssd, file, True, model_traffic_lights)
    print(file, out)
