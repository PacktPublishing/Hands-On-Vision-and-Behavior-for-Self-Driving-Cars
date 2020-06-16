import cv2
import object_detection
from tensorflow.keras.applications.inception_v3 import preprocess_input

files = object_detection.find_files('traffic_input_full/*.png')
model = object_detection.load_ssd_coco()

num_traffic_lights = 0
num_files = 0

print("Files:", len(files))

for file in files:
    (img_rgb, out, file_name) = object_detection.detect_image(model, file, None)
    if (num_files % 10) == 0:
        print("Files processed:", num_files)
        print("Traffic lights found: ", num_traffic_lights)
    num_files = num_files + 1
    for idx in range(len(out['boxes'])):
        obj_class = out["detection_classes"][idx]
        if obj_class == object_detection.LABEL_TRAFFIC_LIGHT:
            box = out["boxes"][idx]
            traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
            traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)
            cv2.imwrite("traffic_input_cropped/" + str(num_traffic_lights) + ".png", traffic_light)
            num_traffic_lights = num_traffic_lights + 1

print("Traffic lights found:", num_traffic_lights)
