import tensorflow as tf
import pathlib
import cv2
import numpy as np
import glob
from keras.applications.inception_v3 import preprocess_input


def find_files(pattern):
    files = []
    for file_name in glob.iglob(pattern, recursive=True):
        files.append(file_name)

    return files


def load_images_rgb(pattern, shape=None):
    files = find_files(pattern)

    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]

    if shape:
        return [cv2.resize(img, shape) for img in images]
    else:
        return images


# COCO labels are here: https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
LABEL_PERSON = 1
LABEL_CAR = 3
LABEL_BUS = 6
LABEL_TRUCK = 8
LABEL_TRAFFIC_LIGHT = 10


def load_model(model_name):
    url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)

    print("Model path: ", model_dir)
    model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def load_ssd_coco():
    return load_model("ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03")


def save_image_annotated(img_rgb, file_name, output, model_traffic_lights=None):
    out_file = file_name.replace('images', 'out_images')
    for idx in range(len(output['boxes'])):
        obj_class = output["detection_classes"][idx]
        score = int(output["detection_scores"][idx] * 100)
        box = output["boxes"][idx]
        color = None
        label_text = ""
        if obj_class == LABEL_CAR:
            color = (255, 255, 0)
            label_text = "Car " + str(score)
        if obj_class == LABEL_BUS:
            color = (255, 255, 0)
            label_text = "Bus " + str(score)
        if obj_class == LABEL_TRUCK:
            color = (255, 255, 0)
            label_text = "Truck " + str(score)
        elif obj_class == LABEL_PERSON:
            color = (0, 255, 255)
            label_text = "Person " + str(score)
        elif obj_class == LABEL_TRAFFIC_LIGHT:
            color = (255, 255, 255)
            label_text = "Traffic Light " + str(score)
            if model_traffic_lights:
                # Run inference
                img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                img_inception = cv2.resize(img_traffic_light, (299, 299))
                cv2.imwrite(out_file.replace('.jpg', '_crop.jpg'), cv2.cvtColor(img_inception, cv2.COLOR_RGB2BGR))
                img_inception = np.array([preprocess_input(img_inception)])

                prediction = model_traffic_lights.predict(img_inception)
                label = np.argmax(prediction)
                score_light = str(int(np.max(prediction) * 100))
                if label == 0:
                    label_text = "Green " + score_light
                elif label == 1:
                    label_text = "Yellow " + score_light
                elif label == 2:
                    label_text = "Red " + score_light
                else:
                    label_text = 'NO-LIGHT'  # Not a real traffic light

        if color and label_text and accept_box(output["boxes"], idx, 5.0):
            cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(out_file, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def center(box, coord_type):
    return (box[coord_type] + box[coord_type + "2"]) / 2


def accept_box(boxes, box_index, tolerance):
    box = boxes[box_index]

    for idx in range(box_index):
        other_box = boxes[idx]
        if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(center(other_box, "y") - center(box, "y")) < tolerance:
            return False

    return True


def detect_image(model, file_name, save_annotated=None, model_traffic_lights=None):
    img_bgr = cv2.imread(file_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output = model(input_tensor)

    print("num_detections:", output['num_detections'], int(output['num_detections']))

    # Converts the tensors to NumPy array
    num_detections = int(output.pop('num_detections'))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output['num_detections'] = num_detections

    print('Detection classes:', output['detection_classes'])
    print('Detection Boxes:', output['detection_boxes'])

    # detection_classes should be ints.
    output['detection_classes'] = output['detection_classes'].astype(np.int64)
    output['boxes'] = [
        {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
         "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]

    if save_annotated:
        save_image_annotated(img_rgb, file_name, output, model_traffic_lights)

    return img_rgb, output, file_name


def double_shuffle(images, labels):
    indexes = np.random.permutation(len(images))

    return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]

def reverse_preprocess_inception(img_preprocessed):
    img = img_preprocessed + 1.0
    img = img * 127.5
    return img.astype(np.uint8)