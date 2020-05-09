import cv2
import time

import sys
sys.path.append('../')
from utils import save_and_show

# Prepare the hog detector to detect pedestrians
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_pedestrians_hog(file, stride, scale, hitThreshold = 0, finalThreshold = 1):
    start = time.time()
    original = cv2.imread(file)
    hog_image = original.copy()

    (boxes, weights) = hog.detectMultiScale(hog_image, winStride=(stride, stride), padding=(0, 0), scale=scale, hitThreshold=hitThreshold, finalThreshold=finalThreshold)

    # Adds info
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.putText(hog_image, '%.2f' % weights[idx], (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.rectangle(hog_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    parameters = str(stride) + "_" + str(scale) + "_H_" + str(hitThreshold) + "_" + str(finalThreshold) + "_"
    cv2.imwrite("out_hog/hog_" + parameters + file, hog_image)
    end = time.time()
    print(stride, scale, "Time: ", end - start, "Thresholds", hitThreshold, finalThreshold, "Boxes", len(boxes))


# Shows some combinations, the precision and the computation time
detect_pedestrians_hog("ped.jpg", 2, 1.05, 0, 1)
detect_pedestrians_hog("ped.jpg", 2, 1.05, 0, 3)
detect_pedestrians_hog("ped.jpg", 2, 1.05, 0.2, 1)
detect_pedestrians_hog("ped.jpg", 2, 1.05, 0.2, 3)
detect_pedestrians_hog("ped.jpg", 4, 1.05)
detect_pedestrians_hog("ped.jpg", 8, 1.05)
detect_pedestrians_hog("ped.jpg", 2, 1.2)
detect_pedestrians_hog("ped.jpg", 4, 1.2)
detect_pedestrians_hog("ped.jpg", 8, 1.2)

def merge_pictures(res, files):
    images = [cv2.imread("out_hog/" + file) for file in files]
    save_and_show(res, images)

# Some interesting results
merge_pictures("hog_ped.jpg", ["hog_2_1.05_H_0_1_ped.jpg", "hog_2_1.05_H_0_3_ped.jpg", "hog_2_1.05_H_0.2_1_ped.jpg"])

