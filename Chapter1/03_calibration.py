import cv2
import numpy as np
import os

import sys
sys.path.append('../')
from utils import show

# Creates a calibration check-board, to eventually print it
black = np.zeros([50, 50], dtype=np.uint8)
white = np.full([50, 50], 255, dtype=np.uint8)

row_1 = cv2.hconcat([white, black, white, black, white, black, white, black, white, black])
row_2 = cv2.hconcat([black, white, black, white, black, white, black, white, black, white])

board = cv2.vconcat([row_1, row_2, row_1, row_2, row_1, row_2, row_1, row_2])
cv2.imwrite("out_calibration/calib_chess.png", board)
show("chessboard", board)

# Calibration parameters
nX = 9
nY = 7
image_points = []
object_points = []

coords = np.zeros((1, nX * nY, 3), np.float32)
coords[0,:,:2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2)

shape = None
use_sub_pixels = True

# Give calibration files to OpenCV
for filename in os.listdir("in_calibration"):
    img_src = cv2.cvtColor(cv2.imread("in_calibration/" + filename), cv2.COLOR_BGR2GRAY)
    img_copy = img_src.copy()
    found, corners = cv2.findChessboardCorners(img_src, (nY, nX), None)

    print(filename, ": ", found, img_src.shape)
    if found:
        # Increase precision to sub-pixel coordinates
        if (use_sub_pixels):
            corners = cv2.cornerSubPix(img_src, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        object_points.append(coords)
        image_points.append(corners)

        img_corners = cv2.drawChessboardCorners(img_src, (nY, nX), corners, True)
        cv2.imwrite("out_calibration/" + filename, img_corners)
        shape = img_src.shape

        if "3685" in filename:
            show("Calibration", [img_copy, img_corners])


height, width = shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, shape[::-1], None, None)
crop_image = True
optimal_matrix, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height), 0 if crop_image else 1,(width,height))
x,y,w,h = roi
print(x,y,w,h, width, height)

for filename in os.listdir("in_calibration"):
    img_src = cv2.cvtColor(cv2.imread("in_calibration/" + filename), cv2.COLOR_BGR2GRAY)
    img_dst = cv2.undistort(img_src, mtx, dist, None, mtx)
    cv2.imwrite('out_calibrated/' + filename, img_dst)

for filename in os.listdir("test_calibration"):
    img_src = cv2.imread("test_calibration/" + filename)
    img_dst = cv2.undistort(img_src, mtx, dist, None, optimal_matrix)
    cv2.imwrite('out_calibrated/' + filename, img_dst)
    show("Calibrated", [img_src, img_dst])


