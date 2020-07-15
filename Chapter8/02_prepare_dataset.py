import cv2
from behavioral import fix_name

import sys
sys.path.append('../')

import glob


def prepare_files(dir_name, steering_angle):
    files = []

    for filename in glob.iglob(dir_name + '/**/*', recursive=True):
        files.append(filename)

    print("*** Processing ", len(files), "files from", dir_name)
    for file in files:
        if ".png" in file or ".jpg" in file:
            file_out = file.replace(dir_name, 'dataset_out')
            file_mirrored = file_out.replace('.jpg', '_MIRROR.jpg')
            file_mirrored = file_mirrored.replace('.png', '_MIRROR.png')
            file_out = file_out.replace('.jpg', '_ORIG.jpg')
            file_out = file_out.replace('.png', '_ORIG.png')
            img = cv2.imread(file)
            img = cv2.resize(img, (200, 133))
            img = img[67:, :]
            print(file, img.shape)

            cv2.imwrite(fix_name(file_out, "jpg", steering_angle, True), img)

            name_mirrored = fix_name(file_mirrored, "jpg", steering_angle, True)

            if name_mirrored:
                cv2.imwrite(name_mirrored, cv2.flip(img, 1))


prepare_files("dataset_in", 0.25)

