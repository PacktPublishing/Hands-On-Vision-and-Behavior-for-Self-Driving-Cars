import cv2
import os
import numpy as np

save_files = True


def set_save_files(save_flag):
    global save_files

    save_files = save_flag


def get_save_files():
    global save_files

    return save_files


def save_and_show(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images

    cv2.imwrite("out_images/" + name, img)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def show(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def save(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images

    cv2.imwrite("out_images/" + name, img)

    return img


def ensure_dir(filename):
    if not (os.path.isdir("out_images")):
        os.mkdir("out_images")
    dir = "out_images/" + filename.lower().replace(".jpg", "")
    if not (os.path.isdir(dir)):
        os.mkdir(dir)

    return dir


def save_dir(img, prefix, filename, scale=1):
    if (save_files and filename):
        cv2.imwrite(ensure_dir(filename) + "/" + prefix + filename, img if scale == 1 else np.uint8(scale * img))

    return img
