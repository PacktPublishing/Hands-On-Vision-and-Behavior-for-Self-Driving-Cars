import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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

def show_history(history_object, plot_graph = True):
    ### print the keys contained in the history object
    print("Min Loss:", min(history_object.history['loss']))
    print("Min Val Loss:", min(history_object.history['val_loss']))
    print("Max Accuracy:", max(history_object.history['accuracy']))
    print("Max Validation Accuracy:", max(history_object.history['val_accuracy']))

    ### plot the training and validation loss for each epoch
    if plot_graph:
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.plot(history_object.history['accuracy'])
        plt.plot(history_object.history['val_accuracy'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['T loss', 'V loss', 'T acc', 'V acc'], loc='upper left')
        plt.show()