import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

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
    if not (os.path.isdir("out_images")):
        os.mkdir("out_images")

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
    print("Min Loss:", min(history_object.history['loss']))
    print("Min Validation Loss:", min(history_object.history['val_loss']))
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

def show_history_regression(history_object, plot_graph = True):
    print("Min Loss:", min(history_object.history['loss']))
    print("Min Validation Loss:", min(history_object.history['val_loss']))
    print("Max Cosine Proximity:", max(history_object.history['cosine_proximity']))
    print("Max Validation Cosine Proximity:", max(history_object.history['val_cosine_proximity']))

    ### plot the training and validation loss for each epoch
    if plot_graph:
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.plot(history_object.history['cosine_proximity'])
        plt.plot(history_object.history['val_cosine_proximity'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['T loss', 'V loss', 'T cos', 'V cos'], loc='upper left')
        plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def bgr2gray(rgb):
    return np.dot(rgb[...,:3], [0.114, 0.587, 0.299])

def find_files(pattern):
    files = []
    for file_name in glob.iglob(pattern, recursive=True):
        files.append(file_name)

    return files

def double_shuffle(images, labels):
    assert len(images) == len(labels)
    use_np = hasattr(images, 'shape')

    if use_np:
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        return (images[indices], labels[indices])

    indexes = np.random.permutation(len(images))

    return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]

def shuffle(data):
    indexes = np.random.permutation(len(data))

    return [data[idx] for idx in indexes]