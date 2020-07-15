import os
import numpy as np

def to_float(str):
    f = float(str)

    return 0 if f == -0.0 else f

def expand_name(file):
    idx = int(max(file.rfind('/'), file.rfind('\\')))
    prefix = file[0:idx]
    file = file[idx:].replace('.png', '').replace('.jpg', '')
    parts = file.split('_')

    (seq, camera, steer, throttle, brake, img_type) = parts

    return (prefix + seq, camera, to_float(steer), to_float(throttle), to_float(brake), img_type)

def fix_name(file_name, extension, steering_correction, create_dir):
    (seq, camera, steer, throttle, brake, img_type) = expand_name(file_name)

    if camera == 'LEFT':
        steer = steer + steering_correction
    if camera == 'RIGHT':
        steer = steer - steering_correction

    if img_type == "MIRROR":
        steer = -steer

    new_name = seq + "_" + camera + "_" + str(steer) + "_" + str(throttle) + "_" + str(brake) + "_" + img_type + "." + extension

    if create_dir:
        directory = os.path.dirname(new_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

    return new_name

def double_shuffle(images, labels):
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
