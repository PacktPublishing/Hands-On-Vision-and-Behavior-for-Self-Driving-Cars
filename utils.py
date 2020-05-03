import cv2

def saveAndShow(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images

    cv2.imwrite("out_images/" + name, img)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def show(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def save(name, images):
    img = cv2.hconcat(images) if isinstance(images, list) else images

    cv2.imwrite("out_images/" + name, img)

    return img