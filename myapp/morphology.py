import cv2
import numpy as np

def load_image_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    return padded_image, image.shape[1], image.shape[0]

def save_image_data(image, output_path):
    cv2.imwrite(output_path, image)


element_structurant = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]], dtype=np.uint8)

def erosion(image):
    return cv2.erode(image, element_structurant, iterations=1)

def dilation(image):
    return cv2.dilate(image, element_structurant, iterations=1)

def ouverture(image):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, element_structurant)

def fermeture(image):
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, element_structurant)

def epaisissement(image):
    dilated = dilation(image)
    return cv2.subtract(dilated, image)

def gradient_morphologique(image):
    dilated = dilation(image)
    eroded = erosion(image)
    return cv2.subtract(dilated, eroded)

def aminicissement(image):
    skel = np.zeros(image.shape, np.uint8)
    element_structurant = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(image, element_structurant)
        temp = cv2.dilate(eroded, element_structurant)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return skel

def tout_ou_rien(image):
    dilated = dilation(image)
    eroded = erosion(image)
    return cv2.bitwise_and(dilated, cv2.bitwise_not(eroded))

def squelettisation(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element_structurant = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(image, element_structurant)
        temp = cv2.dilate(eroded, element_structurant)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return skel

def segmentation_seuillage(image, seuil=127):
    _, thresholded = cv2.threshold(image, seuil, 255, cv2.THRESH_BINARY)
    return thresholded



