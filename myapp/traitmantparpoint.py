import cv2
import numpy as np

def exponential_transformation(image, gamma=1.5):

    # Convert image to floating point representation
    im_out = image.astype(float)
    
    # Apply gamma correction
    im_out = ((im_out / 255.0) ** (1.0 / gamma)) * 255.0
    
    # Convert back to uint8
    im_out = np.uint8(im_out)
    
    return im_out

import numpy as np

def corr_gamma(image, gamma=0.5):
    # Convert image to floating point representation
    im_out = image.astype(float)
    
    # Apply gamma correction
    im_out = ((im_out / 255.0) ** (1.0 / gamma)) * 255.0
    
    # Convert back to uint8
    im_out = np.uint8(im_out)
    
    return im_out



def logarithmic_transformation(image):

      # Convert image to float64
    image = image.astype(np.float64)
    
    # Apply log transformation
    im_out = 255.0 * np.log(image + 1.0) / np.log(256.0)
    
    # Convert back to uint8
    im_out = np.uint8(im_out)
    
    return im_out




def image_inverse(image):

    inverted = 255 - image
    return inverted





def histogram_equalization(image):

    if len(image.shape) == 2 or image.shape[2] == 1:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray_image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    else:
        return equalized



def dynamic_range_adjustment(image):

    # Convert image to double precision
    image = image.astype(np.float64)
    
    # Compute minimum and maximum values
    a = np.min(image)
    b = np.max(image)
    
    # Get image dimensions
    rows, columns = image.shape
    
    # Initialize Rd array
    Rd = np.zeros((rows, columns))
    
    # Apply dynamic range adjustment
    for i in range(rows):
        for j in range(columns):
            Rd[i,j] = 255.0 * (image[i,j] - a) / (b - a)
    
    # Convert Rd back to uint8
    Rd = np.uint8(Rd)
    
    return Rd


def global_thresholding(image, threshold_value=127):
   
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded



































