import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter

def load_image_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    return padded_image, image.shape[1], image.shape[0]

def save_image_data(image, output_path):
    cv2.imwrite(output_path, image)



def filtre_Moy(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_in = img_in.astype(np.float64)
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            img_out[i-ind, j-ind] = np.mean(fenetre)
    
    return img_out.astype(np.uint8)

def filtre_Binomial(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2*ind, cols - 2*ind))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
          
            weights = np.array([1.0])
            for _ in range(N - 1):
                weights = np.convolve(weights, [1, 1])
            weights /= 2**(N - 1)
            result = np.sum(fenetre * np.outer(weights, weights))
            img_out[i-ind, j-ind] = result
    
    return img_out.astype(np.uint8)

def filtre_Median(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            img_out[i-ind, j-ind] = np.median(fenetre)
    
    return img_out.astype(np.uint8)


def filtre_MinMax(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            v_min = np.min(fenetre).astype(np.float64)
            v_max = np.max(fenetre).astype(np.float64)
            if fenetre[ind, ind] > (v_max + v_min) / 2:
                img_out[i-ind, j-ind] = v_max
            else:
                img_out[i-ind, j-ind] = v_min
    
    return img_out




def nearest_neighbor_filter(image, window_size=3):

    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    filtered_image = np.zeros_like(gray_image)

    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            local_window = gray_image[i:i+window_size, j:j+window_size]
            filtered_image[i, j] = np.median(local_window)

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    else:
        return filtered_image

def nagao_filter(image, window_size=3):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    rows, cols = gray_image.shape
    filtered_image = np.zeros_like(gray_image)
    pad = window_size // 2
    padded_image = np.pad(gray_image, pad, mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            local_window = padded_image[i:i+window_size, j:j+window_size]
            filtered_image[i, j] = np.median(local_window)
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    else:
        return filtered_image


def wiener_filter(image, kernel_size=5, noise_var=0.1):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_image = image.astype(np.float32)
    local_mean = uniform_filter(gray_image, size=kernel_size)
    local_sq_mean = uniform_filter(gray_image**2, size=kernel_size)
    local_variance = local_sq_mean - local_mean**2


    noise_var = max(noise_var, 1e-5) 
    filter_response = local_variance / (local_variance + noise_var)
    filtered_image = local_mean + filter_response * (gray_image - local_mean)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image
    
    
def filtre_PHaut(img_in):
    H = 1/16* np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    rows, cols = img_in.shape
    N, M = H.shape
    ind = N // 2
    img_in = img_in.astype(np.float64)
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            M_int = H * fenetre
            img_out[i-ind, j-ind] = np.sum(M_int)
    
    return img_out.astype(np.uint8)
