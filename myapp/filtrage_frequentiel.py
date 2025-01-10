import numpy as np
import math
import cv2

def load_image_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image, image.shape[1], image.shape[0]

def save_image_data(image, output_path):
    cv2.imwrite(output_path, image)
    
    
def passe_bas_butterworth(img_in, D0=30, n=2):
    M, N = img_in.shape
    img_in = img_in.astype(np.float64)
    
    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)
    
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = 1 / (1 + (d / D0) ** (2 * n))
    
    img_fft_filtered = H * img_fft
    img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
    img_out = np.fft.ifft2(img_fft_filtered)
    img_out = np.abs(img_out)
    
    return img_out


def passe_Haut_butterworth(img_in, D0=30, n=3):
    M, N = img_in.shape
    img_in = img_in.astype(np.float64)
    
    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)
    
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = 1 / (1 + (D0 / d ) ** (2 * n))
    
    img_fft_filtered = H * img_fft
    img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
    img_out = np.fft.ifft2(img_fft_filtered)
    img_out = np.abs(img_out)
    
    return img_out


def passe_bas_gaussien(img_in, D0=30):
    M, N = img_in.shape
    img_in = img_in.astype(np.float64)
    
    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)
    
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = np.exp(-d ** 2 / (2 * D0 ** 2))
    
    img_fft_filtered = H * img_fft
    img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
    img_out = np.fft.ifft2(img_fft_filtered)
    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    
    return img_out



def passe_Haut_gaussien(img_in, D0=30):
    M, N = img_in.shape
    img_in = img_in.astype(np.float64)
    
    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)
    
    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = 1-np.exp(-d ** 2 / (2 * D0 ** 2))
    
    img_fft_filtered = H * img_fft
    img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
    img_out = np.fft.ifft2(img_fft_filtered)
    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)
    
    return img_out


def passe_bas_ideal_1demi_1(img_in, D0=30):
    img_in = img_in.astype(np.float64)
    M, N = img_in.shape

    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)

    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if d <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 1 / 2

    img_fft = H * img_fft

    img_fft = np.fft.ifftshift(img_fft)
    img_out = np.fft.ifft2(img_fft)

    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out


def passe_Haut_ideal_1demi_1(img_in, D0=30):
    img_in = img_in.astype(np.float64)
    M, N = img_in.shape

    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)

    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if d <= D0:
                H[u, v] =1 / 2 
            else:
                H[u, v] = 1

    img_fft = H * img_fft

    img_fft = np.fft.ifftshift(img_fft)
    img_out = np.fft.ifft2(img_fft)

    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out


def passe_bas_ideal_0_1(img_in, D0=50):
    img_in = img_in.astype(np.float64)
    M, N = img_in.shape

    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)

    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if d <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0

    img_fft = H * img_fft

    img_fft = np.fft.ifftshift(img_fft)
    img_out = np.fft.ifft2(img_fft)

    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out

def passe_Haut_ideal_0_1(img_in, D0=30):
    img_in = img_in.astype(np.float64)
    M, N = img_in.shape

    img_fft = np.fft.fft2(img_in)
    img_fft = np.fft.fftshift(img_fft)

    H = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            d = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if d <= D0:
                H[u, v] = 0
            else:
                H[u, v] = 1

    img_fft = H * img_fft

    img_fft = np.fft.ifftshift(img_fft)
    img_out = np.fft.ifft2(img_fft)

    img_out = np.abs(img_out)
    img_out = np.clip(img_out, 0, 255).astype(np.uint8)

    return img_out


