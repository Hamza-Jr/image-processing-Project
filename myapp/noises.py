import numpy as np
import cv2

# Gaussian Noise
def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype('float64')
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Salt and Pepper Noise
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    
    # Salt noise
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255  # For grayscale images, salt should be white (255)
    
    # Pepper noise
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0  # For grayscale images, pepper should be black (0)

    return noisy

# Poisson Noise
def add_poisson_noise(image):
    noisy = np.random.poisson(image).astype(np.uint8)
    return noisy

# Speckle Noise
def add_speckle_noise(image, var=0.01):
    row, col = image.shape
    gauss = np.random.normal(0, var**0.5, (row, col))
    noisy = np.clip(image + image * gauss, 0, 255).astype(np.uint8)
    return noisy

# Impulse Noise
def add_impulse_noise(image, prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    num_impulse = int(total_pixels * prob)

    # Impulse noise
    coords = [np.random.randint(0, i-1, num_impulse) for i in image.shape]
    noisy[coords[0], coords[1]] = np.random.choice([0, 255], num_impulse)

    return noisy

# Quantization Noise
def add_quantization_noise(image, bits=8):
    q = 2**bits
    noisy = (np.floor(image / (256 / q)) * (256 / q)).astype(np.uint8)
    return noisy

# Periodic Noise
def add_periodic_noise(image, frequency=50):
    row, col = image.shape
    x = np.arange(col)
    y = np.arange(row)
    X, Y = np.meshgrid(x, y)
    noise = 127 * np.sin(2 * np.pi * frequency * X / col + 2 * np.pi * frequency * Y / row)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy



def image_original(image_path):
    pass

def image_original_grise(image_path):
    pass



import cv2

def image_original_grise2(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return grayscale_image




def image_original_grise(image):
    
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        grayscale_image = image
    else:
        raise ValueError("The image has an unexpected number of channels.")
    
    return grayscale_image
