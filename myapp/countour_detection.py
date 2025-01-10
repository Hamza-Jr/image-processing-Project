import numpy as np
import cv2

def load_image_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    return padded_image, image.shape[1], image.shape[0]

def save_image_data(image, output_path):
    cv2.imwrite(output_path, image)


## gradient methods ------------------------------------------------------------------------------------- ####


def sobel_filter(image):
    Gx = (1/4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = (1/4) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def convolve(img, kernel):
        img_height, img_width = img.shape
        output = np.zeros((img_height - 2, img_width - 2))

        for i in range(1, img_height - 1):
            for j in range(1, img_width - 1):
                region = img[i-1:i+2, j-1:j+2]
                output[i-1, j-1] = np.sum(region * kernel)

        return output
    Gx_img = convolve(image, Gx)
    Gy_img = convolve(image, Gy)

    img_out = np.sqrt(Gx_img**2 + Gy_img**2)
    return img_out


def sobel4Direct_filter(image):
    G0 = (1/4) * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G45 = (1/4) * np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    G90 = (1/4) * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    G135 = (1/4) * np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

    def convolve(img, kernel):
        img_height, img_width = img.shape
        output = np.zeros((img_height - 2, img_width - 2))

        for i in range(1, img_height - 1):
            for j in range(1, img_width - 1):
                region = img[i-1:i+2, j-1:j+2]
                output[i-1, j-1] = np.sum(region * kernel)

        return output
    G0_img = convolve(image, G0)
    G45_img = convolve(image, G45)
    G90_img = convolve(image, G90)
    G135_img = convolve(image, G135)

    img_out = np.maximum(np.maximum(np.abs(G0_img), np.abs(G45_img)), np.maximum(np.abs(G90_img), np.abs(G135_img)))
    return img_out


def prewitt_filter(image):
    Gx = (1/3) * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gy = (1/3) * np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    def convolve(img, kernel):
        img_height, img_width = img.shape
        output = np.zeros((img_height - 2, img_width - 2))

        for i in range(1, img_height - 1):
            for j in range(1, img_width - 1):
                region = img[i-1:i+2, j-1:j+2]
                output[i-1, j-1] = np.sum(region * kernel)

        return output
    Gx_img = convolve(image, Gx)
    Gy_img = convolve(image, Gy)

    img_out = np.sqrt(Gx_img**2 + Gy_img**2)
    return img_out


def gradient_filter(image):
    Gx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
    Gy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

    def convolve(img, kernel):
        img_height, img_width = img.shape 
        output = np.zeros((img_height - 2, img_width - 2))

        for i in range(1, img_height - 1):
            for j in range(1, img_width - 1):
                region = img[i-1:i+2, j-1:j+2]
                output[i-1, j-1] = np.sum(region * kernel)

        return output
    Gx_img = convolve(image, Gx)
    Gy_img = convolve(image, Gy)

    img_out = np.sqrt(Gx_img**2 + Gy_img**2)
    return img_out


def roberts_filter(image):
    Gx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
    Gy = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

    def convolve(img, kernel):
        img_height, img_width = img.shape
        output = np.zeros((img_height - 2, img_width - 2))

        for i in range(1, img_height - 1):
            for j in range(1, img_width - 1):
                region = img[i-1:i+2, j-1:j+2]
                output[i-1, j-1] = np.sum(region * kernel)

        return output
    Gx_img = convolve(image, Gx)
    Gy_img = convolve(image, Gy)

    img_out = np.sqrt(Gx_img**2 + Gy_img**2)
    return img_out


## end gradient methods ------------------------------------------------------------------------------------- ####



## laplacian methods ------------------------------------------------------------------------------------- ####

def filtre_gaussian_laplacian(image):
    Gaus = np.zeros((5, 5))
    for p in range(-2, 3):
        for t in range(-2, 3):
            Gaus[p+2, t+2] = (6/(5*np.sqrt(2*np.pi)))**2 * np.exp(-18*(p**2 + t**2)/25)

    n, m = image.shape
    img = np.zeros((n - 2, m - 2))
    
    for i in range(2, n-2):
        for j in range(2, m-2):
            fenetre1 = image[i-2:i+3, j-2:j+3]
            img[i-1, j-1] = np.sum(fenetre1 * Gaus)
            
    return img


def gaussian_filter(image):
    Gaus2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    n, m = image.shape
    img = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            img[i-1, j-1] = np.sum(fenetre * Gaus2)
    
    return img


def laplacian_variance_filter(image):
    H_V = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    image = variance_filter(image)
    n, m = image.shape
    img = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            img[i-1, j-1] = np.sum(fenetre * H_V)
    
    return img

def laplacian_filter(image):
    H_V = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    n, m = image.shape
    img = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            img[i-1, j-1] = np.sum(fenetre * H_V)
    
    return img


def kirsch_filter(image):
    masquexx = (1/15) * np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    masquexy = (1/15) * np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    masqueyx = (1/15) * np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    masqueyy = (1/15) * np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    
    n, m = image.shape
    img_out = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            f = image[i-1:i+2, j-1:j+2]
            dfxx = np.sum(masquexx * f)
            dfxy = np.sum(masquexy * f)
            dfyx = np.sum(masqueyx * f)
            dfyy = np.sum(masqueyy * f)
            df = np.sqrt(dfxx**2 + dfxy**2 + dfyx**2 + dfyy**2)
            img_out[i-1, j-1] = df
    
    return img_out


def robinson_filter(image):
    masquexx = (1/15) * np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]])
    masquexy = (1/15) * np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
    masqueyx = (1/15) * np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]])
    masqueyy = (1/15) * np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]])
    
    n, m = image.shape
    img_out = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            f = image[i-1:i+2, j-1:j+2]
            dfxx = np.sum(masquexx * f)
            dfxy = np.sum(masquexy * f)
            dfyx = np.sum(masqueyx * f)
            dfyy = np.sum(masqueyy * f)
            df = np.sqrt(dfxx**2 + dfxy**2 + dfyx**2 + dfyy**2)
            img_out[i-1, j-1] = df
    
    return img_out

def variance_filter(image):
    n, m = image.shape
    img_out = np.zeros((n - 2, m - 2))
    
    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            moy = np.mean(fenetre)
            F = (fenetre - moy) ** 2
            img_out[i-1, j-1] = np.mean(F)
    
    return img_out


def local_dispersion_filter(image):
    n, m = image.shape
    img_out = np.zeros((n, m))
    
    # Appliquer le filtre de dispersion locale
    for i in range(1, n-1):
        for j in range(1, m-1):
            fenetre = image[i-1:i+2, j-1:j+2]
            mx = np.max(fenetre)
            mn = np.min(fenetre)
            img_out[i, j] = mx - mn
    
    return img_out




## end laplacian methods ------------------------------------------------------------------------------------- ####


#image1, x_size1, y_size1 = load_image_data("input_1024.jpg")
#print(f"Loaded image: {image1.shape}")
#global_x_size = x_size1 + 2
#global_y_size = y_size1 + 2
#image_op_local = roberts_filter(image1)
#image_op_local2 = sobel_filter(image1)
#image_op_local3 = gradient_filter(image1)
#image_op_local4 = sobel4Direct_filter(image1)
#image_op_local5 = prewitt_filter(image1)
#image_op_local6 = gaussian_filter(image1)
#image_op_local7 = laplacian_filter(image1)
#image_op_local8 = filtre_gaussian_laplacian(image1)
#image_op_local9 = robinson_filter(image1)
#image_op_local10 = kirsch_filter(image1)
#image_op_local11 = variance_filter(image1)
#image_op_local12 = local_dispersion_filter(image1)
#image_op_local13 = laplacian_variance_filter(image1)
#save_image_data(image_op_local, "o.jpg")
#save_image_data(image_op_local2, "o1.jpg")
#save_image_data(image_op_local3, "o2.jpg")
#save_image_data(image_op_local4, "o3.jpg")
#save_image_data(image_op_local5, "o4.jpg")
#save_image_data(image_op_local6, "o5.jpg")
#save_image_data(image_op_local7, "o6.jpg")
#save_image_data(image_op_local8, "o7.jpg")
#save_image_data(image_op_local9, "o8.jpg")
#save_image_data(image_op_local10, "o9.jpg")
#save_image_data(image_op_local11, "o10.jpg")
#save_image_data(image_op_local12, "o11.jpg")
#save_image_data(image_op_local13, "o12.jpg")