import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image_data(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    return padded_image, image.shape[1], image.shape[0]

def save_image_data(image, output_path):
    cv2.imwrite(output_path, image)

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

def filtre_PBas(img_in):
    H = 1/16* np.array([[1,2,1],[2,4,2],[1,2,1]])
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

def filtre_MedHyb(img_in):
    rows, cols = img_in.shape
    img_out = np.zeros((rows - 4, cols - 4))
    
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            fenetre = img_in[i-2:i+3, j-2:j+3]
            croix = np.concatenate([np.diag(fenetre), [fenetre[0, 4], fenetre[1, 3], fenetre[3, 1], fenetre[4, 0]]])
            plus = np.concatenate([fenetre[2, :], fenetre[0:2, 2], fenetre[3:5, 2]])
            centre = fenetre[2, 2]                                                      
            img_out[i-2, j-2] = np.median([np.median(croix), np.median(plus), centre])
    
    return img_out.astype(np.uint8)

def filtre_Med(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            img_out[i-ind, j-ind] = np.median(fenetre)
    
    return img_out.astype(np.uint8)

def filtre_Min(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            img_out[i-ind, j-ind] = np.min(fenetre)
    
    return img_out.astype(np.uint8)

def filtre_Max(img_in, N=3):
    rows, cols = img_in.shape
    ind = N // 2
    img_out = np.zeros((rows - 2, cols - 2))
    
    for i in range(ind, rows - ind):
        for j in range(ind, cols - ind):
            fenetre = img_in[i-ind:i+ind+1, j-ind:j+ind+1]
            img_out[i-ind, j-ind] = np.max(fenetre)
    
    return img_out.astype(np.uint8)

