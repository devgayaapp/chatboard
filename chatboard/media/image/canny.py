import cv2
import numpy as np
from components.image.image import Image

def canny(image: Image, low_threshold = 100, high_threshold = 200):
    canny_arr_img = cv2.Canny(image.np_arr, low_threshold, high_threshold)
    canny_arr_img = canny_arr_img[:, :, None]
    canny_arr_img = np.concatenate([canny_arr_img, canny_arr_img, canny_arr_img], axis=2)
    canny_image = Image.from_numpy(canny_arr_img)
    return canny_image