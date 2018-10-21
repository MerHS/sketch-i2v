import os, pathlib, math
import numpy as np
import cv2

def crop_square_image(img, aspect): # aspect = width / height
    h, w = img.shape[:2] # assume that h == w

    if aspect > 1: # real width > height / crop horizontal letterbox
        cut = math.ceil((h - w / aspect) / 2)
        img = img[cut:-cut, :]
    elif aspect < 1: # real width < height / crop vertical letterbox
        cut = math.ceil((w - h * aspect) / 2)
        img = img[:, cut:-cut]
    return img

def get_cropped_image(image_path, aspect):
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    return crop_square_image(img, aspect)

def is_white(vect):
    return np.all(vect > 250)

def make_whiteout_256px_square(img, crop=False):
    is_bgr = img.shape[2] == 3
    height, width = img.shape[:2]

    if height > width:
        pass
    elif height < width:
        pass
    
    return cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
       

    



