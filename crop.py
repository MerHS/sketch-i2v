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

def make_256px_square(img, crop=False, extend=None):
    """
    extend borders with white pixel or crop image to make it square
    
    crop: boolean -- crop longer part and make square (default: False)
    extend: None | (boolean, boolean) -- extend (left or top, right or bottom) part to white
    
    return: (image, is_cropped: boolean, is_extended: None | (boolean, boolean))
    """
    is_bgr = img.shape[2] == 3
    height, width = img.shape[:2]
    cropped = False

    if crop:
        if height > width: # crop top & bottom
            margin_bottom = (height - width) // 2
            margin_top = (height - width) - margin_bottom
            img = img[margin_top:-margin_bottom]
        elif width > height: # crop left & right
            margin_left = (width - height) // 2
            margin_right = (width - height) - margin_left
            img = img[:, margin_left:-margin_right]

        resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        return (resized_img, True, None) 

    # set extend
    if height > width: # extend horizontally
        lw, rw = is_white(img[:, 0]), is_white(img[:, -1])
        if lw and rw:
            pass
        elif lw:
            pass
        elif rw:
            pass
        else:
            cropped = True

    elif height < width: # extend vertically
        tw, bw = is_white(img[0]), is_white(img[-1])

    diff = math.abs(height - width)
    lx, ly = diff // 2, diff - (diff // 2)

    if height > width: # extend (left, right)    
        if is_bgr:
            zx, zy = np.zeros((height, lx, 3)), np.zeros((height, ly, 3))
        else:
            zx, zy = np.zeros((height, lx)), np.zeros((height, ly))

        if extend == (True, True):
            img = cv2.append(zx, img, axis=1)
            img = cv2.append(img, zy, axis=1)
        elif extend == (False, True):
            img = cv2.append(img, zx, axis=1)
            img = cv2.append(img, zy, axis=1)
        elif extend == (True, False):
            img = cv2.append(zx, img, axis=1)
            img = cv2.append(zy, img, axis=1)

    elif width > height: # extend (top, bottom)
        if is_bgr:
            zx, zy = np.zeros((lx, width, 3)), np.zeros((ly, width, 3))
        else:
            zx, zy = np.zeros((lx, width)), np.zeros((ly, width))

        if extend == (True, True):
            img = cv2.append(zx, img, axis=0)
            img = cv2.append(img, zy, axis=0)
        elif extend == (False, True):
            img = cv2.append(img, zx, axis=0)
            img = cv2.append(img, zy, axis=0)
        elif extend == (True, False):
            img = cv2.append(zx, img, axis=0)
            img = cv2.append(zy, img, axis=0)
    
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
       
    return (resized_img, cropped, None)

    



