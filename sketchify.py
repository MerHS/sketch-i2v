import cv2
import numpy as np
from keras.models import load_model
from helper import *

mod = load_model('mod.h5')

def dog(img, size=(0,0), k=1.6, sigma=0.5, gamma=1):
    img1 = cv2.GaussianBlur(img, size, sigma)
    img2 = cv2.GaussianBlur(img, size, sigma * k)
    return (img1 - gamma * img2)

def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
    aux = dog(img, sigma=sigma, k=k, gamma=gamma) / 255
    for i in range(0, aux.shape[0]):
        for j in range(0, aux.shape[1]):
            if(aux[i, j] < epsilon):
                aux[i, j] = 1*255
            else:
                aux[i, j] = 255*(1 + np.tanh(phi * (aux[i, j])))
    return aux


def get_keras_enhanced(img):
    from_mat = img
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0

    if (width > height):
        if width != 512:
            from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        if height != 512:
            from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512

    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])

    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    line_mat = np.amax(line_mat, 2)

    keras_enhanced = to_keras_enhanced(line_mat)

    return keras_enhanced


def get_keras_high_intensity(img, gamma=1.7):
    keras_img = get_keras_enhanced(img)
    return (255 * ((keras_img / 255.0) ** gamma)).astype(np.uint8)


def get_xdog_image(img, sigma=0.4, k=2.5, gamma=0.95, epsilon=-0.5, phi=10**9):
    xdog_image = xdog(img, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi).astype(np.uint8)
    return xdog_image

    
def get_sketch(img, gamma=1.7, degamma=(1/1.5)):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keras_image = get_keras_high_intensity(img, gamma=gamma)
    xdog_image = get_xdog_image(gray_image)

    xdog_blurred = cv2.GaussianBlur(xdog_image, (5, 5), 1)
    xdog_residual_blur = cv2.addWeighted(xdog_blurred, 0.75, xdog_image, 0.25, 0)
    blended_image = cv2.addWeighted(xdog_residual_blur, 0.25, keras_image, 0.75, 0)
    degamma_image = 255 * ((blended_image / 255.0) ** degamma)
    degamma_image = degamma_image.astype(np.uint8)

    return degamma_image