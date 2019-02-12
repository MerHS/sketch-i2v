import argparse
import cv2
import numpy as np
from pathlib import Path
import sketchify.sketchify as sk
from sketchify.crop import make_square
from sketchify.xdog_blend import get_xdog_image, add_intensity
from skimage import morphology

# from sketchify.sketchify import get_keras_high_intensity

def valone(img, l_stand):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv[:,:,0] = l_stand
    return cv2.cvtColor(hsv, cv2.COLOR_LAB2BGR)

def valsat(img, val, l_stand):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,val] = l_stand
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def valrevert(img, org):
    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv2 = cv2.cvtColor(org, cv2.COLOR_BGR2LAB)
    hsv1[:,:,0] = hsv2[:,:,0]
    return cv2.cvtColor(hsv1, cv2.COLOR_LAB2BGR)

def quant(img, K=9):
    Z = img.reshape((-1,1))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("file_name")
    p.add_argument("--train_file", default="model.pth")

    args = p.parse_args()

    if not Path(args.file_name).exists():
        raise Exception(f"{args.file_name} does not exists.")

    ximg = cv2.imread(args.file_name)
    # ximg = cv2.resize(ximg, None, fx=0.15, fy=0.15)
    # cv2.imshow("color", ximg)
    gray = cv2.cvtColor(ximg, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    sketch = sk.get_keras_enhanced(ximg)

    # cv2.imshow("sketch", sketch)
    sk_high = add_intensity(sketch, 1.6)
    #cv2.imshow("intensity", sk_high)

    xdog = get_xdog_image(gray, 0.35, 2.2, 0.95)
    # cv2.imshow("xdog", xdog)

    xdog_blurred = cv2.GaussianBlur(xdog, (5, 5), 1)
    xdog_residual_blur = cv2.addWeighted(xdog_blurred, 0.75, xdog, 0.25, 0)
    #cv2.imshow("xdog_res_blur", xdog_residual_blur)

    blend_sketch = np.copy(sk_high)
    print(xdog_residual_blur.shape, blend_sketch.shape)
    if blend_sketch.shape != xdog_residual_blur.shape:
        blend_sketch = cv2.resize(blend_sketch, xdog_residual_blur.shape, interpolation=cv2.INTER_AREA)
    print(xdog_residual_blur.shape, blend_sketch.shape)
    blended_image = cv2.addWeighted(xdog_residual_blur, 0.25, blend_sketch, 0.75, 0)

    #cv2.imshow("blend", blended_image)
    final_image = sk.add_intensity(blended_image, 0.8)
    # cv2.imshow("final", blended_image)

    cv2.imwrite('color.png', ximg)
    cv2.imwrite('sketch.png', sketch)
    cv2.imwrite('xdog.png', xdog)
    cv2.imwrite('final.png', blended_image)

    #cv2.waitKey(0)
