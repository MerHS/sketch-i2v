import argparse
import cv2
import numpy as np
from pathlib import Path
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
    p.add_argument("--sketch", action='store_true')
    p.add_argument("--blend", action='store_true')
    args = p.parse_args()

    if not Path(args.file_name).exists():
        raise Exception(f"{args.file_name} does not exists.")

    img = cv2.imread(args.file_name)
    ximg, _, _ = make_square(img, size=512, extend=(True, True))
    cv2.imshow("main", ximg)
    img = add_intensity(ximg, 1.9)
    cv2.imshow("intensity", img)
    #sketch = get_keras_high_intensity(img, 2)
    xdog = get_xdog_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0.45, 2.2, 0.99)
    # lab = valsat(img, 2, 200)
    #bent = cv2.addWeighted(sketch, 0.1, lab, 0.9, 0)
    #cv2.imshow("sketch", sketch)
    cv2.imshow("xdog", xdog)
    mask = xdog < 10
    mask = morphology.remove_small_objects(mask, min_size=4, connectivity=2)
    cleaned = np.copy(xdog)
    cleaned[mask==False] = 255
    cv2.imshow("cluster", cleaned)
    cv2.imwrite('cleaned.png', cleaned)
    #cv2.imshow("add", bent)
    # bil3 = cv2.bilateralFilter(img, 9, 80, 120)
    # bil1 = cv2.bilateralFilter(quant(bil3, 12), 9, 80, 120)
    
    # cv2.imshow("bil1", bil3)
    # cv2.imshow("bil2", quant(bil3, 12))
    # cv2.imshow("bil3", bil1)
    # cv2.imshow("bil4", quant(bil1, 12))

    cv2.waitKey(0)
