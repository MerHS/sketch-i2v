import cv2
import os
from xdog_blend import *
import sketchify
import numpy as np
from pathlib import Path

cur_dir = os.path.dirname(os.path.abspath(__file__))
image_dir_path = Path(os.path.join(cur_dir, 'col'))
result_path = Path(os.path.join(cur_dir, 'res'))

img_list = []
path_list = []
for fn in image_dir_path.iterdir():
    if fn.is_file():
        suf = fn.suffix
        if suf.lower() not in ['.png', 'jpg', 'jpeg']:
            continue
        path_list.append(fn.name)
        img_list.append(cv2.resize(cv2.imread(str(fn)), (512, 512), interpolation=cv2.INTER_LANCZOS4))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

for p_list, chunk in zip(chunks(path_list, 16), chunks(img_list, 16)):
    krs = sketchify.batch_keras_enhanced(chunk)

    for name, k, sketch in zip(p_list, chunk, krs):
        # gray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)

        # sk_high = add_intensity(sketch, 1.6)

        # xdog = get_xdog_image(gray, 0.35, 2.2, 0.95)

        # xdog_blurred = cv2.GaussianBlur(xdog, (5, 5), 1)
        # xdog_residual_blur = cv2.addWeighted(xdog_blurred, 0.75, xdog, 0.25, 0)

        # blend_sketch = np.copy(sk_high)
        # if blend_sketch.shape != xdog_residual_blur.shape:
        #     blend_sketch = cv2.resize(blend_sketch, xdog_residual_blur.shape, interpolation=cv2.INTER_AREA)
        # blended_image = cv2.addWeighted(xdog_residual_blur, 0.25, blend_sketch, 0.75, 0)

        # final_image = sketchify.add_intensity(blended_image, 0.85)
        sketch = add_intensity(sketch, 1.4)
        cv2.imwrite(str(result_path / name), sketch)