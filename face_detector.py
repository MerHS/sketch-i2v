import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
class AnimeFaceDetector(object):
    def __init__(self, cascade_file="lbpcascade_animeface.xml"):
        if not os.path.isfile(cascade_file):
            raise RuntimeError("lbpcascade_animeface.xml not found")
        self.cascade = cv2.CascadeClassifier(cascade_file)

    def detect(self, rgb_image):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        
        return faces

    def read_and_draw(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        faces = self.detect(image)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return image

if __name__ == '__main__':
    import sys
    detector = AnimeFaceDetector()

    if len(sys.argv) != 2:
        sys.stderr.write("usage: detect.py <dataset path>\n")
        sys.exit(-1)
    
    dataset_path = Path(sys.argv[1])

    get_count = 0
    multi_count = 0
    drop_count = 0
    total_count = 0

    with open(dataset_path / 'no_face.txt', 'w') as fnf:
        with open(dataset_path / 'face_rect.txt', 'w') as fr:
            train = list((dataset_path / 'rgb_train').iterdir())
            test = list((dataset_path / 'rgb_test').iterdir())

            for fn in tqdm(train + test):
                if total_count % 500 == 0:
                    print (f'total: {total_count} get: {get_count} multi: {multi_count} drop: {drop_count}')

                total_count += 1

                img = cv2.imread(str(fn), cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                faces = detector.detect(img)

                if len(faces) == 0:
                    drop_count += 1
                    fnf.write(f'{fn.stem}\n')
                    continue
                elif len(faces) > 1:
                    multi_count += 1
                
                fx, fy, fw, fh = faces[0]
                fr.write(f'{fn.stem} {fx / w} {fx / w} {fw / h} {fh / h}\n')
                get_count += 1
            
    print (f'total: {total_count} get: {get_count} multi: {multi_count} drop: {drop_count}')