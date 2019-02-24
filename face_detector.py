import os
import cv2

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
        sys.stderr.write("usage: detect.py <filename>\n")
        sys.exit(-1)
    
    image = detector.read_and_draw(sys.argv[1])
    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
