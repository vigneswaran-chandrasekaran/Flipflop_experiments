import cv2
import os
import numpy as np
images = os.listdir('lstm')

for f in images:
    if '.png' in f:
        img = cv2.imread('lstm\\' + f, 2)
        (_, img) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, np.ones((5, 5)), 5)
        cv2.imwrite('lstm\\thickened' + f, img)
