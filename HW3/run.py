from typing import List
import numpy as np

from cv2 import cv2

def sift(img_path: str, str_path: str):
    img = cv2.imread(img_path)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, _) = descriptor.detectAndCompute(img, None)
    cv2.drawKeypoints(img, kps, img, (0,255,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(str_path, img)

if __name__=='__main__':
    img_path = 'data/1.jpg'
    str_img = 'results/1.jpg'
    sift(img_path, str_img)
