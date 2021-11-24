import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname = 'londontower.jpg'
image = cv2.imread(imgname, 1)
img = cv2.imread(imgname, 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img, None)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
point_img = cv2.drawKeypoints(img, kp, img, color=(255, 0, 255))
plt.subplot(131)
plt.imshow(image)
plt.title('Original')
plt.subplot(132)
plt.imshow(gray_img)
plt.title('Gray')
plt.subplot(133)
plt.imshow(point_img)
plt.title('SIFT')
plt.show()