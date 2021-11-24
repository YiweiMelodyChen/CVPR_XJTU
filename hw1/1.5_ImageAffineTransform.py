import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Lenna.png', 1)
rows, columns = image.shape[:2]
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
TranMatrix = cv2.getAffineTransform(pts1, pts2)
# 获得变换矩阵
Trans = cv2.warpAffine(image, TranMatrix, (rows, columns))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(Trans)
plt.show()