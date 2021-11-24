import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Lenna.png', 1)
rows, columns = image.shape[:2]
pts1 = np.float32([[56, 65], [238, 52], [28, 237], [239, 240]])
pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
TransMatrix = cv2.getPerspectiveTransform(pts1, pts2)
Trans = cv2.warpPerspective(image, TransMatrix, (400, 400))
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(Trans)
plt.show()