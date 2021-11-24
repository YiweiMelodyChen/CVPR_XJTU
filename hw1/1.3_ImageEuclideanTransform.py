import cv2
import numpy as np
import matplotlib.pyplot as plt


def EuclideanTransform(tx, ty, RotationAngle):
    image = cv2.imread('Lenna.png', 1)
    tran_matrix1 = np.float32([[1, 0, tx],
                               [0, 1, ty]])  # 变换矩阵1
    rows, columns = image.shape[:2]
    tran_matrix2 = cv2.getRotationMatrix2D(((columns + tx) / 2, (rows + ty) / 2), RotationAngle, 1)
    # 变换矩阵2，第一个参数是旋转中心，第二个参数是参数的旋转角度，第三个参数是缩放比例
    trans = cv2.warpAffine(image, tran_matrix1, (columns, rows))
    trans = cv2.warpAffine(trans, tran_matrix2, (columns, rows))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(trans)
    plt.show()


EuclideanTransform(100, 50, 45)