import cv2
import matplotlib.pyplot as plt


def Spin(RotationAngle):
    image = cv2.imread('Lenna.png', 1)
    rows, columns = image.shape[:2]
    trans_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), RotationAngle, 1)
    # 第一个参数是旋转中心，第二个参数是参数的旋转角度，第三个参数是缩放比例
    trans = cv2.warpAffine(image, trans_matrix, (columns, rows))
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(trans)
    plt.show()


Spin(45)