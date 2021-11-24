import cv2
import numpy as np
import matplotlib.pyplot as plt


def Pan(tx,ty):
    image = cv2.imread('Lenna.png', 1)  # 返回np.array类型，读取三维数组，宽度，高度，颜色通道数
    tran_matrix = np.float32([[1, 0, 100],
                              [0, 1, 50]])  # 变换矩阵
    rows, columns = image.shape[:2]  # 读取赌片大小
    trans = cv2.warpAffine(image, tran_matrix, (columns, rows))
    # 三个参数：图片，偏移矩阵，转变后的图片大小
    plt.subplot(121)  # 表示将整个图片窗口分为1行2列，当前位置为1
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(trans)
    plt.show()


Pan(100, 50)