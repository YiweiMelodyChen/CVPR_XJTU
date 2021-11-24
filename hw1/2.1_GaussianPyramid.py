import cv2 as cv
import matplotlib.pyplot as plt


# 高斯金字塔
def pyramid_image(image):
    level = 3  # 金字塔的层数
    temp = image.copy()  # 拷贝图像
    pyramid_images = []
    plt.subplot(1, level + 1, 1)
    plt.imshow(image)
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        plt.subplot(1, level + 1, i + 2)
        plt.imshow(dst)
        # cv.imshow("Gaussian" + str(i), dst)
        temp = dst.copy()
    plt.show()
    return pyramid_images


src = cv.imread('Lenna.png')
pyramid_image(src)




