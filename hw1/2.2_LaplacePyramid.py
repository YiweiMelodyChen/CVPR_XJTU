import cv2 as cv
import matplotlib.pyplot as plt


def pyramid_image(image):
    level = 3   # 金字塔的层数
    temp = image.copy()   # 拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        # cv.imshow("高斯金字塔"+str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def laplian_image(image):
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    plt.subplot(1, level + 1, 1)
    plt.imshow(image)
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            plt.subplot(1, level + 1, i + 2)
            plt.imshow(lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            plt.subplot(1, level + 1, i + 2)
            plt.imshow(lpls)
    plt.show()


src = cv.imread('Lenna.png')
laplian_image(src)