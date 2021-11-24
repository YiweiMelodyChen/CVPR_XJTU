from matplotlib import pyplot as plt
import cv2
import numpy as np


def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])


def orb_detect(image_a, image_b):
    # feature match
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image_a, None)
    kp2, des2 = orb.detectAndCompute(image_b, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(image_a, kp1, image_b, kp2, matches[:100], None, flags=2)

    return bgr_rgb(img3)


def sift_detect(img1, img2, detector='surf'):
    if detector.startswith('si'):
        # print("sift detector......")
        sift = cv2.xfeatures2d.SURF_create()
    else:
        # print("surf detector......")
        sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return bgr_rgb(img3)


def Spin(image, RotationAngle):
    # image = cv2.imread('Lenna.png', 1)
    rows, columns = image.shape[:2]
    trans_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), RotationAngle, 1)
    # 第一个参数是旋转中心，第二个参数是参数的旋转角度，第三个参数是缩放比例
    trans = cv2.warpAffine(image, trans_matrix, (columns, rows))
    '''plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(trans)
    plt.show()'''
    return trans


if __name__ == "__main__":
    '''image_a = cv2.imread('londontower.jpg')
    img_a = cv2.resize(image_a, (512, 256), interpolation=cv2.INTER_CUBIC)
    trans = Spin(image_a, 90)
    img = sift_detect(img_a, trans)
    # img = sift_detect(img_a, img_b)'''
    image1 = cv2.imread('flower1.jpg')
    image2 = cv2.imread('flower2.jpg')
    img = sift_detect(image1, image2)
    plt.imshow(img)
    plt.show()