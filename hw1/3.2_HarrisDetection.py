import numpy as np
from PIL import Image
from skimage import filters
from datetime import datetime
import matplotlib.pyplot as plt

IMAGE_PATH = 'Lenna.png'  # 图片路径
WITH_NMS = False  # 是否非极大值抑制，True/False
k = 0.04  # 响应函数参数k
threshold = 0.01  # 界定阈值

img = Image.open(IMAGE_PATH)
img_gray = img.convert('L')
img_num = np.pad(np.asarray(img_gray, dtype=np.float32), ((1, 1), (1, 1)), 'constant')
h, w = img_num.shape  # padding之后的图像尺寸

# 计算Ix,Iy
grad = np.empty([h, w, 2], dtype=np.float)
grad[:, 1:-1, 0] = img_num[:, 2:] - img_num[:, :-2]  # Ix
grad[1:-1, :, 1] = img_num[2:, :] - img_num[:-2, :]  # Iy

# 计算Ixx,Iyy,Ixy
m = np.empty([h, w, 3], dtype=np.float)
m[:, :, 0] = filters.gaussian(grad[:, :, 0] ** 2, sigma=2)  # Ixx
m[:, :, 1] = filters.gaussian(grad[:, :, 1] ** 2, sigma=2)  # Iyy
m[:, :, 2] = filters.gaussian(grad[:, :, 0] * grad[:, :, 1], sigma=2)  # Ixy
m = [np.array([[m[i, j, 0], m[i, j, 2]],
               [m[i, j, 2], m[i, j, 1]]]) for i in range(h) for j in range(w)]

# 记录一下R计算时耗
start = datetime.now()
D, T = list(map(np.linalg.det, m)), list(map(np.trace, m))
R = np.array([d - k * t ** 2 for d, t in zip(D, T)])
end = datetime.now()
print(end - start)

R_max = np.max(R)
R = R.reshape(h, w)

# 标注角点
record = np.zeros_like(R, dtype=np.int)
img_row = np.pad(np.asarray(img, dtype=np.float32), ((1, 1), (1, 1), (0, 0)), 'constant')
for i in range(1, h - 2):
    for j in range(1, w - 2):
        if WITH_NMS:
            if R[i, j] > R_max * threshold and R[i, j] == np.max(R[i - 1:i + 2, j - 1:j + 2]):
                record[i, j] = 255
                img_row[i, j] = [255, 255, 255]
        else:
            if R[i, j] > R_max * 0.01:
                record[i, j] = 255
                img_row[i, j] = [255, 255, 255]


# 图像展示与保存
res = Image.fromarray(np.uint8(record[1:-1, 1:-1]))
img_row = Image.fromarray(np.uint8(img_row[1:-1, 1:-1]))

plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(res)
plt.subplot(133)
plt.imshow(img_row)
plt.show()