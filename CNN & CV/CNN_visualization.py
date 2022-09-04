import cv2
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

img = cv2.imread('gugong.jpg')

# imread 读的彩色图按照BGR像素储存，转换为RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.show()

# 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize
grayImage = grayImage.astype('float32') / 255
plt.imshow(grayImage, cmap='gray')
plt.show()

filter_vals = np.array([
    [-1, -1, 1, 1],
    [-1, -1, 1, 1],
    [-1, -1, 1, 1],
    [-1, -1, 1, 1]
])

print('Filter shape: ', filter_vals.shape)

# 定义滤波器
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# 可视化filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y, x), color='white' if filters[i][x][y] < 0 else 'black')

