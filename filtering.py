import os, random, shutil
from PIL import Image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import csv


def salt(img, n):
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j,i] = 255
        elif img.ndim == 3:
            img[j,i,0]= 255
            img[j,i,1]= 255
            img[j,i,2]= 255
    return img

src = cv2.imread("/root/cabbage/image_cut/lena.png")
img=salt(src, 500)
blur = cv2.blur(img, (5, 5))  #均值滤波
bilateral=cv2.bilateralFilter(img, 0, 100, 5)  #双边滤波
median=cv2.medianBlur(img, 5)  #中值滤波
spatial_radius = 10  # 空间半径
color_radius = 30    # 颜色半径
pyrMeanShift = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius)  #均值漂移
# cv2.imwrite("/root/cabbage/image_cut/img.png", img)
# cv2.imwrite("/root/cabbage/image_cut/blur.png", blur)
# cv2.imwrite("/root/cabbage/image_cut/bilateral.png", bilateral)
# cv2.imwrite("/root/cabbage/image_cut/median.png", median)
# cv2.imwrite("/root/cabbage/image_cut/pyr.png", pyrMeanShift)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)   ##要二值化图像，必须先将图像转为灰度图
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), dtype=np.uint8)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
cv2.imwrite("/root/cabbage/image_cut/binary_image.png", binary_image)
cv2.imwrite("/root/cabbage/image_cut/closing.png", closing)
cv2.imwrite("/root/cabbage/image_cut/opening.png", opening)
