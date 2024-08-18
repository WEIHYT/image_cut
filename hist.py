import cv2
import numpy as np
from matplotlib import pyplot as plt

#通过OpenCV读取图片信息
img = cv2.imread('/root/cabbage/yolov5/幼苗期/1_jpg.rf.8ff73b6f6b4f1334830151396c2715ce.jpg')
# BGR图转为HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 提取hsv中H通道数据
# H, S, V = cv2.split(hsv)
h = hsv[:, :, 0].ravel()
s= hsv[:, :, 1].ravel()
v= hsv[:, :, 2].ravel()
# hist_h=cv2.calcHist([H],[0],None,[180],[0,180])
# hist_s= cv2.calcHist([S],[0],None,[256],[0,256])
# hist_v=cv2.calcHist([V],[0],None,[256],[0,256])
# 直方图显示
plt.hist(h ,180, [0, 180])
plt.savefig('/root/cabbage/image_cut/calculate/h.jpg')
# plt.hist(s, 256, [0, 256])
# plt.savefig('/root/cabbage/image_cut/2/s.jpg')
# plt.hist(v, 256, [0, 256])
# plt.savefig('/root/cabbage/image_cut/2/v.jpg')
plt.show()
