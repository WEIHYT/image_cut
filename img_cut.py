import os, random, shutil
from PIL import Image
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import matplotlib.pyplot as plt
import cv2 as cv
import torch

# class ImageViewer(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.image_path = "/home/xu519/HYT/cabbage/yolov5/VOC_cabbage1/Cam_incubator_front_20240129095701.jpg"  # 图片路径
#         self.image = QPixmap(self.image_path)

#         self.label = QLabel(self)
#         self.label.setPixmap(self.image)
#         self.label.mousePressEvent = self.mouse_click_event

#         self.selected_points = []

#         self.setGeometry(100, 100, self.image.width(), self.image.height())
#         self.setWindowTitle('Image Viewer')
#         # self.show()

#     def mouse_click_event(self, event):
#         if event.button() == Qt.LeftButton:
#             self.selected_points.append(event.pos())
#             if len(self.selected_points) == 4:
#                 print("Selected Points:", self.selected_points)
#                 self.draw_rectangle()

#     def draw_rectangle(self):
#         painter = QPainter(self.image)
#         painter.setPen(QPen(Qt.red, 2))
#         for i in range(3):
#             painter.drawLine(self.selected_points[i], self.selected_points[i + 1])
#         painter.drawLine(self.selected_points[-1], self.selected_points[0])
#         self.label.setPixmap(self.image)
#         self.selected_points = []

# def cut(fileDir,tarDir):
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     filenumber = len(pathDir)
   
#     # 根据指定的区域裁剪图像
#     for image in pathDir:
#         rows,cols,_=image.shape 
#          # 定义要裁剪的区域
#         left = 100   # 左上角x坐标
#         top = 50     # 左上角y坐标
#         right = 300  # 右下角x坐标
#         bottom = 200 # 右下角y坐标
#         # width = right - left
#         # height = bottom - top
#         cropped_image = image.crop((left, top, right, bottom))
#         shutil.move(fileDir + name, tarDir + "/" )
        
def SetPoints(windowname, img):
    """
    输入图片，打开该图片进行标记点，返回的是标记的几个点的字符串
    """
    print('(提示：单击需要标记的坐标，Enter确定，Esc跳过，其它重试。)')
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv.namedWindow(windowname)
    cv.imshow(windowname, temp_img)
    cv.setMouseCallback(windowname, onMouse)
    key = cv.waitKey(0)
    if key == 13:  # Enter
        print('坐标为：', points)
        del temp_img
        cv.destroyAllWindows()
        return str(points)
    elif key == 27:  # ESC
        print('跳过该张图片')
        del temp_img
        cv.destroyAllWindows()
        return
    else:
        print('重试!')
        return SetPoints(windowname, img)


if __name__ == '__main__':
    # app = QApplication(sys.argv)
    # window = ImageViewer()
    # window.show()  # 显示主窗口
    
    image_path ="/home/xu519/HYT/cabbage/yolov5/VOC_cabbage1/Cam_incubator_front_20240129095701.jpg"
    img_cv   = cv.imread(image_path)
    # cv.imshow('img',img_cv)
    # sys.exit(app.exec_())
    # SetPoints('img',img_cv)
    # fileDir = r"/home/xu519/HYT/cabbage/DenseNet/test" + "/"  # 源图片文件夹路径
    # tarDir = r'/home/xu519/HYT/cabbage/DenseNet/test_1'  # 图片移动到新的文件夹路径

    