import os, random, shutil
from PIL import Image
import cv2
import numpy as np
import time
from pathlib import Path

#修改文件名
def change(fileDir):
    xlsbpath=fileDir
    os.chdir(xlsbpath) #更改当前路径
    filelist = os.listdir(xlsbpath) # 该文件夹下所有的文件（包括文件夹）
    i=1
    for name in filelist:
        old=xlsbpath+name #旧文件名
        new=xlsbpath+'q'+name #新文件名
        # new=xlsbpath+str(i)+'.txt' #新文件名
        os.rename(old,new) #重命名
        print(new)
        i+=1
    return


def moveimg(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate =0.125  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + "/" )
    return

# def movelabel(file_list, file_label_train, file_label_val):
#     for i in file_list:
#         if i.endswith('.jpg'):
#             # filename = file_label_train + "\\" + i[:-4] + '.xml'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
#             filename = file_label_train + "/" + i[:-4] + '.txt'  # 可以改成xml文件将’.txt‘改成'.xml'就可以了
#             if os.path.exists(filename):
#                 shutil.move(filename, file_label_val)
#                 print(i + "处理成功！")

# 分类
# def classification(fileDir,tarDir,keyword):
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     for img in pathDir:
#         if keyword in img:
#             print(img)
#             shutil.move(fileDir + img, tarDir + "/" )

def cut(fileDir,tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    print(filenumber)
    # 根据指定的区域裁剪图像
    for filename in pathDir :
        image_path=os.path.join(fileDir, filename)
        # image=cv2.imread(image_path)
        image=Image.open(image_path)
        # rows,cols,_=image.shape 
         # 定义要裁剪的区域
        left = 960   # 左上角x坐标
        top = 0   # 左上角y坐标
        right = 1920  # 右下角x坐标
        bottom = 1080 # 右下角y坐标
        # width = right - left
        # height = bottom - top
        img=image.copy()
        cropped_image = img.crop((left, top, right, bottom))
        crop_path = os.path.join(tarDir, filename)
        print(crop_path)
        cropped_image.save(crop_path)
        print(filename)
        # shutil.move(fileDir + image, tarDir + "/" )

def transform(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    print(filenumber)
    i=0
    for filename in pathDir :
        image_path=os.path.join(fileDir, filename)
        img = cv2.imread(image_path)
        rows,cols,_=img.shape
        src_points = np.float32([[1120,320],[1590,336],[1182,1076],[1736,978]])  #左上、右上、左下、右下
        dst_points = np.float32([[0,0],[650,0],[0,rows],[650,rows]])
        trans_matrix = cv2.getPerspectiveTransform(src_points,dst_points) 
        trans_image = cv2.warpPerspective(img,trans_matrix,(650,rows))
        trans_path = os.path.join(tarDir, filename)
        print(trans_path)
        i+=1
        # time.sleep (0.1)
        if cv2.imwrite(trans_path,trans_image) is not None:
            print(i)

def moving_to_path(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    print(filenumber)
    i=0
    for filename in pathDir :
        image_path=os.path.join(fileDir, filename)
        p=Path(image_path)
        txt_path='/root/cabbage/DenseNet/NPK/valid/labels/'+str(p.stem)+'.txt'
        data = []
        with open(txt_path,"r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                line = line.split()
                data.append(line)
                c=data[0][0]
                shutil.move(image_path, tarDir+'/' +str(c)+ "/" )
                print(image_path)


if __name__ == '__main__':
    fileDir = r"/root/cabbage/image_cut/image.v8i.yolov5pytorch/valid/labels" + "/"  # 源图片文件夹路径
    tarDir = r'/root/cabbage/DenseNet/VOC_NPK/val'  # 图片移动到新的文件夹路径
    change(fileDir)
    # moving_to_path(fileDir,tarDir)
    # transform(fileDir,tarDir)
    # moveimg(fileDir, tarDir)
    # classification(fileDir,tarDir,'Leafminer')
    # file_list = os.listdir(tarDir)
    # file_label_train = r"/home/xu519/HYT/cabbage/yolov5-master/VOC/labels/train"  # 源图片标签路径
    # file_label_val = r"/home/xu519/HYT/cabbage/yolov5-master/VOC/labels/val"  # 标签
    #   # 移动到新的文件路径
    # movelabel(file_list, file_label_train, file_label_val)
