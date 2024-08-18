import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from datetime import datetime


lettuce_DenseNet_path = '/root/cabbage/yolov5/runs/train/exp/results.csv'
lettuce_MSFF_DenseNet_path='/root/cabbage/yolov5/runs/train/leaves_100_coslr_lr0/results.csv'
# lettuce_MSFF_DenseNet_FL_path='/root/cabbage/work/lettuce_MSFF_DenseNet_FL/train.csv'

x_axis_data, y_axis_data1,y_axis_data2 = [], [],[]
with open(lettuce_DenseNet_path) as f:
    reader = csv.reader(f)
    header_row = next(reader)  # 返回文件的下一行，在这便是首行，即文件头
    for row in reader:
        # x = datetime.strptime(row[0], '%Y-%m-%d')
        y1 =float(row[7]) 
        # x_axis_data.append(x)
        y_axis_data1.append(y1)

with open(lettuce_MSFF_DenseNet_path) as f:
    reader1 = csv.reader(f)
    header_row1 = next(reader1)  # 返回文件的下一行，在这便是首行，即文件头
    for row in reader1:
        x = float(row[0])
        y2 = float(row[7])
        x_axis_data.append(x)
        y_axis_data2.append(y2)  

# with open(lettuce_MSFF_DenseNet_FL_path) as f:
#     reader2 = csv.reader(f)
#     header_row2 = next(reader2)  # 返回文件的下一行，在这便是首行，即文件头
#     for row in reader2:
#         y3 = float(row[2])
#         y_axis_data3.append(y3)  

# y_axis_data1=y_axis_data1[:219]
# print(y_axis_data1)
#epoch,acc,loss,val_acc,val_loss
# x_axis_data =[]
# y_axis_data1 = [68.72,69.17,69.26,69.63,69.35,70.3,66.8]
# y_axis_data2 = [71,73,52,66,74,82,71]
# y_axis_data3 = [82,83,82,76,84,92,81]

        
#画图 
plt.plot(x_axis_data, y_axis_data1,  alpha=1, linewidth=1, label='Growth Stage Recognition Model')#'
plt.plot(x_axis_data, y_axis_data2,  alpha=1, linewidth=1, label='Leaf Type Recognition Model')
# plt.plot(x_axis_data, y_axis_data3,  alpha=0.5, linewidth=1, label='MSFF-DenseNet-FL')
# plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='acc')

 
plt.legend()  #显示上面的label
plt.xlabel('train_epochs')
plt.ylabel('mAP_0.5:0.95')#accuracy
plt.savefig('/root/cabbage/image_cut/mAP_0.5:0.95.png')
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()
