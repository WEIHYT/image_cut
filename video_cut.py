import cv2
import os

def apart(video_path, video_name, image_path):
    """
    功能：将视频拆分成图片
    参数：
        video_path：要拆分的视频路径
        video_name：要拆分的视频名字（不带后缀）
        image_path：拆分后图片的存放路径
    """

    # 在这里把后缀接上
    video = os.path.join(video_path, video_name + '.mp4')

    # 提取视频的频率，每１帧提取一个
    frameFrequency = 5

    if not os.path.exists(image_path):
        #如果文件目录不存在则创建目录
        os.makedirs(image_path)

    # 获取视频
    use_video = cv2.VideoCapture(video)

    # 初始化计数器
    count = 0
    n=0
    # 开始循环抽取图片
    print('Start extracting images!')
    while True:
        res, image = use_video.read()
        if(n  % frameFrequency == 0):
            count += 1
        n = n + 1    

        # 如果提取完图片，则退出循环
        if not res:
            print('not res , not image')
            break

        # 将图片写入文件夹中
        cv2.imwrite(image_path + str(count) + '.jpg', image)
        print(image_path + str(count) + '.jpg')

    print('End of image extraction!')
    use_video.release()

if __name__ == '__main__':
    video_path = '/home/xu519/HYT/cabbage/image_cut'
    video_name = 'bokchoy2'
    image_path = '/home/xu519/HYT/cabbage/image_cut/image_cabbage2/'
    apart(video_path, video_name, image_path)
