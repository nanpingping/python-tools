#!/usr/bin/ env python -*- coding:UTF-8 -*-
import cv2

vc = cv2.VideoCapture('E:\\videoSlide\\12.mp4')  # 读入视频文件，命名cv
n = 1  # 计数

if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

timeF = 60  # 视频帧计数间隔频率

i = 4709-6
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    if (n % timeF == 0):  # 每隔timeF帧进行存储操作
        i += 1
        print(i)
        cv2.imwrite('E:\\videoSlide\\img\\'+'image_'+str(i)+'.jpg', frame)  # 存储为图像
    n = n + 1
    cv2.waitKey(1)
vc.release()