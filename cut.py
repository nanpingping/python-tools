#coding:utf-8
import os
import cv2
det = 'C:/Users/YY/Desktop/zimushuj'
out = 'C:/Users/YY/Desktop/output'
listdir = os.listdir(det)
number = 0
for imgname in listdir:
    list_all = []
    if imgname.split('.')[1] == 'jpg' :
        imgpath = os.path.join(det ,imgname)
        txtpath = os.path.join(det ,imgname.split('.')[0] + '.txt')
        img = cv2.imread(imgpath)
        with open(txtpath, "r", encoding = "utf-8") as f:
            list_all = f.read().splitlines()
        for line in list_all:
            list_num = line.split(' ')
            y_max = int((float(list_num[2]) + float(list_num[4])/2.0) * img.shape[0])
            y_min = int((float(list_num[2]) - float(list_num[4])/2.0) * img.shape[0])
            x_max = int((float(list_num[1]) + float(list_num[3])/2.0) * img.shape[1])
            x_min = int((float(list_num[1]) - float(list_num[3])/2.0) * img.shape[1])
            cropped = img[y_min:y_max, x_min:x_max] # 裁剪坐标为[y0:y1, x0:x1]
            savepath = os.path.join(out ,"{}".format(number) + ".jpg")
            number += 1
            cv2.imwrite(savepath ,cropped)