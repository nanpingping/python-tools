#coding=utf-8
import os
import cv2
import shutil
import numpy as np

data_root = '/home/disk2/zkchen/database/crnn_datasets/data/unlabel'  #需要rename的图片目录
remove = '/home/disk2/zkchen/database/crnn_datasets/data/drop'  #移除不合适的图片，如背景
label_txt = './results.txt'  #已经识别完的txt文本

imgfiles = os.listdir(data_root)

with open(label_txt) as f:
    lines = f.readlines()
    for line in lines:
        #分割
        number = line.split('.')[0]
        label = line.split('|')[1]
        last_name = line.split('|')[0]
        new_name = number + '-' + label[:-1] + '.jpg'
        remove_name = os.path.join( remove, last_name)
        img_last_name = os.path.join( data_root , last_name)
        img_new_name = os.path.join( data_root , new_name)
        #给标注错误的图片rename，len<2是因为没有图片数据小于2，用来筛选背景图片
        if len(label) < 2 and os.path.exists(img_last_name): 
            #print(img_last_name + ' -> ' + remove_name)
            shutil.move(img_last_name,remove_name)
            continue
        if os.path.exists(img_last_name):
            #print(img_last_name + ' -> ' + img_new_name)
            os.rename(img_last_name , img_new_name)
