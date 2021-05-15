#!/usr/bin/env python2
# -*- encoding: utf-8 -*-
"""
Created on Tue Oct  9 14:29:23 2018

@author: zkgao
"""

import os
import cv2
import re
import numpy as np
import scipy
import keras
from keras_preprocessing import image
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from skimage import color
from skimage import img_as_ubyte
import shutil
import pdb

#用来对数据进行增广
def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    if len(x.shape) == 3:
        x = np.rollaxis(x, channel_axis, 0)
        channel_images = [scipy.ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=1,
                mode=fill_mode,
                cval=cval) for x_channel in x]

        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    else:
        x = scipy.ndimage.interpolation.affine_transform(
            x,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval
        )
    return x
#错切
def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                            [0, np.cos(shear), 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(shear_matrix, h, w)
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x   

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                            sat_shift_limit=(-255, 255),
                            val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        img = color.rgb2hsv(image)
        h, s ,v = img[:,:,0],img[:,:,1],img[:,:,2]
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])

        h = h + hue_shift

        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = s + sat_shift

        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = v + val_shift

        img[:,:,0],img[:,:,1],img[:,:,2] = h, s ,v

        image = color.hsv2rgb(img)
    return image

def horizontal_flip_img(img):
    horizontal_flip = img[:,::-1,:]
    return horizontal_flip

def vertical_flip_img(img):
    vertical_flip = img[::-1,:,:]
    return vertical_flip

def random_scale_img(x, s1, s2):
    scale1 = round(np.random.uniform(s1,s2),2)
    scale2 = round(np.random.uniform(s1,s2),2)
    scaled_img = transform.rescale(x, [scale1, scale2])   
    return scaled_img

def random_rotate_img(x, rotate_limit):
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    img_rot = rotate(x, theta)
    return img_rot

def random_shear_img(x, intensity):
    sh = np.random.uniform(-intensity, intensity)
    img_shear = shear(x, sh)
    return img_shear

#判断是否存在中文字符
def contain_zh(word):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    #word = word.decode('utf-8') #python2
    zh = zh_pattern.findall(word)
    return zh

#--------------------------------Main----------------------------#


#遍历所有图片，开始进行增广
images = []
root = './data'
words_data_pools = ['data_img1' , 'data_img2' , 'data_img3']
det = os.path.join(root, 'data_aug')
if os.path.exists(det):
    shutil.rmtree(det)
os.makedirs(det)

dirs = os.listdir(root)

#record.txt记录所有字符个数
list_num = [0] * 127
for files in dirs:
    if files in words_data_pools:
        filespath = os.path.join(root , files)
        filesdir = os.listdir(filespath)
        for imgname in filesdir:
            img = imgname.split('.')[0].split('-' , 1)[1]
            for i in img:
                list_num[ord(i)] += 1
count_i = 0
result = open('./record_indon.txt' , 'w')
for i in list_num:
    if i != 0 :
        savelist = chr(count_i) + " " + str(i) 
        #print(savelist )
        result.write(savelist + '\n')
    count_i += 1
result.close()
#读取record.txt，记录每个汉字对应的个数
labelDict = {}
record = './record_indon.txt'
with open(record) as f:
    lines = f.readlines()
    #pdb.set_trace()
    l = len(lines)
    for i in range(l):
        line = lines[i].strip()
        label_zh = line.split(' ')[0]
        num = int(line.split(' ')[1])
        labelDict[label_zh] = num
        #print("label:{}   num:{}".format(label_zh, num))


for subdir in dirs:
    if subdir in words_data_pools:
        subpath = os.path.join(root, subdir)
        imgfiles = os.listdir(subpath)
        for imgfile in imgfiles:
            if imgfile[-4:] == '.jpg' or imgfile[-4:] == '.bmp':
                imgpath = os.path.join(subpath, imgfile)
                images.append(imgpath)
print('Src images: {0}'.format(len(images)))

print('Start data augmentation...')

#尺度变换->剪切or旋转->对比度变换
rotate_limit = (-1, 1)
intensity = 0.3

for i in range(0,len(images)):
    #pdb.set_trace()
    #加载原始图像
    imgpath = images[i]
    imgfile = os.path.split(imgpath)[1]
    try:
        img = io.imread(imgpath)
    except IOError:
        print(imgpath)
        os.remove(imgpath)
        continue
    
    #根据imgpath确定det路径
    #det = os.path.split(imgpath)[0] + '_aug'
    #if not os.path.exists(det):
    #    os.makedirs(det)
    
    #找到汉字，确定汉字个数
    #pdb.set_trace()
    
    imgname = os.path.splitext(imgfile)[0]
    #print("imgname:{}".format(imgname))
    label_seq = imgname.split('-' , 1)[-1]

    for i in label_seq:
        if i in labelDict:
            label_zh_num = labelDict[i]
        else:
            continue
        threshold = 1000
        if label_zh_num < threshold:
            print("label_zh_num:{}".format(label_zh_num))
        if label_zh_num > threshold and label_zh_num == 0 :
            continue
        else:
            n = int(threshold / label_zh_num)

        
        #开始增广
        for j in range(0,n):
            #随机尺度缩放
            rescaled_img = random_scale_img(img, 0.8, 1.1)
            
            #随机旋转,第一次不旋转
            #if 'detect_LP' in imgpath:
            img_magic = rescaled_img
            #else:
            #   img_magic = random_rotate_img(rescaled_img, rotate_limit)

            aug_index = (5-len(str(j)))*'0' + str(j)
            processed_img_file = 'aug_' + aug_index + '_' + imgfile #用于二次识别
            #pdb.set_trace()
            savePath = os.path.join(det, processed_img_file)
            try:
                io.imsave(savePath, img_magic)
            except ValueError:
                print("Save failed: {0}".format(processed_img_file))
                continue
