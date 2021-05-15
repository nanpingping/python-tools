#coding=utf-8
import os
import cv2
import numpy as np

subpath = '.'

imgfiles = os.listdir(subpath)
ind=0

for imgfile in imgfiles:
    #读入原始图像
    if imgfile[-4:] == ".jpg":
        img = cv2.imread(imgfile)
        channels = img.shape[2]
        width = img.shape[1]
        height = img.shape[0]
        for row in range(height):
            for col in range(width):
                for c in range(channels):
                    pv = img[row, col, c]
                    img[row, col, c] = 255-pv
        cv2.imwrite("new_" + imgfile , img)
        print(imgfile + "Done!")