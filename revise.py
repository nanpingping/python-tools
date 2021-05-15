import os
from PIL import Image
import re
# 图像文件存储路径
Start_path='C:\\Users\\YY\\Desktop\\1\\'
# 需要调整图片的宽度与高度
pic_width=854
pic_depth=480
# 读取路径下的图片文件
list=os.listdir(Start_path)
#print list
count=0
# 遍历图片文件名
for pic in list:
    # 单个图片的路径
    path=Start_path+pic
    # 输出文件名（含路径）
    print (path)
    # 打开图片文件 图像句柄为im
    im=Image.open(path)
    # 返回im的宽度与高度
    w,h=im.size
    # 大于宽度修改
    if w>pic_width:
        print (pic)
        print ("图片名称为"+pic+"图片被修改")
        # 按比例缩放
        h_new=pic_width*h//w
        w_new=pic_width
        count=count+1
        out = im.resize((w_new,h_new),Image.ANTIALIAS)
        new_pic=re.sub(pic[:-4],pic[:-4],pic)
        #print new_pic
        new_path=Start_path+new_pic
        out.save(new_path)
    # 大于高度修改
    if h>pic_depth:
        print (pic)
        print ("图片名称为"+pic+"图片被修改")
        # 按比例缩放
        w=pic_depth*w//h
        h=pic_depth
        count=count+1
        out = im.resize((pic_width,pic_depth),Image.ANTIALIAS)
        new_pic=re.sub(pic[:-4],pic[:-4],pic)
        #print new_pic
        new_path=Start_path+new_pic
        out.save(new_path)

print ('END')
count=str(count)
print ("共有"+count+"张图片尺寸被修改")