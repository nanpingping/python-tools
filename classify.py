#-*- coding:utf-8 -*-
import os
import sys
import numpy as np
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

caffe_root='/home/disk_E/jyf/caffe_ocr_for_linux/' #修改成你的Caffe项目路径
sys.path.append(caffe_root+'python')
import caffe
import time
from pylab import *
from ctcDecoder.BeamSearch import ctcBeamSearch
from ctcDecoder.PrefixSearch import ctcPrefixSearch

import pdb

def print_list(list):
    res=''
    for x in list:
        res += x
    print(res)


#设置国家
IndonCard = 1

#设置配置参数
GRAY = 0 #输入是否是灰度图像
TEST = 1 #是否测试评估功能，否则为输出预测结果功能

#设置为CPU模式
GPU = 1
if GPU:
    caffe.set_mode_gpu()
    caffe.set_device(0)
else:
    caffe.set_mode_cpu()

#根据国家配置

if IndonCard == 1:
    # 修改成你的deploy.prototxt文件路径
    '''
    model_name = 'crnn'
    model_def = os.path.join('../models', model_name, 'deploy.prototxt') 
    model_weights = '../caffemodel/IndonCard_crnn_iter_50000.caffemodel' # 修改成你的caffemodel文件的路径
    '''
    '''
    model_name = 'dense_crnn_v3'
    model_def = os.path.join('../models', model_name, model_name + '_deploy.prototxt') 
    model_weights = '../caffemodel/dense_crnn_v3_iter_100000.caffemodel'
    '''

    model_name = 'crnn_v2'
    model_def = os.path.join('../models', model_name, 'deploy.prototxt') 
    model_weights = '../caffemodel/crnn_v2_iter_40000.caffemodel'
    

    resized_w = 512 #224
    blank_label_idx = 0
   
    words = '/home/disk_E/jyf/caffe_ocr_for_linux/data/video_text/cfg/words_indon.txt' #indon
    #test_txt = '/home/disk_E/jyf/caffe_ocr_for_linux/data/video_text/train_val_test/indon/val.txt'  #imgpath label
    test_txt = '/home/disk_E/jyf/caffe_ocr_for_linux/data/video_text/train_val_test/indon/all.txt'  #imgpath label

#网络初始化
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                

#预设置

if not GRAY:
    mean = [104, 117, 123]
    #mean = [0, 0, 0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # 通道变换，例如从(530,800,3) 变成 (3,530,800)
    #transformer.set_mean('data', np.load(npyMean).mean(1).mean(1)) #如果你在训练模型的时候没有对输入做mean操作，那么这边也不需要
    transformer.set_mean('data', np.array(mean, dtype=np.float))
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformer.set_input_scale('data', 0.017)
    #transformer.set_input_scale('data', 0.00392157)
    
else:
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # 通道变换，例如从(530,800,3) 变成 (3,530,800)
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_input_scale('data', 0.017)
    #transformer.set_input_scale('data', 0.00392157)


    

#取出标签文档
char_set = []
classes = ''
with open(words) as f:
    line = f.readline()
    while line:
        line = line.strip().decode('utf-8')
        char_set.append(line)
        if line != '#':
            classes += line
        line = f.readline()


if TEST:
    #对测试集进行评估
    count = 0 #记录测试图片
    right = 0 #分类正确个数
    error = 0 #分类错误个数
    total_time = 0.0 #处理的总时间

    model_name = model_weights.split('/')[-1][:-11]
    test_txt_name = test_txt.split('/')[-1][:-4]

    result_txt_name = 'result_' + test_txt_name + '_' + model_name + '.txt'
    error_txt_name = 'error_' + test_txt_name + '_' + model_name + '.txt'
    
    error_txt = open('result/' + error_txt_name, 'w') # 识别错误的结果写入error_txt
    result_txt = open('result/' + result_txt_name, 'w') # 写入识别结果

    #冒号数量统计
    num_all = 0
    num_next = 0

    with open(test_txt) as f:
        lines = f.readlines()
        l = len(lines)
        for i in range(l):
            line = lines[i].strip().decode('utf-8')
            line_list = line.split(' ')
            
            #pdb.set_trace()
            imgpath = line_list[0]
            gt_label_list = line_list[1:]
            
            try:
                gt_label = "".join(char_set[int(x)] for x in gt_label_list if int(x) != blank_label_idx)
            except UnicodeEncodeError:
                print(imgpath)
                continue
            
            if not GRAY:
                try:
                    image = caffe.io.load_image(imgpath) 
                except IOError:
                    print(imgpath)
                    continue
                transformed_image = transformer.preprocess('data', image)
                net.blobs['data'].data[...] = transformed_image
                net.blobs['data'].reshape(1, 3, 32, resized_w)
            else:
                try:
                    image = caffe.io.load_image(imgpath, 0) #0 - 灰度图 1 - 彩色图
                except IOError:
                    print(imgpath)
                    continue
                transformed_image = transformer.preprocess('data', image)
                net.blobs['data'].data[...] = transformed_image
                net.blobs['data'].reshape(1, 1, 32, resized_w)

            start = time.time()
            
            output = net.forward()
            end = time.time()
            output_prob = net.blobs['probs'].data
            #pdb.set_trace()
            #初始结果
            ori_res = ''
            time_step = output_prob.shape[0]
            labelsNum = output_prob.shape[2]
            reshape_prob = np.reshape(output_prob, (time_step, labelsNum))
            
            for i in range(time_step):
                data = output_prob[i,:,:]
                index = data.argmax()
                ori_res += char_set[index]
                
            #获取最终结果
            blank_label = '#'
            prev = '-1'
            res = ''
            for x in ori_res:
                if x != blank_label and x != prev:
                    res += x
                prev = x
            
            
            total_time += (end -start)
            #pdb.set_trace()
            #对比预测结果和实际结果,这里对比两种，一种是汉字不同，一种是字符不同
            if res == gt_label:
                right += 1
            else:
                error += 1
                
                error_txt.write(imgpath + '|' + res + '\n')
                #替换字符输入'_'':' '%''/'
                #error_txt.write(imgpath + '|' + 
                #res.replace('_' , ':').replace('%' , '/') + '\n')
            '''
            #######结果冒号预测框的所有值
            if '_' in gt_label or '_' in res :
                print(imgpath + "|" + res + "   " + ori_res) 

            #######
            '''

            #每隔100或者1000打印
            count += 1
            if count >0 and count < 1000 and count % 100 ==0:
                print(count)
            elif count >= 1000 and count < 10000 and count % 1000 ==0:
                print(count)
            elif count >=10000 and count % 10000 == 0:
                print(count)

    #计算平均时间
    avgtime = total_time * 1000 / count
    accuracy = right * 100.0 / float(count)
    
    #打印输出
    print("Total classify {0} images:".format(count))
    print("right: {0}".format(right))
    print("error: {0}".format(error))
    print("Accuarcy is {0}%".format(accuracy))
    print('Avg classification time is {0} ms'.format(avgtime))


    #写入输出
    result_txt.write("Total classify {0} images:\n".format(count))
    result_txt.write("right: {0}\n".format(right))
    result_txt.write("error: {0}\n".format(error))
    result_txt.write("Accuarcy is {0}%\n".format(accuracy))
    result_txt.write('Avg classification time is {0} ms'.format(avgtime))
else:
    #record the use time
    use_time = 0.0
    avg_time = 0.0
    
    #ctcDecoder 
    BeamSearch = 0
    BestPath = 1
    PrefixSearch = 0
    
    #测试图片路径
    data_root = '/home/disk_D/database/video_text/unlabel'
    #data_root = '/home/disk_D/database/video_text/test'
    imgfiles = os.listdir(data_root)
    
    #输出结果
    result = open('./results.txt', 'w')
    #pdb.set_trace()
    
    for imgfile in imgfiles:
        imgpath = os.path.join(data_root, imgfile)

        if not GRAY:
            try:
                image = caffe.io.load_image(imgpath) 
            except IOError:
                print(imgpath)
                continue
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            net.blobs['data'].reshape(1, 3, 32, resized_w)
        else:
            try:
                image = caffe.io.load_image(imgpath, 1) #0 - 灰度图 1 - 彩色图
            except IOError:
                print(imgpath)
                continue
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image
            net.blobs['data'].reshape(1, 1, 32, resized_w)
        
        start = time.time()
        output = net.forward()
        end = time.time()
        
        output_prob = net.blobs['probs'].data
        time_step = output_prob.shape[0]
        labelsNum = output_prob.shape[2]
        reshape_prob = np.reshape(output_prob, (time_step, labelsNum))

        if BeamSearch:
            print('BeamSearch:')
            result = ctcBeamSearch(reshape_prob, classes, None, 5)
            print(result)
        
        if PrefixSearch:
            print('PrefixSearch:')
            result = ctcPrefixSearch(reshape_prob, classes)
            print(result)
        if BestPath:
            #print('BestPath')
            #获取初始结果
            #初始结果
            ori_res = ''
            for i in range(time_step):
                data = output_prob[i,:,:]
                index = data.argmax()
                ori_res += char_set[index]
            #pdb.set_trace()

            #获取最终结果
            blank_label = '#'
            prev = '-1'
            res = ''
            for x in ori_res:
                if x != blank_label and x != prev:
                    res += x
                prev = x

    #######预测框的所有值

            print(imgfile + "|" + res + "   " + ori_res) 

    #######
        #pdb.set_trace()
        result.write(imgfile + '|' + res + '\n')
        use_time += (end - start)
    avgtime = use_time * 1000 / len(imgfiles)
    print("Avg Prediction uses {0} ms.".format(use_time))