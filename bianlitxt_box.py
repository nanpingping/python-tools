import sys
import json
import os

train_path=[]
train_path_txt='E:\\LCQ\\graduation_project\\image\\train_path.txt'
for line in open(train_path_txt):   
    print(line.replace(".jpg",".txt"))
    train_path.append(line.replace(".jpg",".txt"))
box=[]
for i in train_path:
    for line in open(i[:-1]):  
        print(line)   
        boxx=line.split(" ")
        w=(float(boxx[3]))*845
        h=(float(boxx[4]))*480
        box.append(str(w)+" "+str(h))
txt_object = open("yy_box.txt", "w+") 
for i in box:      
    txt_object.write(i+"\n")
txt_object.close()