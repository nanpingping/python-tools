import re                           ##通过训练的txt画出损失函数图#
import matplotlib.pyplot as plt     ##                         #
epoch=[]                            ##          BY:LCQ         #
loss=[]
filename=input("请输入你的训练结果txt：")
#filename=r'C:\Users\99034\Desktop\a.txt'
with open(filename, 'r') as f1:
    lines = f1.readlines()
#line = '1: 926.914551, 926.914551 avg, 0.000000 rate, 1.016110 seconds, 64 images'
for line in lines:
    if line.find('images')+1:
        line.rstrip('\n')
        searchObj = re.search(r'(.*): (.*?) .*', line, re.M | re.I)
        if searchObj:
           # print("searchObj.group() : ", searchObj.group())
            print("searchObj.group(1) : ", searchObj.group(1))
            epoch.append(int(searchObj.group(1)))
            print("searchObj.group(2) : ", searchObj.group(2).rstrip(','))
            loss.append(float(searchObj.group(2).rstrip(',')))
        else:
            print("Nothing found!!")
plt.figure()
plt.xlabel('Items')
plt.ylabel('Loss')
plt.plot(epoch, loss, color='r', linestyle='-.')
plt.show()