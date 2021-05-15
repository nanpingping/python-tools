# -*- coding: utf-8 -*-
"""
Spyder 编辑器

这是一个临时脚本文件。
"""

import os

print("**********************************************")
path = "C:\\Users\\YY\\Desktop\\images";
# walk方法会返回一个三元组，分别是root、dirs和files。
# 其中root是当前正在遍历的目录路径；dirs是一个列表，包含当前正在遍历的目录下所有
# 的子目录名称，不包含该目录下的文件；files也是一个列表，包含当前正在遍历的
# 目录下所有的文件，
for root, dirs, files in os.walk(path):
    print("目录：" + root)
    for name in files:
        print(name)
        print("文件重命名后：")
        print("@@@@@@@@@@@@@@@@@@@@@")
        NewFileName = name.replace(" ", '');
        NewFileName = os.path.join(root, NewFileName);
        print(NewFileName);
        os.rename(os.path.join(root, name), os.path.join(root, NewFileName))
    #        print("==================")
    #        print(os.path.join(root,name))
    #        print("==================");
    #   NewFileName=name.replace(' ', '');
    #   os.rename(name,NewFileName);
    for name in dirs:
        print("文件目录：");
        print(os.path.join(root, name))