import os
det = '.'
pools = ['database1']

files = os.listdir(det)

for filename in files:
    if filename in pools:
        imgfile = os.listdir(filename)
        for img in imgfile:
            name = img.split('.')[0].split('-')[1]
            if name == "255IAB":
                new_name = img.split('-')[0] + "-2551IAB." +img.split('.')[1]
                last_img = os.path.join(det , filename ,img)
                new_img = os.path.join(det , filename ,new_name)
                os.rename(last_img,new_img)