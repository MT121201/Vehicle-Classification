# open txt file

import os

path = '/data/its/vehicle_cls/vp3_202307_crop/annotations/train_retrieval.txt'
classes = []
with open(path, 'r') as f:
    # read line by line, take 2nd element of each line
    # split by space, remove \n
    # append to list
    for line in f:
        classes.append(line.split(' ')[1].replace('\n', ''))
classes_name = ('xe sedan', 'xe SUV', 'xe ban tai', 'xe ba gac', 'xe tai nho', 
                 'xe tai lon', 'xe container', 'xe may xuc, may cau, xe lu, xe cho be tong', 
                 'xe 16 cho', 'xe 29-32 cho', 'xe 52 cho' )

#from classes, get class_name
class_name = []
for i in classes:
    # import pdb; pdb.set_trace()
    class_name.append(classes_name[int(i)])
print(class_name)