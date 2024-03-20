import os
from os import path
import cv2 as cv
import random
list_path = "/home/manhnguyen/new/co-separation/hai_sample_hfd5/val.txt"
img_path = "/ext_data2/manhnh/MUSIC_dataset/output_img/"
kind = ['duet', 'MUSIC_solo2']
all_file = []  
id_list = []     
for filename in kind: #filename is duet and solo
    for class_name in os.listdir(img_path + filename): #class_name is erhu, flute, ...
        for idvid_name in os.listdir(img_path + filename + "/" + class_name): #idvid is 1j....
            all_file.append(img_path + filename + "/" + class_name + "/" + idvid_name)
            id_list.append(idvid_name)
                        
imglist = []
img_txt = open(list_path, "r")
img_PATH = open("/home/manhnguyen/new/co-separation/hai_sample_hfd5/val_hdf5.txt", 'w')
for line in img_txt:
    line = line.replace("\n", "")
    if "." in line:
       continue
    x = all_file[int(id_list.index(line))]
    print(x)
    img_PATH.write("%s\n" % (x))
    
