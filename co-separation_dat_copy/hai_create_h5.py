import numpy as np
import h5py
import os

path = "/home/manhnguyen/new/co-separation/hai_sample_hfd5/val.txt"
npylist = []
img_txt = open(path, "r")
for line in img_txt:
    line = line.replace("\n", "")
    if line.replace(' ', '') == '':
        continue   
    npylist.append(line)
file = h5py.File("/home/manhnguyen/new/co-separation/hai_sample_hfd5/val.hdf5",'w')
dt = h5py.special_dtype(vlen=str)
path = file.create_dataset('detection',(int(len(npylist)),),dtype=dt)
for i in range(len(npylist)):
    path[i] = npylist[i]